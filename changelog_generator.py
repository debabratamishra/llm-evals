#!/usr/bin/env python3
"""
changelog-generator.py — Generate CHANGELOG.md from git history.

Produces a markdown changelog from conventional commits between two refs.
Uses 'git log' and parses commit messages by prefix.

Usage:
    uv run python changelog-generator.py > CHANGELOG.md
    uv run python changelog-generator.py --from v0.1.0 --to main > CHANGELOG.md

Outputs to stdout. Pipe into CHANGELOG.md to overwrite.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import UTC, datetime


def git(*args: str) -> str:
    """Run a git command and return trimmed stdout. Dies on error."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(
            f"fatal: git {' '.join(args)} failed:\n{e.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)


def get_commits(since: str, to: str) -> list[dict]:
    """Get parsed commits in the range (since..to], ordered oldest-first."""
    raw = git(
        "log",
        f"{since}..{to}",
        "--format=%H%n%ae%n%aI%n%s%n---%n",
        "--reverse",
    )
    if not raw:
        return []

    blocks = raw.split("\n---\n")
    commits = []
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            continue

        sha = lines[0]
        body = "\n".join(lines[3:])
        msg = lines[3] if len(lines) > 3 else ""
        commits.append(
            {
                "sha": sha[:7],
                "author_email": lines[1],
                "date": lines[2],
                "message": msg,
                "full_body": body,
                "is_breaking": "BREAKING" in body.upper() or "BREAKING" in msg.upper(),
            }
        )
    return commits


def classify(message: str) -> tuple[str, str]:
    """Classify a commit message into (category, description).

    Returns ('uncategorized', message) if no prefix is found.
    """
    # Patterns: type(scope): description
    patterns = [
        r"^(?P<type>feat|fix|docs|test|refact|refactor|perf|chore|style|ci|build|deps)(?:\((?P<scope>[^)]*)\))?:\s*(?P<desc>.*)$",
    ]
    for pat in patterns:
        m = re.match(pat, message)
        if m:
            cat = m.group("type")
            desc = m.group("desc") or message
            return cat, desc.strip()

    # Fallback: try to extract a prefix
    m = re.match(r"^(\w[\w-]*)\s*:\s*(.*)$", message)
    if m:
        return m.group(1).lower(), m.group(2).strip()

    return "other", message.strip()


CATEGORY_ORDER = [
    ("feat", "🚀 Features"),
    ("fix", "🐛 Bug Fixes"),
    ("security", "🔒 Security"),
    ("perf", "⚡ Performance"),
    ("refact", "♻️ Refactoring"),
    ("refactor", "♻️ Refactoring"),
    ("test", "✅ Tests"),
    ("docs", "📖 Documentation"),
    ("deps", "📦 Dependencies"),
    ("ci", "🔧 CI/CD"),
    ("build", "🏗️ Build"),
    ("chore", "🧹 Chores"),
    ("style", "💄 Style"),
    ("other", "📝 Other Changes"),
]


def format_changelog(
    version: str,
    commits: list[dict],
    previous_tag: str | None = None,
    date_str: str | None = None,
) -> str:
    """Format a complete markdown changelog section."""
    if date_str is None:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")

    lines: list[str] = []

    # Section header
    if version.lower() == "unreleased":
        lines.append(f"## [Unreleased]")
    else:
        lines.append(f"## [v{version}] - {date_str}")

    # Count breakdown
    breaking_count = sum(1 for c in commits if c["is_breaking"])
    total = len(commits)
    if total > 0:
        lines.append(
            f"\n_{total} commits_{' · ' + str(breaking_count) + ' breaking' if breaking_count else ''}_\n"
        )

    # Group by category
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for cat_key, _ in CATEGORY_ORDER:
        grouped[cat_key] = []

    # Merged categories
    category_map: dict[str, str] = {}
    for cat_key, cat_label in CATEGORY_ORDER:
        category_map[cat_key] = cat_label

    for c in commits:
        cat, desc = classify(c["message"])
        if c["is_breaking"]:
            cat = "security"  # breaking changes go top
        # Normalize refact/refactor
        if cat in ("refactor",):
            cat = "refact"
        if cat not in category_map:
            cat = "other"
        entry = {
            "sha": c["sha"],
            "desc": desc,
            "is_breaking": c["is_breaking"],
            "date": c["date"][:10],
        }
        grouped[cat].append(entry)

    # Emit categories
    for cat_key, cat_label in CATEGORY_ORDER:
        entries = grouped.get(cat_key, [])
        if not entries:
            continue
        # Skip breaking-only headers if there are none
        if cat_key == "security" and not any(e["is_breaking"] for e in entries):
            # Actually, just show normally
            pass

        lines.append(f"\n### {cat_label}\n")
        for e in entries:
            breaking_marker = " **[BREAKING]**" if e["is_breaking"] else ""
            lines.append(
                f"- {e['desc']}{breaking_marker} ([`{e['sha']}`](https://github.com/debabratamishra/llm-evals/commit/{e['sha']}))"
            )

    return "\n".join(lines)


def get_latest_tag() -> str | None:
    """Get the most recent version tag, or None."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
            capture_output=True,
            text=True,
            check=True,
        )
        tag = result.stdout.strip()
        return tag if tag.startswith("v") else None
    except subprocess.CalledProcessError:
        return None


def first_commit_sha() -> str:
    """Get the SHA of the first commit."""
    return git("rev-list", "--max-parents=0", "HEAD")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CHANGELOG.md section from git history."
    )
    parser.add_argument(
        "--from",
        dest="since",
        help="Start ref (default: latest tag, or first commit if no tags)",
        default=None,
    )
    parser.add_argument(
        "--to",
        dest="to",
        help="End ref (default: HEAD)",
        default="HEAD",
    )
    parser.add_argument(
        "--version",
        help="Version string for the header (default: auto-detect)",
        default=None,
    )
    parser.add_argument(
        "--date",
        help="Release date (default: today)",
        default=None,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate the full changelog from first commit (not just current range)",
    )
    args = parser.parse_args()

    since = args.since
    if since is None and not args.full:
        latest = get_latest_tag()
        if latest:
            since = latest
        else:
            # No tags yet — use first commit
            since = first_commit_sha()
            print(
                "info: no tags found, generating changelog from first commit",
                file=sys.stderr,
            )
    elif args.full:
        since = first_commit_sha()

    commits = get_commits(since, args.to)
    version = args.version

    if version is None:
        # Try to derive from the 'since' tag
        if since and since.startswith("v"):
            # This is the tag we're *after* — derive next version
            # Or if this is the first release, use 0.1.0
            latest = get_latest_tag()
            if latest and latest.startswith("v"):
                # We're showing changes since latest -> "[Unreleased]"
                version = "Unreleased"
            else:
                version = "0.1.0"
        elif args.full:
            version = "0.1.0"
        else:
            version = "Unreleased"

    # If there's a latest tag and we're comparing against it, show Unreleased
    latest = get_latest_tag()
    if latest and args.since is None and not args.full:
        if since == latest:
            version = "Unreleased"

    output = format_changelog(
        version=version,
        commits=commits,
        previous_tag=since if since and since.startswith("v") else None,
        date_str=args.date,
    )
    print(output)


if __name__ == "__main__":
    main()