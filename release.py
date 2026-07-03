#!/usr/bin/env python3
"""
release.py — Bump version, tag, and generate changelog in one shot.

Handles the full release workflow:
  1. Detect what kind of release this is (patch/minor/major) from commits.
  2. Bump version in pyproject.toml.
  3. Generate CHANGELOG.md with the new version section.
  4. Create and push a git tag.

Usage:
    uv run python release.py          # auto-detect bump from commits
    uv run python release.py patch    # force patch bump
    uv run python release.py minor    # force minor bump
    uv run python release.py major    # force major bump

Dry-run mode (no tags, no file changes):
    uv run python release.py --dry-run

Auto-detect rules (from conventional commits since last tag):
    - feat: → minor
    - fix:, test:, refact: → patch
    - BREAKING in any message → major
    - mix of all three → major
"""

from __future__ import annotations

import argparse
import enum
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


# ── helpers ──────────────────────────────────────────────────────────────────


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"fatal: git {' '.join(args)} failed:\n{e.stderr.strip()}", file=sys.stderr)
        sys.exit(1)


def read_pyproject_version() -> str:
    """Read current version from pyproject.toml."""
    content = PYPROJECT_TOML.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not m:
        print("fatal: could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def write_pyproject_version(version: str) -> None:
    """Update version in pyproject.toml."""
    content = PYPROJECT_TOML.read_text()
    new_content = re.sub(
        r'^version\s*=\s*"[^"]+"',
        f'version = "{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    PYPROJECT_TOML.write_text(new_content)


def get_latest_tag() -> str | None:
    """Return the latest v* tag, or None."""
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
            capture_output=True,
            text=True,
            check=True,
        )
        return tag.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def get_commits_since(tag: str | None) -> list[str]:
    """Get commit messages from tag..HEAD (or from first commit if no tag)."""
    since = tag if tag else git("rev-list", "--max-parents=0", "HEAD")
    raw = git("log", f"{since}..HEAD", "--reverse", "--format=%s%n%b===")
    if not raw:
        return []
    return raw.split("\n===\n")


# ── conventional commit classification ───────────────────────────────────────


class Bump(enum.Enum):
    """Semver bump level."""
    PATCH = 0
    MINOR = 1
    MAJOR = 2


def classify_bump(commits: list[str]) -> Bump:
    """Walk all commit messages and derive the required semver bump.

    Highest priority wins: one BREAKING → MAJOR.
    """
    bump = Bump.PATCH
    for msg in commits:
        lower = msg.lower()

        if "BREAKING" in msg.upper() or "BREAKING" in lower:
            return Bump.MAJOR  # short-circuit

        # feat → minor (unless it's also breaking, already caught above)
        if re.match(r"feat(\(.*\))?:", lower):
            if bump.value < Bump.MINOR.value:
                bump = Bump.MINOR

        # fix / test / refact / deps / chore → patch
        # (patch is the default, so nothing to escalate for these)

    return bump


def semver_bump(current: str, bump_type: Bump) -> str:
    """Apply the bump to a semver string like '0.1.0'."""
    parts = [int(x) for x in current.split(".")]
    while len(parts) < 3:
        parts.append(0)
    major, minor, patch = parts[:3]

    if bump_type == Bump.MAJOR:
        return f"{major + 1}.0.0"
    elif bump_type == Bump.MINOR:
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Cut a new release: bump version, tag, generate changelog."
    )
    parser.add_argument(
        "level",
        nargs="?",
        choices=["patch", "minor", "major"],
        help="Override auto-detected bump level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes",
    )
    args = parser.parse_args()

    # ── 1. Determine the bump type ─────────────────────────────────────
    latest_tag = get_latest_tag()
    current_version = read_pyproject_version()

    if latest_tag is None:
        # First release — use the version in pyproject.toml as-is
        new_version = current_version
        bump_label = "initial"
        reason = "first release (no prior tags)"
    elif args.level:
        # Forced level
        bump_type = Bump[args.level.upper()]
        new_version = semver_bump(current_version, bump_type)
        bump_label = bump_type.name.lower()
        reason = f"manually specified --{args.level}"
    else:
        # Auto-detect from commits since last tag
        commits = get_commits_since(latest_tag)
        if not commits:
            print("No new commits since last release. Nothing to do.", file=sys.stderr)
            sys.exit(0)

        bump_type = classify_bump(commits)
        new_version = semver_bump(current_version, bump_type)
        bump_label = bump_type.name.lower()
        reason = f"auto-detected from {len(commits)} commit(s)"

    print(f"Current version: {current_version}")
    print(f"Bump type:       {bump_label}  ({reason})")
    print(f"New version:     {new_version}")
    print()

    # ── 2. Dry-run exit ────────────────────────────────────────────────
    if args.dry_run:
        print("[dry-run] Would apply changes — no files modified, no tag created.")
        sys.exit(0)

    # ── 3. Update pyproject.toml ────────────────────────────────────────
    write_pyproject_version(new_version)
    git("add", str(PYPROJECT_TOML))
    print(f"✓ Updated pyproject.toml to v{new_version}")

    # ── 4. Generate changelog ───────────────────────────────────────────
    sys.path.insert(0, str(REPO_ROOT))
    from changelog_generator import get_commits as cl_commits
    from changelog_generator import format_changelog as cl_format
    from changelog_generator import get_latest_tag as cl_latest

    since_tag = cl_latest()
    if since_tag:
        commits_dict = cl_commits(since_tag, "HEAD")
    else:
        # First release — get all commits
        first = git("rev-list", "--max-parents=0", "HEAD")
        commits_dict = cl_commits(first, "HEAD")

    section = cl_format(
        version=new_version,
        commits=commits_dict,
        previous_tag=since_tag,
        date_str=datetime.now(UTC).strftime("%Y-%m-%d"),
    )
    print(section)

    # ── 5. Update CHANGELOG.md ──────────────────────────────────────────
    changelog_path = REPO_ROOT / "CHANGELOG.md"
    header = "# Changelog\n\nAll notable changes to this project are documented here.\n\n"

    # Prepend the new section
    existing = changelog_path.read_text() if changelog_path.exists() else ""
    if existing:
        # Remove the old header if it exists so we don't double it
        lines = existing.split("\n")
        # Find first section that starts with ##
        body_start = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                body_start = i
                break
        if body_start > 0:
            existing = "\n".join(lines[body_start:])
        else:
            existing = "\n".join(lines[1:])  # skip "# Changelog"
        existing = existing.strip()

    new_changelog = header + section
    if existing:
        new_changelog += "\n\n" + existing

    changelog_path.write_text(new_changelog)
    git("add", str(changelog_path))
    print(f"✓ Updated {changelog_path.name}")

    # ── 6. Commit the version bump ──────────────────────────────────────
    git("commit", "-m", f"chore: bump version to v{new_version}")
    print(f"✓ Committed version bump")

    # ── 7. Tag ──────────────────────────────────────────────────────────
    tag_name = f"v{new_version}"
    git("tag", "-a", tag_name, "-m", f"Release {tag_name}")
    print(f"✓ Created tag {tag_name}")

    # ── 8. Push ─────────────────────────────────────────────────────────
    print()
    print(f"Run the following to push the release:")
    print(f"  git push origin main && git push origin {tag_name}")


if __name__ == "__main__":
    main()