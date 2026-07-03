#!/usr/bin/env python3
"""
ci-bump-version.py — Run by CI to derive the next semver version.

Reads the latest v* tag, bumps the patch digit.
If no tag exists, returns 0.1.0.
"""
from __future__ import annotations

import subprocess


def main():
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
        capture_output=True,
        text=True,
    )
    tag = result.stdout.strip()
    if tag.startswith("v"):
        parts = tag[1:].split(".")
        v = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
    else:
        v = "0.1.0"
    print(v, end="")


if __name__ == "__main__":
    main()