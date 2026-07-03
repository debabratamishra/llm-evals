#!/bin/bash
# Install the conventional-commit git hook into .git/hooks/commit-msg
set -euo pipefail

HOOK_SOURCE="$(dirname "$0")/commit-msg-hook"
HOOK_TARGET="$(git rev-parse --git-dir 2>/dev/null)/hooks/commit-msg"

if [ ! -f "$HOOK_SOURCE" ]; then
    echo "Error: source hook $HOOK_SOURCE not found"
    exit 1
fi

cp "$HOOK_SOURCE" "$HOOK_TARGET"
chmod +x "$HOOK_TARGET"
echo "✓ Installed conventional-commit hook at $HOOK_TARGET"