#!/usr/bin/env python3
"""Pre-commit guard: refuse commits that attribute work to Claude / AI tooling.

The user prefers BillyChern as the sole repo contributor. This guard fails
the commit if the staged change set or commit message includes any of:

  - "Co-Authored-By: Claude"
  - "Generated with Claude Code"
  - "Co-authored-by: Anthropic"
  - "🤖" emoji on commit-message lines (a common AI-tool tell)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    re.compile(r"co-authored-by:\s*claude", re.IGNORECASE),
    re.compile(r"co-authored-by:\s*anthropic", re.IGNORECASE),
    re.compile(r"co-authored-by:\s*chatgpt", re.IGNORECASE),
    re.compile(r"co-authored-by:\s*openai", re.IGNORECASE),
    re.compile(r"generated with .*claude", re.IGNORECASE),
    re.compile(r"\U0001F916"),  # robot emoji
]


def _check_commit_message() -> list[str]:
    """Return list of issues found in COMMIT_EDITMSG (if present)."""
    msg_path = Path(".git/COMMIT_EDITMSG")
    if not msg_path.exists():
        return []
    text = msg_path.read_text(encoding="utf-8", errors="ignore")
    issues = []
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            issues.append(f"  - commit message matches: {pat.pattern!r}")
    return issues


def _check_staged_diff() -> list[str]:
    """Return list of issues found in the staged diff content."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--no-color"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return []
    issues = []
    for line_num, line in enumerate(out.splitlines(), 1):
        # Only check added lines
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for pat in FORBIDDEN_PATTERNS:
            if pat.search(line):
                issues.append(f"  - staged diff line {line_num} matches {pat.pattern!r}: {line[:80]}")
                break
    return issues


def main() -> int:
    issues = _check_commit_message() + _check_staged_diff()
    if issues:
        print("Refusing to commit: AI / Claude attribution detected.\n")
        for i in issues:
            print(i)
        print(
            "\nThis repository tracks BillyChern as the sole contributor.\n"
            "Remove the offending lines from the commit and try again.\n"
            "(See .pre-commit-config.yaml -> forbid-claude-attribution.)"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
