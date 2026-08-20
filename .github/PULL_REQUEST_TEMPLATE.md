## Summary
<!-- One sentence: what does this PR do? -->

## Motivation
<!-- Linked issue or paper section, plus a sentence on why now. -->

Closes #<issue>

## Changes
<!-- Bullet list of meaningful changes. -->

-
-

## Reproduction / verification

How a reviewer can verify this PR:

```bash
# minimum command(s) to exercise the new code
```

Expected behavior:
<!-- mIoU number, output snapshot, etc. -->

## Checklist

- [ ] `ruff check src/ tests/ scripts/` passes (the exact lint-CI command)
- [ ] `pytest tests/ -v` passes (`<N>/<N>`); CI runs the same suite minus
      `tests/test_tau_invariance.py`, which needs torch
- [ ] If quoting the BEV secondary-task number: the 100-seeded-frame protocol is stated
      alongside it
- [ ] If touching the eval path: `python scripts/eval.py eval/val_1step --checkpoint ... → 38.54`
- [ ] If touching docs/README: rendered locally and links resolve
- [ ] No new `[URL]` placeholders introduced
- [ ] No `print()` added to library code (use `logger`)
- [ ] No Claude / AI-tooling attribution in commits

## Reviewer notes
<!-- Anything you want the reviewer to focus on or look out for. -->
