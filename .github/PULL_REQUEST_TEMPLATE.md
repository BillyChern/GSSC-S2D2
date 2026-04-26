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

- [ ] `ruff check src/ scripts/ tests/` passes
- [ ] `pytest tests/` passes (`<N>/<N>`)
- [ ] If touching the eval path: `python scripts/eval.py eval/val_1step --checkpoint ... → 38.54`
- [ ] If touching docs/README: rendered locally and links resolve
- [ ] No new `[URL]` placeholders introduced
- [ ] No `print()` added to library code (use `logger`)
- [ ] No Claude / AI-tooling attribution in commits

## Reviewer notes
<!-- Anything you want the reviewer to focus on or look out for. -->
