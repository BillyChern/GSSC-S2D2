---
name: Bug report
about: Something is broken or behaves unexpectedly
title: "[bug] "
labels: bug
assignees: BillyChern
---

## Summary
<!-- One sentence: what is broken? -->

## Reproduction

```bash
# Exact commands you ran. Replace your local paths with the ones from
# the README quick-start so we can reproduce on a fresh clone.
git clone https://github.com/BillyChern/GSSC-S2D2.git
cd GSSC-S2D2
uv venv --python 3.10 && uv sync && uv pip install spconv-cu126==2.3.8
source .venv/bin/activate   # otherwise `python` below is the system interpreter
# ... your commands ...
```

## Expected vs actual

* **Expected:** <!-- what should have happened -->
* **Actual:** <!-- what happened -->

## Logs / traceback

<details><summary>Full stderr</summary>

```
<paste the full traceback here, not just the last line>
```

</details>

## Environment

Run `python -c "import torch, sys; print(sys.version, torch.__version__, torch.version.cuda)"`
(from the activated `.venv`, or as `uv run python -c ...`)
and paste the output:

```
<paste here>
```

* OS:
* GPU:
* spconv version (`pip show spconv-cu126`):
* GSSC-S2D2 commit (`git rev-parse HEAD`):

## Have you reproduced on a fresh clone?

- [ ] Yes, the bug reproduces on a fresh `uv sync` of `main`
- [ ] No, only on my modified working tree

## Anything else
<!-- Optional: screenshots, related issues, hypotheses -->
