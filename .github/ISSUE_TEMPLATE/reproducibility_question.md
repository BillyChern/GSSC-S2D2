---
name: Reproducibility question
about: A paper-claimed number you can't reproduce
title: "[repro] cannot match Tab. X / Fig. Y"
labels: reproducibility
assignees: BillyChern
---

## Which number?
<!-- e.g. "Tab. I 38.54% val mIoU at 1-step S2D2 correction sampling" or "Tab. XV 36.09% BEV". -->

## Reproduction command

```bash
# Exact command from the README / docs/REPRODUCIBILITY.md you ran:
python scripts/eval.py eval/val_1step \
    --checkpoint data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

## Result you got

| metric | paper | yours |
|---|---|---|
| mIoU | 38.54 | <X.XX> |
| Completion IoU | 52.66 | <X.XX> |

## Have you verified the asset hashes?

```bash
sha256sum data/checkpoints/gssc_mf/gssc_31k_mf_step40000/model_ema.safetensors
```

Expected: see [docs/MODEL_ZOO.md](../docs/MODEL_ZOO.md).

## Environment

* GPU:
* CUDA:
* `spconv-cu126` version:
* `torch` version:
* GSSC-S2D2 commit:

## Anything you changed?
<!-- Modified config? Different seed? Subset of val? -->

## Logs

<details><summary>Full eval log</summary>

```
<paste log here>
```

</details>
