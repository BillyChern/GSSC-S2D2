---
name: Reproducibility question
about: A paper-claimed number you can't reproduce
title: "[repro] cannot match Tab. X / Fig. Y"
labels: reproducibility
assignees: BillyChern
---

## Which number?
<!-- e.g. "Tab. I 38.54 % val mIoU at 1-step S2D2 correction sampling". -->

> **Before filing a BEV secondary-task (main paper § IV-D, 34.8 -> 36.1) mismatch:** that row is
> measured with the *training-time 2D BEV evaluator on 100 fixed val samples (seed 42)*,
> **not** the 4071-frame semantic-kitti-api protocol, and it comes from
> `data/checkpoints/bev/bev_s2d2_scpnet/`. A full-val number is not comparable to it,
> and neither is a `--max-frames 100` run: the published sample is a seeded draw over a
> differently-filtered frame list. See `docs/MODEL_ZOO.md` and the checkpoint's own
> `config.json`; the supplement states the same protocol in prose next to Tab. XXI.

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

`checksums.txt` ships at the root of the Hugging Face checkpoints repo and the download
unpacks it into `data/checkpoints/`, so this is a verdict, not a digest you have to
eyeball. Paths inside it are relative to `checkpoints/`, so run it from INSIDE that
directory — from `data/` every line FAILs open-or-read even on a perfect download:

```bash
cd data/checkpoints && sha256sum -c checksums.txt
```

- [ ] every line printed `OK`

Per-file digests are also tabulated in [docs/MODEL_ZOO.md](../../docs/MODEL_ZOO.md).

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
