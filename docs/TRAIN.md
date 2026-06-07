# Training recipes

All training runs use the same trainer (`gssc.training.train_scene_completion`)
driven by Hydra-style YAML configs.

## Headline (the paper's 38.54%/39.2% number)

```bash
python scripts/train.py train/31k_mf --gpu 0,1
```

* 100K iterations
* Batch size 4 (effective ~8 with 2× H100)
* AdamW, lr 1e-4, no warmup
* Loss: KL posterior + Lovász (0.3) + auxiliary (5e-4)
* Eval every 5K steps with N=100 S²D² correction sampling

Wall-clock: ~37 hours on 2× H100 80 GB.
Output: `outputs/train_31k_mf/step_{5000,10000,...,100000}.pt` + `best.pt`.

## Data-scaling ablations (Tab. VII)

```bash
for cfg in 0K_sf 10K_sf 20K_sf 31k_sf 57K_sf; do
  python scripts/train.py train/$cfg --gpu 0
done
```

## Training-timestep ablations (Tab. XII)

```bash
for cfg in T10 T50 T100skewed; do
  python scripts/train.py train/$cfg --gpu 0
done
```

## Pyramid diffusion (offline data-aug pipeline)

The pyramid is trained once, offline, before scene-completion training:

```bash
python -m gssc.training.train_pyramid_s2 --resolution 64
python -m gssc.training.train_pyramid_s3 --resolution 256
```

Stage 1 (32^3) is fast and can be merged into the S2 launcher.

## JS3C-Net cross-base (paper tab:portable_s2d2; v1.1.0)

Requires the JS3C-Net predictions dataset (`docs/REPRODUCIBILITY.md` covers
the one-time dumper setup):

```bash
python scripts/train.py train/js3c_real --gpu 0,1
```

* Real-only sequences (00-07, 09, 10) — no synthetic pool (JS3C-Net's seg
  head OOMs on voxel-derived fake point clouds; see paper supp § H).
* 100K iterations, batch size 4, lr 1e-4, ema_decay 0.9999.
* `cold_diffusion=true` (REQUIRED for cross-base — deterministic forward).
* Expected val mIoU at step 100K: **26.72 %** (paper tab:portable_s2d2, +3.99 pp
  over the JS3C-Net base 22.73 %).

Wall-clock: ~37 hours on 2× H100 80 GB (identical to the headline 31k_mf run).
Output: `outputs/train_js3c_real/step_{5000,...,100000}.pt`.

## LMSCNet cross-base (paper tab:portable_s2d2, third base; v2.1.0)

Requires the LMSCNet predictions dataset (`docs/REPRODUCIBILITY.md` covers
the one-time dumper setup):

```bash
python scripts/train.py train/lmscnet_real --gpu 0,1
```

* Real-only sequences (00-07, 09, 10) — no synthetic pool. Mirrors the
  JS3C-Net real-only protocol exactly, differing only in `base_kind` and
  `base_pred_dir`.
* 100K iterations, batch size 4, lr 1e-4, ema_decay 0.9999.
* `cold_diffusion=true` (REQUIRED for cross-base — deterministic forward).
* `bev_from_base=true` — the seed BEV is height-pooled from LMSCNet's own 3D
  prediction (never GT BEV), so the val number below is already an at-deploy,
  derived-BEV result.
* Expected val mIoU at step 100K: **16.59 %** (paper tab:portable_s2d2, +4.49 pp
  over the LMSCNet base 12.10 %) under the official `semantic-kitti-api` evaluator.

Wall-clock: ~37 hours on 2× H100 80 GB (identical to the headline 31k_mf run).
Output: `outputs/train_lmscnet_real/step_{5000,...,100000}.pt`.

## BEV second task

```bash
python scripts/train.py train/bev_secondary --gpu 0
```

## Resume

The trainer accepts `--resume <checkpoint>` (forwarded through
`scripts/train.py` to `gssc.training.train_scene_completion`); it restores the
model, optimizer, scheduler, and step counter:

```bash
python scripts/train.py train/31k_mf --resume outputs/train_31k_mf/step_50000.pt
```
