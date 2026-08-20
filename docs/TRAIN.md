# Training recipes

All training runs use the same trainer (`gssc.training.train_scene_completion`)
driven by Hydra-style YAML configs.

## Headline (the paper's 38.54%/39.2% number)

```bash
python scripts/train.py train/31k_mf --gpu 0,1
```

* 100K iterations
* Effective batch size 4 (2 per GPU across 2× H100)
* AdamW, lr 1e-4, no warmup
* Loss: KL posterior + Lovász (0.3) + auxiliary (5e-4)
* Eval every 5K steps with N=100 S²D² correction sampling

Cost: ~37 GPU-hours on 2× H100 80 GB (≈18.5 h wall-clock at 2 GPUs) **to reach
step 40000** — the released checkpoint (`gssc_31k_mf_step40000`) and the figure
the paper's compute table prices as the S²D² headline. The config above runs to
100000 iterations; letting it finish costs ~90 GPU-hours at the same observed
~0.9 GPU-h per 1K steps. Stopping at 40K is what reproduces the paper.
Output: `outputs/train_31k_mf/step_{5000,10000,...,100000}.pt` + `best.pt`.

## Single-frame data-scaling companion sweep (NOT Tab. VII)

```bash
for cfg in 0K_sf 10K_sf 20K_sf 31k_sf 57K_sf; do
  python scripts/train.py train/$cfg --gpu 0
done
```

These are the **single-frame** retrains (single-frame LiDAR at both training and
deployment). The paper reports their sweep as prose in supplementary App. C-B,
not as a table. They are a companion to, not the source of, `tab:data_scaling`
(supplementary Tab. VII), which is the **multi-frame** sweep and ships no
per-row checkpoints — the supplement states this explicitly ("It is not Table
VII, which is the multi-frame full-val sweep"). See `docs/MODEL_ZOO.md`
("Single-frame data-scaling companion sweep") and `docs/REPRODUCIBILITY.md`.

## Training-timestep ablations (internal; no paper table)

```bash
for cfg in T10 T50 T100skewed; do
  python scripts/train.py train/$cfg --gpu 0
done
```

The paper prints no timestep-ablation table, and the label earlier revisions of
these docs cited for one does not exist in it.
Supplementary App. C-A discusses only the two `T=100` schedules, uniform against
`t=T`-skewed, and states that the shorter schedules were trained and omitted
rather than reported, because β stays linear on [1e-4, 0.1] irrespective of `T`.
The `T10` / `T50` runs above are therefore internal ablations.

## Pyramid diffusion (offline data-aug pipeline)

The pyramid is trained once, offline, before scene-completion training:

```bash
python -m gssc.training.train_pyramid_s2
python -m gssc.training.train_pyramid_s3
```

The resolution is fixed per stage (S2 = 64³, S3 = 256³) inside each module, so
no `--resolution` flag is exposed; both accept only
`--data-root`/`--output-dir`/`--batch-size`/`--epochs`/`--lr`/`--gpu`/`--resume`/`--num-workers`/`--no-scale-lr`/`--warmup-epochs`
(S3 uses `--ssc-root`/`--quantized-root` in place of `--data-root`). Stage 1
(32³) is fast and can be merged into the S2 launcher.

## JS3C-Net cross-base (paper tab:portable_s2d2; v1.1.0)

Requires the JS3C-Net predictions dataset (`docs/REPRODUCIBILITY.md` covers
the one-time dumper setup):

```bash
python scripts/train.py train/js3c_real --gpu 0,1
```

* Real-only sequences (00-07, 09, 10) — no synthetic pool (JS3C-Net's seg
  head misclassifies the voxel-derived fake point clouds as out-of-distribution
  and crashes the dumper on them; see paper supp § H and the "Known gap on the
  synthetic pool" section in `docs/REPRODUCIBILITY.md`).
* 100K iterations, batch size 4, lr 1e-4, ema_decay 0.9999.
* `cold_diffusion=true` (REQUIRED for cross-base — deterministic forward).
* Expected val mIoU at step 100K: **26.05 %** — a GT-BEV DIAGNOSTIC under the
  official `semantic-kitti-api`, **not** the paper's headline for this base. The
  paper's JS3C-Net headline is **24.3 % (+1.6 pp)**, derived BEV, which is what
  `eval/js3c_val_realistic` reproduces. The string "26.1" appears nowhere in the
  paper or its supplement, so it is not a rounding of anything the paper prints. The same protocol under the paper's internal
  training-time evaluator reads 26.72 % (+3.99 pp, a continuity row); the
  reproducible at-deploy derived-BEV number is 24.32 % (+1.59 pp) under the
  official `semantic-kitti-api`.

Cost: ~90 GPU-hours on 2× H100 80 GB (≈45 h wall-clock). This run trains to the full 100000 iterations, unlike the headline, which is read off at step 40000 for ~37 GPU-h; the paper's compute table prices the alt-base runs at ~90 GPU-h each.
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
* Expected val mIoU at step 100K: **16.59 %** (paper rounds to 16.6; paper
  tab:portable_s2d2, +1.8 pp over the LMSCNet base 14.76 %, re-scored from
  on-disk predictions, superseding the earlier 12.10 base) under the official
  `semantic-kitti-api` evaluator.
* NOTE: the released LMSCNet `model_ema.safetensors` ships complete (278 tensors,
  45 BN buffers) and reproduces 16.59 directly; no full-state-checkpoint
  workaround is needed.

Cost: ~90 GPU-hours on 2× H100 80 GB (≈45 h wall-clock). This run trains to the full 100000 iterations, unlike the headline, which is read off at step 40000 for ~37 GPU-h; the paper's compute table prices the alt-base runs at ~90 GPU-h each.
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
