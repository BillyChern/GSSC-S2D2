# Training recipes

All training runs use the same trainer (`gssc.training.train_scene_completion`)
driven by Hydra-style YAML configs.

> **Prerequisite for the scene-completion recipes on this page: the
> preprocessed 256³ voxel cache.** That is the headline, the single-frame sweep,
> the timestep ablations, both cross-base recipes and the BEV second task — i.e.
> everything driven by `scripts/train.py`. `gssc.training.train_scene_completion`
> does not read raw SemanticKITTI `.bin`/`.label` files; it reads
> `data/SemanticKITTI_3D/256/<seq>/<frame>_{voxels,bev,gt_scene}.npy`, which
> `scripts/download_assets.py` does **not** provision (the hosted assets are
> checkpoints, base predictions and the synthetic pool). Build it once from your
> raw SemanticKITTI download:
>
> ```bash
> python scripts/prepare_data.py --root data/SemanticKITTI          # verify the raw layout
> python scripts/prepare_256_data.py --semantickitti_root data/SemanticKITTI
> ```
>
> Budget for it: measured on the staged cache, the eleven annotated sequences
> `00`-`10` are 69,603 files / 92.1 GiB / 98.8 GB (three `.npy` per frame), of
> which val seq `08` alone is 12,213 files / 16.2 GiB / 17.3 GB. Add
> 127.1 GiB / 136.5 GB if you also stage the 31K synthetic pool as
> `256/synthetic/`.
>
> Without it those recipes enumerate zero samples and the trainer aborts with
> `MissingVoxelCacheError`, which names exactly this command.
> **The two *pyramid* recipes further down do not read this tree at all** — they
> have their own, different prerequisite (a pre-quantized corpus; see "Pyramid
> diffusion" below). See `docs/DATASET.md` for the raw download itself.

## Headline (the paper's 38.54%/39.2% number)

```bash
python scripts/train.py train/31k_mf --gpu 0
```

* 100K iterations
* Batch size 4 on **one** GPU
* AdamW, lr 1e-4, no warmup
* Loss: KL posterior + Lovász (0.3) + auxiliary (5e-4)
* Eval every 5K steps with N=100 S²D² correction sampling

> **The released scene-completion trainer is single-GPU.**
> `gssc.training.train_scene_completion` implements no `DataParallel`,
> no `DistributedDataParallel` and no `torch.distributed`; `scripts/train.py`
> launches one plain subprocess and pins one device. Passing `--gpu 0,1` only
> sets `CUDA_VISIBLE_DEVICES=0,1` — the second card is made visible and then
> left idle. So `batch_size: 4` is four samples on one 80 GB card, not "2 per
> GPU across 2× H100", and no wall-clock halving is available. The paper
> describes the original research codebase, which is not the code released
> here; every recipe below therefore uses `--gpu 0`. (The *pyramid* trainers,
> a separate entry point, do implement DDP.)

> **The `_mf` in `31k_mf` needs a second cache this release does not build, and
> its absence is silent.** `s3_mode: teacher` makes the loader look for
> preprocessed multi-frame LiDAR at
> `data/SemanticKITTI_3D/256_multi_frame/<seq>/<frame>.npz`
> (`SemanticKITTIDataset`'s `multi_frame_dir` default,
> `src/gssc/data/semantickitti.py:60`). **No script in this release produces that
> tree** — `grep -rln 256_multi_frame scripts/ src/` returns only its two
> consumers, and `scripts/prepare_256_data.py` builds the single-frame 256³ cache
> only. When the `.npz` is missing *and* `scpnet_pred_dir` is set (as it is in
> every shipped `_mf` config), the frame is kept rather than skipped
> (`semantickitti.py:199-202`, "Cold diffusion doesn't need multi-frame") and
> `__getitem__` falls back to the single-frame voxels
> (`semantickitti.py:419-422`) **with no warning on that branch**. So the command
> above will train to completion on single-frame LiDAR while the recipe name and
> the trainer banner say multi-frame. If you are reproducing the multi-frame
> headline, stage `256_multi_frame/` yourself (per frame: an `.npz` holding a
> `coords` array of occupied `(x, y, z)` voxel indices in the 256×256×32 grid,
> accumulated over the neighbouring sweeps) and check that
> `data/SemanticKITTI_3D/256_multi_frame/08/000000.npz` exists before you launch.
> The single-frame `_sf` recipes below are unaffected.

Cost: the paper's compute table prices this run at ~37 GPU-hours **to reach
step 40000** — the released checkpoint (`gssc_31k_mf_step40000`) and the figure
it calls the S²D² headline. Because the released trainer occupies one GPU,
read that as wall-clock on a single H100 80 GB; earlier revisions of this line
promised ≈18.5 h by splitting it across two GPUs, which this code cannot do.
The config above runs to 100000 iterations; letting it finish costs ~90
GPU-hours at the same observed ~0.9 GPU-h per 1K steps. Stopping at 40K is what
reproduces the paper.
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

The pyramid is trained once, offline, before scene-completion training. **It
does not run from a bare release checkout**, and neither module below should be
launched bare: `train_pyramid_s2`'s `--data-root` and `train_pyramid_s3`'s
`--ssc-root` / `--quantized-root` default to a `datasets/` tree
(`datasets/SemanticKITTI_quantized`, `datasets/dataset_SemanticKITTI_SSC`) that
nothing in this release creates, and with those defaults the dataset loads 0
frames and the DataLoader raises `num_samples=0` within a few seconds. Going
through `scripts/train.py` (below) avoids that, because the dispatcher supplies
both roots from `--data-root`.

**Prerequisite — the pre-quantized corpus.** Both stages read a coarse-resolution
corpus quantized from SemanticKITTI, which is *not* the same tree as
`data/SemanticKITTI/` and is *not* provisioned by `scripts/download_assets.py`.
The layout the loaders require is:

```
data/SemanticKITTI_quantized/
├── s1/sequences/{00..10}/{frame}.npy       # 32²×4   (S2 conditioning)
├── s2/sequences/{00..10}/{frame}.npy       # 64²×8   (S2 target, S3 conditioning)
└── s3_cond/sequences/{00..10}/{frame}_sub{0..3}.npy   # optional; S3 falls
                                                     # back to online upsampling
```

S3 additionally reads the full-resolution `.label` grids straight from the raw
SSC tree (`data/SemanticKITTI/sequences/{seq}/voxels/*.label`), so
`--ssc-root data/SemanticKITTI` is the raw download documented in
`docs/DATASET.md`. **No script in this release produces the quantized corpus**
— `grep -rni quantiz scripts/*.py` hits only `scripts/train.py`, which consumes
the tree and never builds it — so the pyramid **stages are not reproducible from
a bare release checkout**. Only their trained checkpoints ship
(`pyramid/pyramid_s1|s2|s3` in `docs/MODEL_ZOO.md`), and the screened synthetic
pools ship as data, so no paper number depends on retraining them. If a later
revision adds a quantizer, this paragraph is the one to replace with its
command.

With the corpus in place, prefer the shipped configs. They pin the recipe
explicitly — batch size (S2 `16`, S3 `9`), `lr: 0.004`, `epochs: 1000`,
`warmup-epochs: 5` — so it survives a change to any module default, and they
carry the released checkpoints' recorded `epoch` / `global_step` / `source_run`
in their `_description`:

```bash
python scripts/train.py train/pyramid_s2 --data-root data --gpu 0
python scripts/train.py train/pyramid_s3 --data-root data --gpu 0
```

`scripts/train.py` appends the per-stage roots for you — S2 gets
`--data-root data/SemanticKITTI_quantized`; S3 gets
`--ssc-root data/SemanticKITTI --quantized-root data/SemanticKITTI_quantized` —
and `--gpu` is handled by the dispatcher as `CUDA_VISIBLE_DEVICES` rather than
forwarded. Add `--dry-run` to print the resolved module command without
launching it. The equivalent hand-rolled invocations, if you would rather name
every root yourself:

```bash
python -m gssc.training.train_pyramid_s2 \
    --data-root data/SemanticKITTI_quantized \
    --output-dir outputs/checkpoints/s2 \
    --batch-size 16 --lr 0.004 --epochs 1000 --gpu 0

python -m gssc.training.train_pyramid_s3 \
    --ssc-root data/SemanticKITTI \
    --quantized-root data/SemanticKITTI_quantized \
    --output-dir outputs/checkpoints/s3 \
    --batch-size 9 --lr 0.004 --gpu 0
```

Pass `--lr 0.004` explicitly, so the recipe survives a change to either
module's argparse default (both currently *are* `0.004`, so an omitted flag
gives the same run today). It is the rate both released pyramid checkpoints
record (`base_lr: 0.004` in
`pyramid/pyramid_s2/config.json` and `pyramid/pyramid_s3/config.json`), and the
paper's pyramid hyperparameter table gives `0.004` as shared across S1, S2 and
S3. The same table gives the epoch counts the released checkpoints reached —
S1 2,940, S2 1,000, S3 584 — which is why `--epochs 1000` is spelled out for S2
(its default is 10,000) and why the released S3 file is the best checkpoint at
epoch 584 rather than the end of its 1,000-epoch launch.

Cost: **wall-clock lower bounds, not GPU-hours.** The release publishes no
compute measurement for the pyramid, so the figures below are derived from the
checkpoint-file mtimes of the three runs that produced the released weights —
identified by digest, not by directory name: each stage's shipped `config.json`
records a `source_run` and a `source_sha256` that `sha256sum` reproduces on the
research-tree file. Those runs live under the research tree's
`outputs/checkpoints/` directory as `s1/s1_epoch_2940.pt`,
`s2/s2_epoch_1000.pt` and `s3_v2_lr004/best_miou.pt` — research-tree paths, not
release-payload ones; none of them ships in `data/checkpoints/`.

| Stage | First saved ckpt → released ckpt | Elapsed |
|---|---|---|
| S1 (32²×4, to epoch 2,940)  | 2025-12-01 11:06 → 2025-12-03 02:40 | **≈ 40 h** (1.7 d) |
| S2 (64²×8, to epoch 1,000)  | 2025-12-04 13:12 → 2025-12-05 13:43 | **≈ 25 h** (1.0 d) |
| S3 (256²×32, to epoch 584)  | 2025-12-22 02:22 → 2026-02-08 01:47 | **≈ 1,151 h** (48 d) |
| **Total** | | **≈ 1,216 h** (51 d) |

Read each figure as a **lower bound on the run's elapsed wall-clock** and an
**upper bound on its GPU-time** — never as GPU-hours. Three reasons:

* Each run's first saved checkpoint is epoch 10 (S1, S3) or epoch 100 (S2), so
  every run started *earlier* than the left column says. At each run's own
  observed epoch rate that is a further ≈0.1 h (S1), ≈2.7 h (S2) and ≈21 h (S3).
* Nothing records device exclusivity over these spans, so they cannot be
  converted to GPU-hours. They are elapsed calendar time on a machine whose
  other tenants, if any, went unrecorded — which is why they bound GPU-time from
  *above*, not below.
* S3 dominates the total (≈29× S1). Its rate is steady rather than bursty — the
  epoch-100 markers land 216.8 h, 189.3 h, 189.3 h and 189.4 h apart, i.e.
  ≈1.9 h/epoch throughout — but the span still includes whatever idle time a
  recorded resume left behind (`s3_v2_lr004/training_resumed.log`), and the run
  kept going past the released best to ≈epoch 733 before it stopped.

Do not read the S3 figure off the research tree's sibling `s3/` directory
(2025-12-06 → 12-21, ≈355 h). That is the superseded lr-0.006 run which the released
`pyramid/pyramid_s3/config.json` explicitly disowns in its `restaged_reason`;
the released weights come from `s3_v2_lr004/`.

The resolution is fixed per stage (S2 = 64³, S3 = 256³) inside each module, so
no `--resolution` flag is exposed; both accept only
`--data-root`/`--output-dir`/`--batch-size`/`--epochs`/`--lr`/`--gpu`/`--resume`/`--num-workers`/`--no-scale-lr`/`--warmup-epochs`
(S3 uses `--ssc-root`/`--quantized-root` in place of `--data-root`). Stage 1
(32³) is fast and can be merged into the S2 launcher. Unlike the
scene-completion trainer, these two entry points *do* implement DDP.

## JS3C-Net cross-base (paper tab:portable_s2d2; v1.1.0)

Requires the JS3C-Net predictions dataset (`docs/REPRODUCIBILITY.md` covers
the one-time dumper setup):

```bash
python scripts/train.py train/js3c_real --gpu 0
```

* Real-only sequences (00-07, 09, 10) — no synthetic pool (JS3C-Net's seg
  head misclassifies the voxel-derived fake point clouds as out-of-distribution
  and crashes the dumper on them; see paper supp § H and the "Known gap on the
  synthetic pool" section in `docs/REPRODUCIBILITY.md`).
* 100K iterations, batch size 4, lr 1e-4, ema_decay 0.9999.
* `cold_diffusion=true` (REQUIRED for cross-base — deterministic forward).
* Expected val mIoU at step 100K — **the paper's headline for this base is
  24.3 % (+1.6 pp over the 22.7 % base)**: derived BEV, scored by the official
  `semantic-kitti-api`, precise output **24.32 %**, reproduced by
  `eval/js3c_val_realistic`. The **same derived-BEV setting** read by the
  paper's internal training-time evaluator is **26.7 %** (precise **26.72 %**,
  +3.99 pp) — the supplement's continuity row. *The evaluator is what separates
  those two numbers, not the BEV source.* Separately, `eval/js3c_val_paper`
  (= `eval/js3c_val_1step`, `bev_source: gt`) prints a **GT-BEV diagnostic** of
  ours at **26.05 %** under the official api. The paper does not print it —
  neither "26.05" nor "26.1" appears anywhere in it — so it is not a rounding of
  anything the paper carries, and it is **not** the protocol behind 26.72.

Cost: ~90 GPU-hours to the full 100000 iterations — the figure the paper's
compute table prices the alt-base runs at, and one this run does reach (the
released `gssc_js3c_s2d2_real` checkpoint records `global_step: 100000`). The
headline is read off at step 40000 for ~37 GPU-h instead. The released trainer
occupies one GPU, so read GPU-hours here as wall-clock on a single H100 80 GB.
Output: `outputs/train_js3c_real/step_{5000,...,100000}.pt` + `best_miou.pt`.

## LMSCNet cross-base (paper tab:portable_s2d2, third base; v2.1.0)

Requires the LMSCNet predictions dataset (`docs/REPRODUCIBILITY.md` covers
the one-time dumper setup):

```bash
python scripts/train.py train/lmscnet_real --gpu 0
```

* Real-only sequences (00-07, 09, 10) — no synthetic pool. Mirrors the
  JS3C-Net real-only protocol exactly, differing only in `base_kind` and
  `base_pred_dir`.
* 100K iterations *as configured*, batch size 4, lr 1e-4, ema_decay 0.9999.
  The released checkpoint's own run stopped at 65K — see the expected-mIoU
  bullet below.
* `cold_diffusion=true` (REQUIRED for cross-base — deterministic forward).
* `bev_from_base=true` — the seed BEV is height-pooled from LMSCNet's own 3D
  prediction (never GT BEV), so the val number below is already an at-deploy,
  derived-BEV result.
* Expected val mIoU: **16.59 %** (paper rounds to 16.6; paper
  tab:portable_s2d2, +1.8 pp over the LMSCNet base 14.76 %, re-scored from
  on-disk predictions, superseding the earlier 12.10 base) under the official
  `semantic-kitti-api` evaluator. **This is not a step-100000 reading.** The
  released `gssc_lmsc_s2d2_real` checkpoint is the best-mIoU selection at
  **step 65,000** and records `global_step: 65000` in its own `config.json`,
  even though the config's `num_iterations` is 100000. Do not expect your own
  `step_100000.pt` to reproduce 16.59 exactly; compare against `best_miou.pt`.
* NOTE: the released LMSCNet `model_ema.safetensors` ships complete (278 tensors,
  45 BN buffers) and reproduces 16.59 directly; no full-state-checkpoint
  workaround is needed.

Cost: **budgeted, not spent.** The paper's compute table prices the alt-base
runs at ~90 GPU-h each for a full 100000-iteration launch, and that is what the
config above will cost you if you let it run to the end. The released LMSCNet
checkpoint is not that run: it stopped at step 65,000, and this release
publishes no measured cost for it, so treat ~90 GPU-h as the budget for the
full launch rather than the price of the shipped artifact. The released trainer
occupies one GPU, so read GPU-hours as wall-clock on a single H100 80 GB.
Output: `outputs/train_lmscnet_real/step_{5000,...,100000}.pt` + `best_miou.pt`
(the released checkpoint corresponds to `best_miou.pt`, taken at step 65000).

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
