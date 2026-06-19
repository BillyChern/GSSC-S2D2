# Datasets

## SemanticKITTI raw data (required, ~80 GB)

Download from the [SemanticKITTI website](http://www.semantic-kitti.org/dataset.html#download):

1. **Velodyne point clouds** (`data_odometry_velodyne.zip`, 80 GB)
2. **SemanticKITTI labels** (`data_odometry_labels.zip`, 179 MB)
3. **Voxel labels** (`SemanticKITTI_voxels.zip`, 1.6 GB) — **REQUIRED for SSC task**

Place under `data/SemanticKITTI/` so the layout becomes:

```
data/SemanticKITTI/
└── sequences/
    ├── 00/
    │   └── voxels/
    │       ├── 000000.bin       # sparse LiDAR voxel mask
    │       ├── 000000.label     # GT 256x256x32 voxel grid
    │       ├── 000000.invalid   # never-observed mask
    │       └── 000000.occluded
    ├── 01/ ...
    ├── 08/   <-- official validation split
    └── 11..21/  <-- hidden test split (no labels)
```

Verify:

    python scripts/prepare_data.py --root data/SemanticKITTI

## SCPNet predictions (required, ~178 GB real + synth)

These are precomputed SCPNet base predictions for val seq 08, test seqs
11..21, and the synthetic pools, so you can run the S²D² refiner without first
running SCPNet yourself:

    python scripts/download_assets.py --predictions

The hosted mirror is released upon paper publication. Until then
`download_assets.py --predictions` exits with the manual-download instructions.

### Per-frame triplet

Each real sequence directory holds three files per frame:

```
data/scpnet_predictions/
├── 00/ 01/ … 21/
│   ├── {id:06d}_pred.npy        # (256, 256, 32) uint8, class indices in [0, 19]
│   ├── {id:06d}_bev_top.npy     # (256, 256)     uint8, top-occupied-voxel BEV, [0, 19]
│   └── {id:06d}_bev_vote.npy    # (256, 256)     uint8, height-majority-vote BEV, [0, 19]
├── synthetic/                   # base preds over the 57,650-frame pool (58,021 files; superset)
├── synthetic_10000/             # 9,999-frame data-scaling subset
├── synthetic_30000/             # 29,999-frame data-scaling subset
└── synthetic_31k/               # 32,039-frame headline pool
```

`{id}_pred.npy` is the dense 3D completion. `{id}_bev_top.npy` and
`{id}_bev_vote.npy` are two 2D bird's-eye-view projections of that completion
(top-most occupied voxel vs. column majority vote); both are `(256, 256)`
uint8 in the 20-class learning-map space `[0, 19]`. The four `synthetic*`
directories are the SCPNet base predictions over the PS³ pool and its
data-scaling subsets; only `synthetic_31k/` matches the 32,039-frame headline
pool used in the paper.

## JS3C-Net predictions (required for cross-base reproduction, ~190 GB)

Precomputed for val seq 08 + train seqs 00-07, 09, 10 + test 11-21 + the 31K
and 57K synthetic pools (the `31K` pool is the 32,039-frame `synthetic_31k`
dir). Required to reproduce the v1.1.0 cross-base headline (paper
tab:portable_s2d2 cross-base rows; official `semantic-kitti-api` headline
22.7 → 26.1, +3.3 pp val mIoU):

    python scripts/download_assets.py --js3c-predictions

The hosted mirror is released upon paper publication; until then the command
above exits with the manual-download instructions (clone JS3C-Net and dump
locally, shown below). Layout mirrors `scpnet_predictions/` exactly:

```
data/js3cnet_predictions/
├── 00/ 01/ … 21/                       # real-frame predictions (4 541 + 1 101 + … frames per seq)
├── synthetic_31k/                      # 31 442 of 32 039 pool frames; 597 (1.9%) net missing per known gap
├── synthetic_filtered/                 # 38 322 frames (covers the 57K pool variant)
├── synthetic_31k_bad_frames.txt        # 600-entry blacklist (disjoint from the 597 net missing; see js3cnet_predictions/README.md)
└── README.md
```

The cross-base headline (26.1 % val mIoU under the official `semantic-kitti-api`;
26.7 % under the paper's internal training-time evaluator, a continuity row;
24.3 % at-deploy with derived BEV) is trained on real frames only; the synth
subdirs are shipped for the synth-augmentation analysis in the paper's
supplementary validation-protocol table and future use.

Alternatively, dump locally from your own JS3C-Net clone:

```bash
git clone --depth 1 https://github.com/yanx27/JS3C-Net external/JS3C-Net
bash external/JS3C-Net/download_pretrained.sh
python scripts/dump_js3c_predictions.py \
    --js3c-repo external/JS3C-Net \
    --semantickitti_root data/SemanticKITTI \
    --output_dir data/js3cnet_predictions \
    --sequences 00 01 02 03 04 05 06 07 08 09 10
```

See `docs/REPRODUCIBILITY.md` for the full cross-base protocol.

## LMSCNet predictions (required for the LMSCNet cross-base row, ~46 GB)

LMSCNet (Roldão et al., 3DV 2020) is the third structurally different frozen base
(a ~0.4M-param 2D-CNN SSC model) used for the cross-base demonstration in
paper tab:portable_s2d2. The LMSCNet+S²D² row lifts val mIoU **14.8 → 16.6 (+1.8 pp)**
(the LMSCNet base is re-scored from on-disk predictions through the official
`semantic-kitti-api`, superseding the earlier 12.10 → 16.59 / +4.49 summary)
and is trained on **real frames only** (sequences 00-07, 09, 10 + val 08).

Hosted predictions are released upon paper publication. Until then, dump them
locally from the publicly released multiscale LMSCNet checkpoint:

```bash
# 1. Clone LMSCNet and fetch its pretrained multiscale checkpoint
#    (Google Drive link in the official LMSCNet README).
git clone --depth 1 https://github.com/astra-vision/LMSCNet external/LMSCNet

# 2. Dump per-frame base predictions. Real-only reproduction needs the
#    train split + val 08; the hidden test (11-21) is optional.
python scripts/dump_lmscnet_predictions.py \
    --lmscnet-repo external/LMSCNet \
    --checkpoint external/LMSCNet/pretrained_models/LMSCNet.pth \
    --semantickitti_root data/SemanticKITTI \
    --output_dir data/lmscnet_predictions \
    --sequences 00 01 02 03 04 05 06 07 08 09 10
#   (--weights is accepted as an alias for --checkpoint.)
```

Once the mirror is live, the same predictions can be fetched with:

```bash
python scripts/download_assets.py --lmscnet-predictions
```

Layout mirrors `scpnet_predictions/` / `js3cnet_predictions/`:

```
data/lmscnet_predictions/
└── {00..10}/
    └── {frame_id}_pred.npy   # (256, 256, 32) uint8 class indices in [0, 19]
```

Each frame is ~2.1 MB (uint8, 256×256×32); the released {00..10} set
(23,201 frames = 19,130 train + 4,071 val) is ~46 GB.
`configs/{train,eval}/lmscnet_*.yaml` read this directory via
`base_pred_dir: data/lmscnet_predictions`. See `docs/REPRODUCIBILITY.md` for
the full cross-base protocol.

## Cross-dataset zero-shot data (KITTI-360, SemanticPOSS)

Two evaluation-only domains for the cross-dataset zero-shot rows. The frozen
SemanticKITTI checkpoint is applied as-is: no fine-tuning, no target labels at
train time. Provision these only to reproduce the zero-shot table.

**SSCBench-KITTI-360** (val seq 06; same-sensor near-domain). Download from
[github.com/ai4ce/SSCBench](https://github.com/ai4ce/SSCBench) (SSCBench-KITTI360
voxel labels and the matching point clouds). Place under
`data/SSCBench-KITTI360/` so the layout becomes:

```
data/SSCBench-KITTI360/
└── sequences/
    └── 06/                          <-- the only sequence the eval reads
        └── voxels/
            ├── {frame_id}.bin       # sparse LiDAR voxel mask
            └── {frame_id}.label     # GT 256x256x32 voxel grid (16 shared classes)
```

`scripts/eval_kitti360.py` reads `data/SSCBench-KITTI360/sequences/06/voxels/`.

**SemanticPOSS** (val seq 02; cross-sensor domain). Download from
[www.poss.pku.edu.cn/semanticposs](http://www.poss.pku.edu.cn/semanticposs.html).
Place under `data/SemanticPOSS/` so the layout becomes:

```
data/SemanticPOSS/
└── sequences/
    └── 02/                          <-- the only sequence the eval reads
        ├── velodyne/
        │   └── {frame_id}.bin       # raw point cloud (voxelized at eval time)
        └── labels/
            └── {frame_id}.label     # GT labels (11-class TALoS Tab. 4 map)
```

`scripts/eval_semanticposs.py` reads `data/SemanticPOSS/sequences/02/`. Both
runs are evaluation-only: the SemanticKITTI-trained weights are never adapted to
the target domain.

## Object bank (required for training, 448 MB)

57,789 rare-class instances across 8 classes, used by the PS³
object-bank-paste step and by training-time copy-paste augmentation:

    python scripts/download_assets.py --object-bank

The hosted mirror is released upon paper publication. Until then
`download_assets.py --object-bank` exits with the manual-download instructions.

### On-disk format

The bank is shipped in two redundant forms under `data/object_bank/`:

```
data/object_bank/
├── metadata.json            # total + per-class counts, is_3d=true
├── object_bank_3d.pkl       # monolithic Object3DBank (~229 MB), all instances
├── bicycle/   obj_{id:06d}.npz
├── motorcycle/ …
├── truck/ …
├── other-vehicle/ …
├── person/ …
├── bicyclist/ …
├── motorcyclist/ …
└── trunk/     obj_{id:06d}.npz
```

Each `obj_{id:06d}.npz` is one instance, stored as a **sparse 3D occupancy**:
`voxels` is an `(N, 3)` float32 array of occupied-voxel coordinates, `labels`
is the matching `(N,)` per-voxel class array, and `class_id` is the
learning-map class id (e.g. bicycle = 2, trunk = 16). Side fields `bbox_size`,
`centroid`, `height_range`, and `source_scene` record where the instance came
from and how to place it. The monolithic `object_bank_3d.pkl` holds the same
instances grouped by class (`is_3d = true`) for fast in-memory loading; the
per-class `.npz` directories are the unpacked equivalent.

### Composition

Counts come from `data/object_bank/metadata.json`:

| Class | Instances |
|---|---|
| bicycle | 6,269 |
| motorcycle | 3,392 |
| truck | 2,464 |
| other-vehicle | 6,777 |
| person | 6,402 |
| bicyclist | 1,355 |
| motorcyclist | 1,130 |
| trunk | 30,000 |
| **total** | **57,789** |

The bank is heavily skewed: `trunk` alone holds 30,000 instances and the other
seven classes total ~27,789. If you train on it, weight or subsample by class
so the trunk pool does not dominate the paste augmentation (see
limitations below).

## Synthetic pool (optional, ~128/230 GB uncompressed)

The headline synthetic pool holds **32,039** synthetic (sparse, complete) pairs.
Pooled with the **19,130** real SemanticKITTI training frames this gives a
**51,169-frame** total training set — a **2.67× expansion** over the real-only
split. (The `31K` shorthand in the variant/directory names below is the
historical config-dir label `synthetic_pool_31K` / `--synthetic-pool 31K`; the
actual frame count is 32,039.) Five sizes are released for the data-scaling
ablation. The full pool is mirrored to IEEE DataPort upon paper publication;
until then the download script exits with the manual-download instructions:

    # Via download script (prints the manual-download note until the mirror is live):
    python scripts/download_assets.py --synthetic-pool 31K   # ~128 GB uncompressed (approx.; .tar.gz mirror is smaller)
    python scripts/download_assets.py --synthetic-pool 57K   # ~230 GB uncompressed (approx.; .tar.gz mirror is smaller)

You only need the synthetic pool if you want to **retrain from scratch**. The
released checkpoint already contains the trained weights.

If instead you want to **regenerate the pool yourself** (the PS³ HDL-64E
ray-tracer in `src/gssc/data/lidar_resampler_v2.py`), install the `ps3` extra
(`uv pip install numba`, or `pip install -e ".[ps3]"`) so the resampler uses
the Numba-accelerated fast path. Without Numba it falls back to a slower
pure-Python loop.

### Synthetic-pool file format

Both released pools (`synthetic_pool_31K/`, `synthetic_pool_57K/`) store one
flat directory of frames. Each frame is a triplet of `.npy` files keyed by a
6-digit id:

```
data/synthetic_pool_31K/
├── {id:06d}_voxels.npy     # (256, 256, 32) uint8, sparse LiDAR input, values {0, 1}
├── {id:06d}_gt_scene.npy   # (256, 256, 32) uint8, complete semantic target, [0, 19]
└── {id:06d}_bev.npy        # (256, 256)     uint8, BEV semantic map, [0, 19]
```

- `_voxels.npy` is the binary occupancy of the resampled sparse scan (~0.4–0.6 %
  occupied, the same density regime as a real single-frame SemanticKITTI scan).
  This is the model input.
- `_gt_scene.npy` is the complete labeled scene the model is trained to recover
  (~5–6 % occupied; `0` = unlabeled/empty). This is the target.
- `_bev.npy` is the bird's-eye-view collapse of `_gt_scene.npy` used by the BEV
  perception head.

The **(sparse, complete) training pair is (`_voxels`, `_gt_scene`)**. The 31K
pool has 32,039 such triplets (id range up to `032041`, with a few generation
gaps); the 57K pool has 57,650 (id range up to `057664`). Each pool also ships a
`generation_metadata.json`: the 31K dir's file is an aggregate-run record (32,039
selected, 51,169 total-with-real, 2.67× expansion, plus the JS filter settings),
while the 57K dir's is a representative per-batch record of the same filter
settings (js_threshold 0.35, top_fraction 0.5, occupancy 2–12 %).

### Packed vs. unpacked

Raw SemanticKITTI ships its voxel mask **bit-packed**: each `.bin` voxel file is
exactly 262,144 bytes (256·256·32 bits / 8). The released `_voxels.npy`,
`_gt_scene.npy`, and every prediction `.npy` in this repository are **dense and
unpacked**: full `(256, 256, 32)` uint8 arrays, one byte per voxel, with no
unpacking step needed. Load them directly with `numpy.load`.

### Label space

Every released `.npy` (synthetic pool, object bank, and all base predictions)
uses the **20-class SemanticKITTI learning-map space `[0, 19]`**, with
`0 = unlabeled/empty` and `1..19` the 19 semantic classes (1=car, 2=bicycle,
…, 16=trunk, …, 19=traffic-sign). This is the standard `learning_map` from the
official [`semantic-kitti.yaml`](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml);
the exact mapping used here is mirrored in
`src/gssc/data/learning_map.py` (`SEMANTICKITTI_LEARNING_MAP`). The released
files are **already mapped**: you do not apply `learning_map` again. To recover
the original 0–255 SemanticKITTI ids (for the official API), invert with
`learning_map_inv`.

### How the synthetic pool was generated (PS³)

The pool is built by the Paired Sparse–Dense Scene Synthesis (PS³) pipeline:

1. **Pyramid multinomial diffusion** generates complete labeled scenes
   coarse-to-fine: S1 (32²×4) → S2 (64²×8) → S3 (256²×32). The S3 checkpoint
   used for the released pools is `s3_v2_lr004/best_miou.pt`.
2. **Quality filtering** keeps only plausible scenes. A Jensen–Shannon-divergence
   class-distribution filter (threshold τ = 0.35, keeping the top fraction 0.5
   of candidates) runs alongside an occupancy filter (2–12 % occupied) and a
   spatial-layout filter. The 57K pool's `generation_metadata.json` records
   these exact settings.
3. **Rare-class object-bank paste** optionally injects rare-class instances from
   the object bank to rebalance the long tail.
4. **HDL-64E ray tracing** turns each enriched complete scene into a realistic sparse
   scan via a Velodyne HDL-64E Bresenham3D / DDA ray-tracer
   (`src/gssc/data/lidar_resampler_v2.py`), producing the `_voxels` half of the
   pair.

## Checkpoints (~4 GB)

```bash
python scripts/download_assets.py --checkpoints
```

## Disk-space summary

| What | Where | Size |
|---|---|---|
| SemanticKITTI raw | `data/SemanticKITTI/` | 80 GB |
| Voxel labels | (inside `data/SemanticKITTI/`) | 1.6 GB |
| SCPNet predictions | `data/scpnet_predictions/` | 178 GB (real + synth; only the val+test subset, ~135 GB, is needed for the eval-only headline below) |
| JS3C-Net predictions (v1.1.0) | `data/js3cnet_predictions/` | 190 GB (real + synth) |
| LMSCNet predictions (v2.1.0) | `data/lmscnet_predictions/` | ~46 GB (train+val08) |
| Object bank | `data/object_bank/` | 448 MB |
| Synthetic pool 31K | `data/synthetic_pool_31K/` | ~128 GB uncompressed (approx.) |
| Synthetic pool 57K (`tab:data_scaling` 57K row) | `data/synthetic_pool_57K/` | ~230 GB uncompressed (approx.) |
| Pretrained checkpoints | `data/checkpoints/` | ~4 GB |
| **Total (eval-only, SCPNet headline)** | | **~135 GB** |
| **Total (eval-only, +cross-base JS3C)** | | **~325 GB** (135 GB SCPNet subset + 190 GB JS3C-Net real + synth) |
| **Total (eval-only, +cross-base LMSCNet)** | | **~181 GB** (135 GB SCPNet subset + 46 GB LMSCNet train+val08) |
| **Total (full retrain, 31K)** | | **~383 GB** (~170 GB SCPNet real-train + synthetic predictions + ~128 GB 31K pool + raw + voxels + object bank + checkpoints) |

## Licenses & terms

Each dataset is governed by its own license; consult the upstream terms before
use and before redistributing anything derived from it.

| Dataset | License / terms | Link |
|---|---|---|
| SemanticKITTI | CC-BY-NC-SA 4.0 (non-commercial, share-alike) | [semantic-kitti.org/dataset.html](http://www.semantic-kitti.org/dataset.html) |
| SSCBench-KITTI360 | Governed by its own terms (see the SSCBench repository) | [github.com/ai4ce/SSCBench](https://github.com/ai4ce/SSCBench) |
| SemanticPOSS | Governed by its own terms (see the SemanticPOSS dataset page) | [www.poss.pku.edu.cn/semanticposs.html](http://www.poss.pku.edu.cn/semanticposs.html) |

The predictions and weights we derive from these datasets inherit any
restriction of the source data. In particular, anything derived from
SemanticKITTI is non-commercial under CC-BY-NC-SA 4.0.

## Recommended uses & limitations

**What this data is for.** The synthetic pool, object bank, and base
predictions support **SSC training augmentation for voxel-grid models**. They
were built and validated with voxel-native bases (SCPNet, LMSCNet) and with the
voxel-grid S²D² refiner, where the synthetic pool contributes the within-SSC
lift folded into the SCPNet +2.37 pp headline.

**Limitations to know before you train.**

- *Out-of-distribution for point-cloud segmenters.* The `_voxels` half of each
  pair is a ray-traced voxel grid, not a real point cloud. Point-voxel hybrid
  heads can read it as OOD. JS3C-Net loses −3.8 pp when the synthetic pool is
  added, and its dump skips 597 of the 32,039 pool frames (1.9 %) that its
  encoder mis-classified; see `data/js3cnet_predictions/README.md` for the
  blacklist and the full account.
- *Object-bank class skew.* `trunk` is 30,000 of 57,789 instances. Subsample or
  reweight by class before pasting, or the trunk pool dominates.
- *Non-commercial.* Everything here derives from SemanticKITTI and is
  CC-BY-NC-SA 4.0 (non-commercial, share-alike).

### Released-asset naming

The download script and the disk-space table use the **released** names
(`synthetic_pool_31K`, `synthetic_pool_57K`, `object_bank`,
`js3cnet_predictions`, `scpnet_predictions`, `lmscnet_predictions`). In the
development tree these map to the original source directories:
`synthetic_pool_31K → synthetic_31k`, `synthetic_pool_57K → synthetic_filtered`,
`object_bank → object_bank_3d`, `js3cnet_predictions → js3c_predictions`. The
contents are identical; only the directory label differs.

## Maintenance

- **Maintainer / contact:** open a GitHub issue at
  [github.com/BillyChern/GSSC-S2D2](https://github.com/BillyChern/GSSC-S2D2).
  This is also the errata channel: file an issue if a frame, count, or shape
  here does not match what you downloaded.
- **Version:** v2.1.0 (`submission-ready-tpami-2026`).
- **Mirror:** the full synthetic pool is mirrored to IEEE DataPort on paper
  publication; until then the download scripts print manual-download
  instructions.
- **Errata posture:** the paper is the source of truth for every reported
  number. If a count in this datasheet disagrees with the paper, the paper
  wins and the discrepancy is an erratum to fix here.
