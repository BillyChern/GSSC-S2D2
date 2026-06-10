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

Precomputed for val seq 08 + test seq 11..21 + the 57K synthetic pool. Lets
users skip running SCPNet themselves:

    python scripts/download_assets.py --predictions

The hosted mirror is released upon paper publication; until then
`download_assets.py --predictions` exits with the manual-download
instructions. Layout matches the JS3C-Net dataset described below
(`{00..21}/{frame_id}_pred.npy` + `synthetic/{frame_id}_pred.npy`).

## JS3C-Net predictions (required for cross-base reproduction, ~54 GB)

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
├── synthetic_31k/                      # 31 442 of 32 039 pool frames; 597 (1.9%) missing per known gap
├── synthetic_filtered/                 # 38 322 frames (covers the 57K pool variant)
├── synthetic_31k_bad_frames.txt        # blacklist for the 597-frame gap
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

## LMSCNet predictions (required for the LMSCNet cross-base row, ~40 GB)

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
└── {00..21}/
    └── {frame_id}_pred.npy   # (256, 256, 32) uint8 class indices in [0, 19]
```

Each frame is ~2.1 MB (uint8, 256×256×32); the real-only set (~19K frames) is
~40 GB. `configs/{train,eval}/lmscnet_*.yaml` read this directory via
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

57,789 rare-class instances across 8 classes (bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, trunk):

    python scripts/download_assets.py --object-bank

The hosted mirror is released upon paper publication; until then
`download_assets.py --object-bank` exits with the manual-download
instructions.

## Synthetic pool (optional, 120-220 GB)

The headline synthetic pool holds **32,039** synthetic (sparse, complete) pairs.
Pooled with the **19,130** real SemanticKITTI training frames this gives a
**51,169-frame** total training set — a **2.67× expansion** over the real-only
split. (The `31K` shorthand in the variant/directory names below is the
historical config-dir label `synthetic_pool_31K` / `--synthetic-pool 31K`; the
actual frame count is 32,039.) Five sizes are released for the data-scaling
ablation. The full pool is mirrored to IEEE DataPort upon paper publication;
until then the download script exits with the manual-download instructions:

    # Via download script (prints the manual-download note until the mirror is live):
    python scripts/download_assets.py --synthetic-pool 31K   # ~120 GB (approx.)
    python scripts/download_assets.py --synthetic-pool 57K   # ~220 GB (approx.)

You only need the synthetic pool if you want to **retrain from scratch**. The
released checkpoint already contains the trained weights.

If instead you want to **regenerate the pool yourself** (the PS³ HDL-64E
ray-tracer in `src/gssc/data/lidar_resampler_v2.py`), install the `ps3` extra
(`uv pip install numba`, or `pip install -e ".[ps3]"`) so the resampler uses
the Numba-accelerated fast path; without Numba it silently falls back to a much
slower pure-Python loop.

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
| JS3C-Net predictions (v1.1.0) | `data/js3cnet_predictions/` | 54 GB |
| LMSCNet predictions (v2.1.0) | `data/lmscnet_predictions/` | ~40 GB (real-only) |
| Object bank | `data/object_bank/` | 448 MB |
| Synthetic pool 31K | `data/synthetic_pool_31K/` | ~120 GB (approx.) |
| Synthetic pool 57K (Tab. V) | `data/synthetic_pool_57K/` | ~220 GB (approx.) |
| Pretrained checkpoints | `data/checkpoints/` | ~4 GB |
| **Total (eval-only, SCPNet headline)** | | **~135 GB** |
| **Total (eval-only, +cross-base JS3C)** | | **~189 GB** (135 GB SCPNet subset + 54 GB JS3C-Net) |
| **Total (eval-only, +cross-base LMSCNet)** | | **~175 GB** (135 GB SCPNet subset + 40 GB LMSCNet real-only) |
| **Total (full retrain, 31K)** | | **~375 GB** |

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
