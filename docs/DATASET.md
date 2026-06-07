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
and 57K synthetic pools. Required to reproduce the v1.1.0 cross-base headline
(paper tab:portable_s2d2 cross-base rows, +3.99 pp val mIoU):

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

The cross-base headline (26.72 % val mIoU) is trained on real frames only;
the synth subdirs are shipped for the synth-augmentation analysis in the
paper's supplementary validation-protocol table and future use.

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
paper tab:portable_s2d2. The LMSCNet+S²D² row lifts val mIoU **12.10 → 16.59 (+4.49 pp)**
and is trained on **real frames only** (sequences 00-07, 09, 10 + val 08).

Hosted predictions are released upon paper publication. Until then, dump them
locally from the publicly released multiscale LMSCNet checkpoint:

```bash
# 1. Clone LMSCNet and fetch its pretrained multiscale checkpoint
#    (Google Drive link in the official LMSCNet README).
git clone --depth 1 https://github.com/cv-rits/LMSCNet external/LMSCNet

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

## Object bank (required for training, 448 MB)

57,789 rare-class instances across 8 classes (bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, trunk):

    python scripts/download_assets.py --object-bank

The hosted mirror is released upon paper publication; until then
`download_assets.py --object-bank` exits with the manual-download
instructions.

## Synthetic pool (optional, 120-220 GB)

The 31K synthetic (sparse, complete) pairs used by the headline run. Five sizes
released for the data-scaling ablation. The full pool is mirrored to IEEE
DataPort upon paper publication; until then the download script exits with the
manual-download instructions:

    # Via download script (prints the manual-download note until the mirror is live):
    python scripts/download_assets.py --synthetic-pool 31K   # ~120 GB (approx.)
    python scripts/download_assets.py --synthetic-pool 57K   # ~220 GB (approx.)

You only need the synthetic pool if you want to **retrain from scratch**. The
released checkpoint already contains the trained weights.

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
