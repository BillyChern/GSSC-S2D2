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

Verify::

    python scripts/prepare_data.py --root data/SemanticKITTI

## SCPNet predictions (required, ~178 GB real + synth)

Precomputed for val seq 08 + test seq 11..21 + the 57K synthetic pool. Lets
users skip running SCPNet themselves::

    python scripts/download_assets.py --predictions

Hosted at `[SCPNET_PREDICTIONS_URL]`. Layout matches the JS3C-Net dataset
described below (`{00..21}/{frame_id}_pred.npy` + `synthetic/{frame_id}_pred.npy`).

## JS3C-Net predictions (required for cross-base reproduction, ~54 GB)

Precomputed for val seq 08 + train seqs 00-07, 09, 10 + test 11-21 + the 31K
and 57K synthetic pools. Required to reproduce the v1.1.0 cross-base headline
(paper Tab. III rows 90-91, +3.99 pp val mIoU)::

    python scripts/download_assets.py --js3c-predictions

Hosted at `[JS3C_PREDICTIONS_URL]`. Layout mirrors `scpnet_predictions/`
exactly:

```
data/js3cnet_predictions/
├── 00/ 01/ … 21/                       # real-frame predictions (4 541 + 1 101 + … frames per seq)
├── synthetic_31k/                      # 31 442 frames (597 missing per known gap)
├── synthetic_filtered/                 # 38 322 frames (covers the 57K pool variant)
├── synthetic_31k_bad_frames.txt        # blacklist for the 597-frame gap
└── README.md
```

The cross-base headline (26.72 % val mIoU) is trained on real frames only;
the synth subdirs are shipped for the supp tab:supp_b6_val analysis and
future use.

Alternatively, dump locally from your own JS3C-Net clone:

```bash
git clone --depth 1 https://github.com/yangyangyang127/JS3C-Net external/JS3C-Net
bash external/JS3C-Net/download_pretrained.sh
python scripts/dump_js3c_predictions.py \
    --js3c-repo external/JS3C-Net \
    --semantickitti_root data/SemanticKITTI/dataset \
    --output_dir data/js3cnet_predictions \
    --sequences 00 01 02 03 04 05 06 07 08 09 10
```

See `docs/REPRODUCIBILITY.md` for the full cross-base protocol.

## Object bank (required for training, 448 MB)

57,789 rare-class instances across 8 classes (bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, trunk)::

    python scripts/download_assets.py --object-bank

Hosted at `[OBJECT_BANK_URL]`.

## Synthetic pool (optional, 120-220 GB)

The 31K synthetic (sparse, complete) pairs used by the headline run. Five sizes
released for the data-scaling ablation::

    # IEEE DataPort (full):
    [SYNTHETIC_POOL_URL]

    # Or via download script:
    python scripts/download_assets.py --synthetic-pool 31K   # ~120 GB
    python scripts/download_assets.py --synthetic-pool 57K   # ~220 GB

You only need the synthetic pool if you want to **retrain from scratch**. The
released checkpoint already contains the trained weights.

## Checkpoints (~3 GB)

```bash
python scripts/download_assets.py --checkpoints
```

## Disk-space summary

| What | Where | Size |
|---|---|---|
| SemanticKITTI raw | `data/SemanticKITTI/` | 80 GB |
| Voxel labels | (inside `data/SemanticKITTI/`) | 1.6 GB |
| SCPNet predictions | `data/scpnet_predictions/` | 178 GB (real + synth) |
| JS3C-Net predictions (v1.1.0) | `data/js3cnet_predictions/` | 54 GB |
| Object bank | `data/object_bank/` | 448 MB |
| Synthetic pool 31K | `data/synthetic_pool_31K/` | 62 GB |
| Synthetic pool 57K (Tab. V) | `data/synthetic_pool_57K/` | 305 GB |
| Pretrained checkpoints | `data/checkpoints/` | ~4.5 GB |
| **Total (eval-only, SCPNet headline)** | | **~135 GB** |
| **Total (eval-only, +cross-base JS3C)** | | **~189 GB** |
| **Total (full retrain, 31K)** | | **~317 GB** |
