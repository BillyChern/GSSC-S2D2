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

## SCPNet predictions (required, ~50 GB)

Precomputed for val seq 08 + test seq 11..21. Lets users skip running SCPNet themselves::

    python scripts/download_assets.py --predictions

Hosted at `[SCPNET_PREDICTIONS_URL]`.

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
| SCPNet predictions | `data/scpnet_predictions/` | 50 GB |
| Object bank | `data/object_bank/` | 448 MB |
| Synthetic pool 31K | `data/synthetic_pool_31K/` | 120 GB |
| Pretrained checkpoints | `data/checkpoints/` | 3 GB |
| **Total (eval-only)** | | **~135 GB** |
| **Total (full retrain)** | | **~255 GB** |
