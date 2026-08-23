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

### Preprocessed 256³ voxel cache (required for TRAINING; not for the eval rows)

`gssc.training.train_scene_completion` never reads the raw `.bin`/`.label` files
above. It reads a preprocessed cache that **no download provisions** — build it
once from the raw tree:

    python scripts/prepare_256_data.py --semantickitti_root data/SemanticKITTI

```
data/SemanticKITTI_3D/256/
├── {00..10}/
│   ├── {id:06d}_voxels.npy     # (256, 256, 32) uint8, sparse LiDAR input
│   ├── {id:06d}_bev.npy        # (256, 256)     uint8, BEV semantic map
│   └── {id:06d}_gt_scene.npy   # (256, 256, 32) uint8, complete semantic target
└── synthetic/                  # the 31K synthetic pool, same triplet layout
```

Measured on the staged cache: the eleven annotated sequences `00`-`10` are
69,603 files / 92.1 GiB / 98.8 GB, of which val seq `08` alone is 12,213 files /
16.2 GiB / 17.3 GB. `synthetic/` is the **same bytes** as the released
`synthetic_pool_31K` (127.1 GiB / 136.5 GB), staged under the cache root rather
than a second copy — see *Synthetic pool* below. Skipping this step aborts
training with `MissingVoxelCacheError`, which names the command; see
`docs/TRAIN.md`.

None of the tabulated **eval** rows needs this cache: `eval/val_1step`,
`eval/js3c_val_realistic` and `eval/lmscnet_val_1step` all derive their seed BEV
from the base's own 3D prediction. The two exceptions are the GT-BEV diagnostics
`eval/js3c_val_paper` and its alias `eval/js3c_val_1step`, which set
`bev_root: SemanticKITTI_3D/256` and read `<seq>/<frame>_bev.npy` out of this
cache — so running those needs seq 08's slice of it (16.2 GiB / 17.3 GB) on top
of the eval-only totals in the disk-space table below.

> **A multi-frame retrain needs a second cache that no script here builds.** The
> `_mf` recipes look for `data/SemanticKITTI_3D/256_multi_frame/<seq>/<frame>.npz`,
> which `prepare_256_data.py` does not produce and no released asset supplies.
> Its absence is **silent** — the loader falls back to single-frame LiDAR instead
> of failing. `docs/TRAIN.md` (box under "Headline") has the code path and the
> pre-launch check. Evaluation of the released checkpoints is unaffected.

## SCPNet predictions (required, ~177 GiB / ~190 GB real + synth)

These are precomputed SCPNet base predictions for val seq 08, test seqs
11..21, and the synthetic pools, so you can run the S²D² refiner without first
running SCPNet yourself:

    python scripts/download_assets.py --predictions

`download_assets.py --predictions` provisions these from the hosted mirror.
Narrow it to what you actually need with `--include`, whose patterns go straight
to `huggingface_hub.snapshot_download(allow_patterns=...)`:

```bash
python scripts/download_assets.py --predictions                                  # whole tree, 177 GiB / 190 GB
python scripts/download_assets.py --predictions --include 'scpnet_predictions/08/*'   # val seq 08 only, 9.07 GB
```

To regenerate them locally instead, run the frozen base yourself against an
upstream SCPNet checkout:

```bash
git clone --depth 1 https://github.com/SCPNet/Codes-for-SCPNet external/Codes-for-SCPNet
python -m gssc.inference.run_scpnet \
    --scpnet-repo external/Codes-for-SCPNet \
    --checkpoint data/checkpoints/scpnet_v2_port.pth \
    --config external/Codes-for-SCPNet/config/semantickitti-multiscan.yaml \
    --data_root data/SemanticKITTI --sequences 08 \
    --output_dir data/scpnet_predictions
```

`--checkpoint` and `--config` default to
`<scpnet-repo>/model_load_dir/pretrained.pth` and
`<scpnet-repo>/config/semantickitti-multiscan.yaml` if omitted;
`scpnet_v2_port.pth` is the ported weight `docs/MODEL_ZOO.md` documents under
"SCPNet base". `--sequences` takes a **comma-separated** list here (or `all`),
unlike the space-separated `--sequences` of the JS3C-Net and LMSCNet dumpers
below. Name `--data_root` and `--output_dir` explicitly: both default to a
`datasets/` tree this release does not create.

### Per-frame triplet

Each real sequence directory holds three files per frame:

```
data/scpnet_predictions/
├── 00/ 01/ … 21/
│   ├── {id:06d}_pred.npy        # (256, 256, 32) uint8, class indices in [0, 19]
│   ├── {id:06d}_bev_top.npy     # (256, 256)     uint8, top-occupied-voxel BEV, [0, 19]
│   └── {id:06d}_bev_vote.npy    # (256, 256)     uint8, height-majority-vote BEV, [0, 19]
├── synthetic/                   # superset over the 57,650-frame pool:
│                                #   58,021 frames / 174,063 files (all three per frame)
├── synthetic_10000/             #  9,999-frame data-scaling subset: 19,998 files (pred + bev_top only)
├── synthetic_30000/             # 29,999-frame data-scaling subset: 59,998 files (pred + bev_top only)
└── synthetic_31k/               # 32,039-frame headline pool:       96,117 files (all three per frame)
```

`{id}_pred.npy` is the dense 3D completion. `{id}_bev_top.npy` and
`{id}_bev_vote.npy` are two 2D bird's-eye-view projections of that completion
(top-most occupied voxel vs. column majority vote); both are `(256, 256)`
uint8 in the 20-class learning-map space `[0, 19]`. The four `synthetic*`
directories are the SCPNet base predictions over the PS³ pool and its
data-scaling subsets; only `synthetic_31k/` matches the 32,039-frame headline
pool used in the paper. **The triplet is not uniform across them**: `synthetic/`
and `synthetic_31k/` ship all three files per frame, while `synthetic_10000/`
and `synthetic_30000/` ship `_pred` and `_bev_top` only — no `_bev_vote`. Every
count here is measured (`ls -U <dir> | wc -l` for entries,
`ls -U <dir> | grep -c '_bev_vote\.npy$'` for the per-kind split), and
`synthetic/` holds one directory entry beyond its 174,063 files: a stray
`synthetic_31k` symlink back to its sibling farm.

> **`synthetic_10000/`, `synthetic_30000/` and `synthetic_31k/` hold no bytes of
> their own in the staging tree** — they are directories of symlinks into
> `synthetic/` (19,998 / 59,998 / 96,117 links respectively), so `synthetic/` is
> counted once and the tree's **unique content is 177 GiB / 190 GB**. A transfer
> that materialises symlinks as real files (most upload tools do) expands it back
> out to **324 GiB / 348 GB**, so size that kind of transfer against the
> materialised figure rather than against the unique content. The materialised
> figure is unique content plus the three farms measured separately
> (176.68 + 20.14 + 60.43 + 66.50 = 323.75 GiB); do **not** try to get it from a
> plain `os.walk(followlinks=True)`, which does not terminate on a sensible
> number here — `synthetic/synthetic_31k` is a symlink back to the sibling farm
> `synthetic_31k/`, which links back into `synthetic/`, and walking that cycle
> reports thousands of GiB.

## JS3C-Net predictions (required for cross-base reproduction, ~189 GiB / ~203 GB)

Precomputed for val seq 08 + train seqs 00-07, 09, 10 + test 11-21 + the 31K
and 57K synthetic pools (the `31K` pool is the 32,039-frame `synthetic_31k`
dir). Required to reproduce the JS3C-Net cross-base rows of paper
tab:portable_s2d2, whose headline is **22.7 → 24.3, +1.6 pp** val mIoU under the
official `semantic-kitti-api` with derived BEV (26.05 is a separate GT-BEV
diagnostic of ours that the paper does not print, and is not the protocol behind
its 26.7 continuity row):

    python scripts/download_assets.py --js3c-predictions

The command above provisions these from the hosted mirror; the manual route
(clone JS3C-Net and dump locally) is shown below as an alternative. Layout mirrors `scpnet_predictions/` exactly:

```
data/js3cnet_predictions/
├── 00/ 01/ … 21/                       # real-frame predictions (4 541 + 1 101 + … frames per seq)
├── synthetic_31k/                      # 31 442 of 32 039 pool frames; 597 (1.9%) net missing per known gap
├── synthetic_filtered/                 # 38 322 frames (covers the 57K pool variant)
├── synthetic_31k_bad_frames.txt        # 600-entry blacklist (disjoint from the 597 net missing; see js3cnet_predictions/README.md)
└── README.md
```

The cross-base headline (**24.3 % val mIoU**, derived BEV under the official
`semantic-kitti-api`; the *same* derived-BEV setting reads **26.7 %** under the
paper's internal training-time evaluator, a continuity row — the evaluator is
what differs, not the BEV source) is trained on real frames only; the synth
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

## LMSCNet predictions (required for the LMSCNet cross-base row, ~45 GiB / ~49 GB)

LMSCNet (Roldão et al., 3DV 2020) is the third structurally different frozen base
(a ~0.4M-param 2D-CNN SSC model) used for the cross-base demonstration in
paper tab:portable_s2d2. The LMSCNet+S²D² row lifts val mIoU **14.8 → 16.6 (+1.8 pp)**
(the LMSCNet base is re-scored from on-disk predictions through the official
`semantic-kitti-api`, superseding the earlier 12.10 → 16.59 / +4.49 summary)
and is trained on **real frames only** (sequences 00-07, 09, 10 + val 08).

Hosted predictions are available. To regenerate them instead, dump them
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

The same predictions are mirrored on the Hugging Face dataset repo
[`Stone-Chern/GSSC-S2D2-datasets`](https://huggingface.co/datasets/Stone-Chern/GSSC-S2D2-datasets)
and can be fetched with:

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
(23,201 frames = 19,130 train + 4,071 val, plus a `LICENSE` and a `README.md`)
measures 45.3 GiB / 48.7 GB, the same pair as the disk-space table below.
`configs/{train,eval}/lmscnet_*.yaml` read this directory via
`base_pred_dir: data/lmscnet_predictions`. See `docs/REPRODUCIBILITY.md` for
the full cross-base protocol.

## Cross-dataset zero-shot data (KITTI-360, SemanticPOSS)

Two evaluation-only domains for the cross-dataset zero-shot rows. The frozen
SemanticKITTI checkpoint is applied as-is: no fine-tuning, no target labels at
train time. Provision these only to reproduce the zero-shot table.

**SSCBench-KITTI-360** (val scene 06; same-sensor near-domain). Download from
[github.com/ai4ce/SSCBench](https://github.com/ai4ce/SSCBench) (SSCBench-KITTI360
voxel labels and the matching raw point clouds). The loader
(`src/gssc/data/kitti360.py`) reads the **SSCBench release layout keyed by
KITTI-360 drive name**, not a `sequences/06/` tree:

```
data/SSCBench-KITTI360/
├── kitti_360_match.txt                       # frame-id <-> raw-velodyne-id table (required, see below)
└── sscbench-kitti/                           # = `kitti360_root`; must DIRECTLY contain the two dirs below
    ├── data_2d_raw/
    │   └── 2013_05_28_drive_0006_sync/       <-- val scene 06, the only sequence the eval reads
    │       └── voxels/
    │           ├── {frame_id}.bin            # occupancy input (bit-packed, as in SemanticKITTI)
    │           ├── {frame_id}.label          # GT 256x256x32 voxel grid, flat uint16, NOT bit-packed
    │           └── {frame_id}.invalid        # raw uint8 per voxel, NOT bit-packed
    └── data_3d_raw/
        └── 2013_05_28_drive_0006_sync/
            └── velodyne_points/data/
                └── {frame_id}.bin            # raw scan, float32 (x, y, z, intensity)
```

Three things about this layout are easy to get wrong:

* **Drive names, not two-digit sequence ids.** `Kitti360SSCDataset` enumerates
  `data_2d_raw/<drive>/voxels/*.label`, with `<drive>` taken from
  `KITTI360_SPLITS` (`val` = `2013_05_28_drive_0006_sync`). The two-digit `06`
  appears only later, when `scripts/eval_kitti360.py` stages GT and predictions
  for the scorer.
* **Both paths come from the config, and repo-relative paths resolve against the
  repo root** (`configs/eval/kitti360_zeroshot_1step.yaml`:
  `kitti360_root: data/SSCBench-KITTI360/sscbench-kitti`,
  `match_table: data/SSCBench-KITTI360/kitti_360_match.txt`). The SSCBench
  archive unpacks with an extra nesting level on some mirrors, so point
  `kitti360_root` at whichever directory *directly* contains `data_2d_raw/` and
  `data_3d_raw/`.
* **`.invalid` is raw here, bit-packed in SemanticKITTI.** The driver repacks it
  before scoring (`repack_invalid_for_semantic_kitti_api`); you do not do this
  by hand.

Sanity check after unpacking: the val drive's `voxels/` directory holds 5,436
files — the `.bin` / `.label` / `.invalid` triplet for each of the 1,812 val
frames — with ids stepping by 5 (`000000`, `000005`, …), while
`velodyne_points/data/` holds consecutively numbered 6-digit scans.

**`kitti_360_match.txt` is required, and here is where it comes from.**
`Kitti360SSCDataset.__init__` opens `cfg.match_table` unconditionally, so the
eval raises `FileNotFoundError` without it — even though this release does not
need its ids: the SSCBench velodyne scans are named with the same 6-digit frame
ids as the voxel files, so the loader indexes velodyne directly by `frame_id`
and keeps the table's 10-digit raw id for provenance (and as a fallback for an
upstream layout that keeps the long names). We could not find the file inside
the SSCBench-KITTI360 archive itself; the copy we ran with is byte-identical
(md5 `a21dc8f1d976fd5b821a3c681efaa425`, 64,316 lines) to the one the PaSCo
repository ships at `pasco/data/kitti360/kitti_360_match.txt`, which is where
ours came from. Take it from there, or from any SSCBench-derived codebase that
ships the same table; each line is
`<drive> <raw_id>.png <voxel_frame_id>.png`.

The zero-shot driver writes the frozen base's KITTI-360 predictions to
`data/scpnet_kitti360_predictions/` (`base_pred_dir` in the same config) and
stages GT plus refined predictions under `data/kitti360_gt_staged/` and
`data/predictions/kitti360_zeroshot/`.

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

`scripts/eval_semanticposs.py` reads `data/SemanticPOSS/sequences/02/`.
SemanticPOSS ships no voxelized SSC ground truth, so the driver materialises it
(single-scan 256×256×32, majority vote + ray-cast `.invalid`) into
`data/SemanticPOSS_voxel_gt/sequences/02/voxels/` before scoring — that path is
`voxel_gt_root` in `configs/eval/semanticposs_seq02.yaml` and needs its own disk
space. Both runs are evaluation-only: the SemanticKITTI-trained weights are
never adapted to the target domain.

## Object bank (required for training, 313 MiB / 328 MB of data; 448 MB of disk blocks)

57,789 rare-class instances across 8 classes, used by the PS³
object-bank-paste step and by training-time copy-paste augmentation:

    python scripts/download_assets.py --object-bank

`download_assets.py --object-bank` provisions this from the hosted mirror;
the manual route below rebuilds it locally instead.

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

## Synthetic pool (optional, ~127 GiB (31K) / ~229 GiB (57K) uncompressed)

The headline synthetic pool holds **32,039** synthetic (sparse, complete) pairs.
Pooled with the **19,130** real SemanticKITTI training frames this gives a
**51,169-frame** total training set — a **2.67× expansion** over the real-only
split. (The `31K` shorthand in the variant/directory names below is the
historical config-dir label `synthetic_pool_31K` / `--synthetic-pool 31K`; the
actual frame count is 32,039.) **Two pools are released**, `31K` and `57K`;
the paper's data-scaling table also reports 0K / 10K / 20K rows, but those are
*subsets drawn at train time* and no 0K / 10K / 20K pool is staged for
download. (`0K` means real-only, so there is nothing to fetch for it at all.) The full pool is archived on IEEE DataPort rather than on the two
Hugging Face mirrors; its DOI has not been minted yet, so `--synthetic-pool`
exits with the manual-build instructions instead of downloading (the same
statement as under *Maintenance* below) and the manual route is the one that
works today. What is outstanding is that one DOI, not a release embargo: the
paper's availability footnote states that the code, the checkpoints and
the PS³ dataset are available at
[github.com/BillyChern/GSSC-S2D2](https://github.com/BillyChern/GSSC-S2D2), and
nothing in this datasheet is withheld pending acceptance.

    # Via the download script (resolves once the DataPort DOI is configured
    # in scripts/download_assets.py):
    python scripts/download_assets.py --synthetic-pool 31K   # ~127 GiB / ~136 GB uncompressed (.tar.gz mirror is smaller)
    python scripts/download_assets.py --synthetic-pool 57K   # ~229 GiB / ~246 GB uncompressed (.tar.gz mirror is smaller)

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

## Checkpoints (~4.58 GiB / ~4.9 GB)

```bash
python scripts/download_assets.py --checkpoints
```

## Disk-space summary

Every figure below with a GiB value was measured on the staged release payload
by summing `stat` apparent sizes with symlinks dereferenced (equivalent to
`du -Lsb`, but per-directory so the symlink farms are not double-counted) and is
quoted as `GiB / GB` so the two conventions cannot be confused. The two
SemanticKITTI raw rows are the upstream download-page figures, not our
measurements, and the eval-only totals therefore inherit whatever accuracy those
carry.

| What | Where | Size |
|---|---|---|
| SemanticKITTI raw | `data/SemanticKITTI/` | 80 GB (upstream figure) |
| Voxel labels | (inside `data/SemanticKITTI/`) | 1.6 GB (upstream figure) |
| SCPNet predictions, whole tree | `data/scpnet_predictions/` | 177 GiB / 190 GB (real seqs 00-21 = 56 GiB, `synthetic/` = 120 GiB) |
| — val seq 08 only (the headline eval) | `data/scpnet_predictions/08/` | **8.5 GiB / 9.1 GB** |
| — val 08 + hidden test 11-21 (for a submission) | | **16.6 GiB / 17.8 GB** |
| JS3C-Net predictions (v1.1.0) | `data/js3cnet_predictions/` | 189 GiB / 203 GB (real + synth) |
| LMSCNet predictions (v2.1.0) | `data/lmscnet_predictions/` | 45 GiB / 49 GB (train+val08) |
| Object bank | `data/object_bank/` | 313 MiB / 328 MB of data across 57,789 instance files (`du -Lsh` reports 448 MB — block usage, inflated by many tiny files) |
| Synthetic pool 31K | `data/synthetic_pool_31K/` | 127 GiB / 136 GB uncompressed |
| Synthetic pool 57K (`tab:data_scaling` 57K row) | `data/synthetic_pool_57K/` | 229 GiB / 246 GB uncompressed |
| Pretrained checkpoints | `data/checkpoints/` | 4.58 GiB / 4.9 GB (the 51-file payload `checksums.txt` covers) |
| Preprocessed 256³ voxel cache (**training only**, built locally) | `data/SemanticKITTI_3D/256/{00..10}/` | 92.1 GiB / 98.8 GB (val seq 08 alone: 16.2 GiB / 17.3 GB) |
| — its `synthetic/` branch | `data/SemanticKITTI_3D/256/synthetic/` | the *same bytes* as the 31K pool row above, not a second copy |
| **Total (eval-only, SCPNet headline on val seq 08)** | | **~96 GB** (80 raw + 1.6 voxels + 4.9 checkpoints + 9.1 SCPNet seq 08) |
| **Total (eval-only + hidden-test submission)** | | **~104 GB** (the row above + 8.7 SCPNet seqs 11-21) |
| **Total (eval-only, +cross-base JS3C)** | | **~299 GB** (~96 GB eval-only + 203 GB JS3C-Net real + synth) |
| **Total (eval-only, +cross-base LMSCNet)** | | **~145 GB** (~96 GB eval-only + 49 GB LMSCNet train+val08) |
| **Total (full retrain, 31K)** | | **~512 GB** (190 SCPNet real + synthetic predictions + 136 31K pool + 99 voxel cache seqs 00-10 + 80 raw + 1.6 voxels + 0.33 object bank + 4.9 checkpoints) |

> **Earlier revisions of this table claimed the SCPNet "val+test subset" was
> ~135 GB and reused that same 135 as the whole eval-only stack total.** Neither
> holds: the val+test subset measures 16.6 GiB / 17.8 GB, and no arrangement of
> the rows above sums to 135. The figures in this table are measured.

> **Fetching only the sequences you need.** Bare
> `scripts/download_assets.py --predictions` pulls the whole
> `scpnet_predictions/` tree (177 GiB / 190 GB). `--include` narrows it —
> the patterns are handed to
> `huggingface_hub.snapshot_download(allow_patterns=...)`, and `*` crosses `/`
> there, so a single pattern selects a sequence:
>
> ```bash
> python scripts/download_assets.py --predictions --include 'scpnet_predictions/08/*'
> ```
>
> That is the 9.07 GB val row above. `--include` applies to the prediction and
> object-bank groups only; `--checkpoints` is always a whole-repo snapshot.
> One caveat: the pattern is anchored against the **local** layout this file
> documents (`scpnet_predictions/<seq>/<frame>_pred.npy`), which is the same
> assumption the pre-existing whole-prefix fetch already made, but it has not
> been checked against the live Hugging Face tree. Spot-check the first use.

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
lift folded into the SCPNet +2.36 pp headline.

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
- **Version:** v2.3.8 (the submission snapshot; bump this WITH supplementary.tex's tag).
- **Mirror:** the full synthetic pool is archived on IEEE DataPort. Its DOI has
  not been minted yet — that, and not paper acceptance, is what is outstanding —
  so `scripts/download_assets.py --synthetic-pool` prints the manual-download
  instructions instead of fetching. The other assets are provisioned by the same
  script, and the pool itself can be rebuilt locally with the PS³ ray-tracer
  (see *Synthetic pool* above).
- **Errata posture:** the paper is the source of truth for every reported
  number. If a count in this datasheet disagrees with the paper, the paper
  wins and the discrepancy is an erratum to fix here.
