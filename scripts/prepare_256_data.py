"""Materialise the preprocessed 256x256x32 voxel cache used by every training recipe.

The trainers and the ``--bev_source gt`` evaluation path do **not** read the raw
SemanticKITTI ``.bin``/``.label`` files directly. They read a preprocessed cache::

    data/SemanticKITTI_3D/256/<seq>/<frame>_voxels.npy    # (256, 256, 32) uint8, 0/1
    data/SemanticKITTI_3D/256/<seq>/<frame>_bev.npy       # (256, 256)     uint8, [0, 19]
    data/SemanticKITTI_3D/256/<seq>/<frame>_gt_scene.npy  # (256, 256, 32) uint8, [0, 19]

That cache is **not** part of ``scripts/download_assets.py`` (the hosted assets are
checkpoints, base predictions and the synthetic pool). It is derived deterministically
from the raw SemanticKITTI voxel release, which requires manual registration, so this
script is the provisioning route:

    python scripts/prepare_data.py --root data/SemanticKITTI          # verify the raw layout
    python scripts/prepare_256_data.py --semantickitti_root data/SemanticKITTI
    python scripts/prepare_multi_frame_data.py --semantickitti_root data/SemanticKITTI
    python scripts/train.py train/31k_mf --gpu 0

Definitions
-----------
``_voxels.npy``   bit-unpacked ``<frame>.bin`` occupancy, reshaped to (256, 256, 32).
``_gt_scene.npy`` ``<frame>.label`` (uint16, original SemanticKITTI ids) mapped through
                  the official ``learning_map`` into the 20-class training space.
``_bev.npy``      per-column **majority vote over the non-empty classes** of
                  ``_gt_scene.npy`` (ties resolved to the lower class index); columns
                  with no occupied voxel stay 0. This is the aggregation the released
                  checkpoints were trained against -- it is *not* a max or a top-most
                  projection, both of which give visibly different maps.

The class mapping is imported from :mod:`gssc.data.learning_map` rather than re-typed
here: a hand-written map silently drops the ``252..259`` *moving* variants and deletes
most of the ground truth for the rare VRU classes.

Sequences 11..21 ship without ``.label``; for those only ``_voxels.npy`` is written.
That is the default; pass ``--require-labels`` to fail loudly on a label-less frame
instead.

This script builds the SINGLE-frame cache only. The ``*_mf`` recipes additionally
read ``data/SemanticKITTI_3D/256_multi_frame/`` -- see
``scripts/prepare_multi_frame_data.py``.

Output size: 2,097,280 B (voxels) + 2,097,280 B (gt_scene) + 65,664 B (bev) =
4,260,224 B per frame. Measured on the shipped cache: val seq 08 is 4,071 frames /
17,343,371,904 B = 17.3 GB (16.2 GiB), and the eleven annotated sequences are
23,201 frames / 98,841,457,024 B = 98.8 GB (92.1 GiB). Point ``--output_dir`` at a
volume with room.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Repo root (scripts/ -> repo). Anchors repo-relative defaults so the script is
# reproducible on any machine, and makes `gssc` importable from a plain checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gssc.data.learning_map import LEARNING_MAP_ARRAY

logger = logging.getLogger("prepare_256_data")

GRID_SHAPE: tuple[int, int, int] = (256, 256, 32)
NUM_CLASSES = 20
DEFAULT_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def unpack_occupancy(bin_path: Path) -> np.ndarray:
    """Bit-unpack a SemanticKITTI ``voxels/<frame>.bin`` into (256, 256, 32) uint8 0/1."""
    packed = np.fromfile(bin_path, dtype=np.uint8)
    return np.unpackbits(packed).reshape(GRID_SHAPE).astype(np.uint8)


def load_scene_labels(label_path: Path) -> np.ndarray:
    """Load ``voxels/<frame>.label`` and map it into the 20-class training space."""
    raw = np.fromfile(label_path, dtype=np.uint16).reshape(GRID_SHAPE)
    # Ids above the official table (never emitted by the release, but cheap to guard)
    # fold to 0 = unlabeled rather than indexing out of the LUT.
    return np.where(raw < LEARNING_MAP_ARRAY.shape[0],
                    LEARNING_MAP_ARRAY[np.minimum(raw, LEARNING_MAP_ARRAY.shape[0] - 1)],
                    0).astype(np.uint8)


def column_majority_bev(scene: np.ndarray) -> np.ndarray:
    """Per-column majority vote over the non-empty classes of a labelled scene.

    Args:
        scene: (X, Y, Z) uint8 labels in the training space, 0 = empty.

    Returns:
        (X, Y) uint8 BEV map; 0 where the whole column is empty. Ties between two
        classes with the same voxel count resolve to the lower class index.
    """
    counts = np.zeros(scene.shape[:2] + (NUM_CLASSES,), dtype=np.int32)
    for cls in range(1, NUM_CLASSES):
        counts[:, :, cls] = (scene == cls).sum(axis=2)
    winner = counts.argmax(axis=2).astype(np.uint8)
    return np.where(counts.max(axis=2) > 0, winner, 0).astype(np.uint8)


def process_frame(task: tuple[str, str, str, bool, bool]) -> tuple[str, str]:
    """Worker: write the three cache arrays for one frame.

    Returns a ``(status, frame_id)`` pair; ``status`` is one of ``ok``, ``skipped``,
    ``no-label`` or ``error: ...``.
    """
    bin_str, label_str, out_dir_str, skip_existing, require_labels = task
    bin_path = Path(bin_str)
    out_dir = Path(out_dir_str)
    frame_id = bin_path.stem

    voxels_out = out_dir / f"{frame_id}_voxels.npy"
    bev_out = out_dir / f"{frame_id}_bev.npy"
    gt_out = out_dir / f"{frame_id}_gt_scene.npy"
    label_path = Path(label_str) if label_str else None
    has_label = label_path is not None and label_path.is_file()

    wanted = [voxels_out] + ([bev_out, gt_out] if has_label else [])
    if skip_existing and all(p.is_file() for p in wanted):
        return ("skipped", frame_id)

    try:
        np.save(voxels_out, unpack_occupancy(bin_path))
        if has_label:
            scene = load_scene_labels(label_path)
            np.save(gt_out, scene)
            np.save(bev_out, column_majority_bev(scene))
        elif require_labels:
            return ("error: missing .label", frame_id)
        else:
            return ("no-label", frame_id)
    except (OSError, ValueError) as exc:
        return (f"error: {exc}", frame_id)
    return ("ok", frame_id)


def build_tasks(raw_root: Path, out_root: Path, sequences: list[str],
                skip_existing: bool, require_labels: bool) -> list[tuple[str, str, str, bool, bool]]:
    """Enumerate per-frame work items and create the output sequence directories."""
    tasks: list[tuple[str, str, str, bool, bool]] = []
    for seq in sequences:
        voxel_dir = raw_root / "sequences" / seq / "voxels"
        if not voxel_dir.is_dir():
            logger.warning("Sequence %s: no voxels dir at %s -- skipping", seq, voxel_dir)
            continue
        out_dir = out_root / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        bin_files = sorted(voxel_dir.glob("*.bin"))
        if not bin_files:
            logger.warning("Sequence %s: %s holds no .bin frames", seq, voxel_dir)
            continue
        logger.info("Sequence %s: %d frames -> %s", seq, len(bin_files), out_dir)
        for bin_path in bin_files:
            label_path = voxel_dir / f"{bin_path.stem}.label"
            tasks.append((str(bin_path), str(label_path) if label_path.is_file() else "",
                          str(out_dir), skip_existing, require_labels))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the data/SemanticKITTI_3D/256 voxel cache the trainers read.")
    parser.add_argument("--semantickitti_root", type=str, default=str(REPO_ROOT / "data" / "SemanticKITTI"),
                        help="Raw SemanticKITTI root holding sequences/<seq>/voxels/ "
                             "(default: <repo>/data/SemanticKITTI, the layout docs/DATASET.md sets up)")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "data" / "SemanticKITTI_3D" / "256"),
                        help="Cache root to write (default: <repo>/data/SemanticKITTI_3D/256, "
                             "which is <data_root>/SemanticKITTI_3D/256 for data_root=data)")
    parser.add_argument("--sequences", type=str, nargs="+", default=DEFAULT_SEQUENCES,
                        help="Sequences to process (default: the eleven annotated train+val sequences)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--force", action="store_true",
                        help="Recompute frames whose cache files already exist (default: skip them)")
    parser.add_argument("--require-labels", action="store_true",
                        help="Fail on frames without a .label instead of writing occupancy only "
                             "(sequences 11..21 legitimately have none)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_root = Path(args.semantickitti_root)
    out_root = Path(args.output_dir)
    if not (raw_root / "sequences").is_dir():
        sys.exit(f"No sequences/ under {raw_root}. Download the SemanticKITTI voxel release first "
                 f"(see docs/DATASET.md) and verify it with "
                 f"`python scripts/prepare_data.py --root {raw_root}`.")

    out_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(raw_root, out_root, args.sequences, not args.force, args.require_labels)
    if not tasks:
        sys.exit(f"No frames found under {raw_root}/sequences for sequences {args.sequences}.")

    logger.info("Processing %d frames with %d workers -> %s", len(tasks), args.workers, out_root)
    tallies: dict[str, int] = {}
    errors: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (status, frame_id) in enumerate(pool.map(process_frame, tasks, chunksize=8), 1):
            key = "error" if status.startswith("error") else status
            tallies[key] = tallies.get(key, 0) + 1
            if key == "error":
                errors.append((frame_id, status))
            if i % 1000 == 0:
                logger.info("  %d/%d frames", i, len(tasks))

    logger.info("Done: %s", ", ".join(f"{k}={v}" for k, v in sorted(tallies.items())))
    if errors:
        for frame_id, status in errors[:20]:
            logger.error("  %s: %s", frame_id, status)
        sys.exit(f"{len(errors)} frame(s) failed; the cache is incomplete.")
    logger.info("Cache ready at %s", out_root)


if __name__ == "__main__":
    main()
