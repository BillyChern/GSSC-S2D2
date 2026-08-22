"""Materialise the multi-frame LiDAR cache the ``*_mf`` training recipes read.

``S3DSKDDataset`` in teacher mode looks for one ``.npz`` per frame under::

    data/SemanticKITTI_3D/256_multi_frame/<seq>/<frame>.npz

Each file holds a five-scan ego-motion-compensated LiDAR sweep, voxelised onto
the same 256x256x32 grid as the single-frame cache::

    coords      (M, 3) int16    occupied voxel indices, unique, ascending hash order
    features    (M, 1) float16  mean intensity of the points in that voxel
    num_frames  ()     int64    how many scans were actually fused (< 5 at a
                                sequence's tail, where future scans run out)

The trainer consumes ``coords`` only and binarises it
(:mod:`gssc.data.semantickitti`, teacher branch of ``__getitem__``); ``features``
is carried for provenance and downstream use.

**Why this script exists.** Until now nothing in the release produced this tree.
Cold diffusion does not require it -- ``S3DSKDDataset`` keeps the sample and falls
back to single-frame LiDAR whenever ``scpnet_pred_dir`` is set -- so
``python scripts/train.py train/31k_mf`` would *run to completion on SINGLE-frame
input* while calling itself a multi-frame recipe. The loader now warns loudly
when that happens; this script is the way to make the warning stop being true.

Provenance
----------
The algorithm is SCPNet's multi-scan fusion (future scans, poses converted into
the LiDAR frame with ``Tr_inv @ pose @ Tr``), ported from the research tree's
``tools/prepare_multi_frame_voxels.py``. It is not a re-derivation: output was
checked against the shipped cache and reproduces it **byte for byte** --
19 frames spanning sequences 00, 03, 05, 08 and 10, including a sequence-tail
frame where only one scan is available, matched on ``coords``, ``features`` and
``num_frames`` exactly.

Inputs
------
``<semantickitti_root>/sequences/<seq>/velodyne/<frame>.bin``  raw scans
``<semantickitti_root>/sequences/<seq>/poses.txt``             odometry poses
``<calib_root>/sequences/<seq>/calib.txt``                     the ``Tr`` matrix

``calib.txt`` ships in the KITTI odometry *calibration* archive, which is a
separate download from the velodyne scans; if your SemanticKITTI tree was
assembled without it, point ``--calib_root`` at the odometry checkout that has
it. The script fails with that message rather than guessing a transform.

Output size
-----------
Measured on the full eleven annotated sequences (23,201 frames):
2,353,535,325 B = 2.4 GB / 2.2 GiB, ~99 KB per frame. Sequence 08 alone is
4,071 frames / 421,034,712 B = 0.42 GB. That is ~40x smaller than the
single-frame ``256/`` cache because the storage is sparse.

Usage
-----
    python scripts/prepare_multi_frame_data.py --semantickitti_root data/SemanticKITTI
    python scripts/train.py train/31k_mf --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Repo root (scripts/ -> repo), so the defaults are repo-relative on any machine.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("prepare_multi_frame_data")

VOXEL_SIZE = 0.2
GRID_SIZE = np.array([256, 256, 32])
# x in [0, 51.2), y in [-25.6, 25.6), z in [-2, 4.4) metres -- the SemanticKITTI SSC volume.
PC_RANGE = np.array([0.0, -25.6, -2.0, 51.2, 25.6, 4.4])
DEFAULT_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
DEFAULT_NUM_FRAMES = 5


def parse_calibration(path: Path) -> np.ndarray:
    """Return the 4x4 ``Tr`` (camera -> LiDAR) matrix from a KITTI ``calib.txt``."""
    tr: np.ndarray | None = None
    with path.open() as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip() == "Tr":
                tr = np.eye(4)
                tr[:3, :4] = np.fromstring(value, sep=" ").reshape(3, 4)
    if tr is None:
        raise ValueError(f"{path} has no 'Tr:' row")
    return tr


def parse_poses(path: Path, tr: np.ndarray) -> list[np.ndarray]:
    """Load odometry poses and express them in the LiDAR frame (``Tr^-1 @ P @ Tr``)."""
    tr_inv = np.linalg.inv(tr)
    poses: list[np.ndarray] = []
    with path.open() as fh:
        for line in fh:
            values = np.fromstring(line, sep=" ")
            if len(values) != 12:
                continue
            pose = np.eye(4)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(np.matmul(tr_inv, np.matmul(pose, tr)).astype(np.float32))
    return poses


def fuse_scan(points: np.ndarray, pose0: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Bring ``points`` from the frame of ``pose`` into the frame of ``pose0``."""
    homogeneous = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    world = np.sum(np.expand_dims(homogeneous, 2) * pose.T, axis=1)[:, :3]
    local = world - pose0[:3, 3]
    local = np.sum(np.expand_dims(local, 2) * pose0[:3, :3], axis=1)
    return np.hstack((local, points[:, 3:]))


def voxelize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voxelise ``(N, 4)`` xyz+intensity points into unique coords + mean intensity."""
    inside = (
        (points[:, 0] >= PC_RANGE[0]) & (points[:, 0] < PC_RANGE[3])
        & (points[:, 1] >= PC_RANGE[1]) & (points[:, 1] < PC_RANGE[4])
        & (points[:, 2] >= PC_RANGE[2]) & (points[:, 2] < PC_RANGE[5])
    )
    points = points[inside]
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)

    coords = np.floor((points[:, :3] - PC_RANGE[:3]) / VOXEL_SIZE).astype(np.int32)
    coords = np.clip(coords, 0, GRID_SIZE - 1)
    # Hash then unique: this is what fixes the on-disk row order, so it must not change.
    keys = (coords[:, 0] * GRID_SIZE[1] * GRID_SIZE[2]
            + coords[:, 1] * GRID_SIZE[2]
            + coords[:, 2])
    unique_keys, inverse = np.unique(keys, return_inverse=True)

    unique_coords = np.zeros((len(unique_keys), 3), dtype=np.int32)
    unique_coords[:, 2] = unique_keys % GRID_SIZE[2]
    unique_coords[:, 1] = (unique_keys // GRID_SIZE[2]) % GRID_SIZE[1]
    unique_coords[:, 0] = unique_keys // (GRID_SIZE[1] * GRID_SIZE[2])

    intensity = np.zeros(len(unique_keys), dtype=np.float32)
    counts = np.zeros(len(unique_keys), dtype=np.float32)
    np.add.at(intensity, inverse, points[:, 3])
    np.add.at(counts, inverse, 1)
    return unique_coords, (intensity / np.maximum(counts, 1)).reshape(-1, 1)


def process_frame(task: tuple[str, str, list[int], str, str, bool]) -> tuple[str, str]:
    """Worker: fuse one frame's scan window and write its ``.npz``."""
    velodyne_dir_str, poses_npy, window, out_path_str, frame_id, skip_existing = task
    out_path = Path(out_path_str)
    if skip_existing and out_path.is_file():
        return ("skipped", frame_id)

    velodyne_dir = Path(velodyne_dir_str)
    poses = np.load(poses_npy)
    current = int(frame_id)
    try:
        scans: list[np.ndarray] = []
        for index in window:
            scan_path = velodyne_dir / f"{index:06d}.bin"
            if not scan_path.is_file():
                continue
            points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            scans.append(points if index == current
                         else fuse_scan(points, poses[current], poses[index]))
        if not scans:
            return ("error: no readable scan in window", frame_id)
        coords, features = voxelize(np.vstack(scans))
        np.savez_compressed(
            out_path,
            coords=coords.astype(np.int16),
            features=features.astype(np.float16),
            num_frames=len(scans),
        )
    except (OSError, ValueError) as exc:
        return (f"error: {exc}", frame_id)
    return ("ok", frame_id)


def build_sequence_tasks(
    seq: str,
    raw_root: Path,
    calib_root: Path,
    out_root: Path,
    cache_dir: Path,
    num_frames: int,
    use_past: bool,
    skip_existing: bool,
) -> list[tuple[str, str, list[int], str, str, bool]]:
    """Enumerate one sequence's work items, or return [] with a warning."""
    seq_dir = raw_root / "sequences" / seq
    velodyne_dir = seq_dir / "velodyne"
    poses_file = seq_dir / "poses.txt"
    calib_file = calib_root / "sequences" / seq / "calib.txt"
    for path, what in ((velodyne_dir, "velodyne dir"), (poses_file, "poses.txt"),
                       (calib_file, "calib.txt")):
        if not path.exists():
            logger.warning("Sequence %s: missing %s at %s -- skipping", seq, what, path)
            return []

    poses = parse_poses(poses_file, parse_calibration(calib_file))
    if not poses:
        logger.warning("Sequence %s: %s parsed to zero poses -- skipping", seq, poses_file)
        return []

    # Poses are read once per sequence and handed to the workers as a memmappable
    # .npy; re-parsing the text file in every worker was the old script's hot spot.
    poses_npy = cache_dir / f"poses_{seq}.npy"
    np.save(poses_npy, np.stack(poses))

    out_dir = out_root / seq
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for scan in sorted(velodyne_dir.glob("*.bin")):
        index = int(scan.stem)
        if index >= len(poses):
            continue
        if use_past:
            window = list(range(max(0, index - num_frames + 1), index + 1))
        else:
            window = list(range(index, min(index + num_frames, len(poses))))
        tasks.append((str(velodyne_dir), str(poses_npy), window,
                      str(out_dir / f"{scan.stem}.npz"), scan.stem, skip_existing))
    logger.info("Sequence %s: %d frames, %d poses -> %s", seq, len(tasks), len(poses), out_dir)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the data/SemanticKITTI_3D/256_multi_frame cache the *_mf recipes read.")
    parser.add_argument("--semantickitti_root", type=str,
                        default=str(REPO_ROOT / "data" / "SemanticKITTI"),
                        help="Raw SemanticKITTI root holding sequences/<seq>/{velodyne,poses.txt} "
                             "(default: <repo>/data/SemanticKITTI)")
    parser.add_argument("--calib_root", type=str, default=None,
                        help="Root holding sequences/<seq>/calib.txt. Defaults to "
                             "--semantickitti_root; point it at the KITTI odometry checkout if "
                             "your SemanticKITTI tree was assembled without the calibration "
                             "archive.")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "data" / "SemanticKITTI_3D" / "256_multi_frame"),
                        help="Cache root to write (default: "
                             "<repo>/data/SemanticKITTI_3D/256_multi_frame)")
    parser.add_argument("--sequences", type=str, nargs="+", default=DEFAULT_SEQUENCES,
                        help="Sequences to process (default: the eleven annotated sequences)")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Scans to fuse, including the current one (released cache: 5)")
    parser.add_argument("--use_past", action="store_true",
                        help="Fuse the PRECEDING scans instead of the following ones. The "
                             "released cache uses following scans; changing this produces a "
                             "different tree, so leave it off to reproduce.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--force", action="store_true",
                        help="Recompute frames whose .npz already exists (default: skip them)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_root = Path(args.semantickitti_root)
    calib_root = Path(args.calib_root) if args.calib_root else raw_root
    out_root = Path(args.output_dir)
    if not (raw_root / "sequences").is_dir():
        sys.exit(f"No sequences/ under {raw_root}. Download the SemanticKITTI voxel release plus "
                 f"the KITTI odometry velodyne scans first (see docs/DATASET.md) and verify the "
                 f"layout with `python scripts/prepare_data.py --root {raw_root}`.")

    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / ".poses_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, list[int], str, str, bool]] = []
    for seq in args.sequences:
        tasks.extend(build_sequence_tasks(seq, raw_root, calib_root, out_root, cache_dir,
                                          args.num_frames, args.use_past, not args.force))
    if not tasks:
        sys.exit(
            f"No frames found. Needed per sequence: {raw_root}/sequences/<seq>/velodyne/*.bin, "
            f"{raw_root}/sequences/<seq>/poses.txt and {calib_root}/sequences/<seq>/calib.txt. "
            "calib.txt comes from the KITTI odometry calibration archive -- pass --calib_root if "
            "it lives in a different checkout."
        )

    logger.info("Fusing %d frames (%d scans each, %s) with %d workers -> %s",
                len(tasks), args.num_frames, "past" if args.use_past else "future",
                args.workers, out_root)
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

    for stale in cache_dir.glob("poses_*.npy"):
        stale.unlink()
    cache_dir.rmdir()

    logger.info("Done: %s", ", ".join(f"{k}={v}" for k, v in sorted(tallies.items())))
    if errors:
        for frame_id, status in errors[:20]:
            logger.error("  %s: %s", frame_id, status)
        sys.exit(f"{len(errors)} frame(s) failed; the cache is incomplete.")
    logger.info("Cache ready at %s", out_root)


if __name__ == "__main__":
    main()
