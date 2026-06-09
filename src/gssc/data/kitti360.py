"""SSCBench-KITTI360 zero-shot evaluation dataset + voxelizer.

Lets the SemanticKITTI-trained SCPNet+S²D² pipeline run zero-shot on
SSCBench-KITTI360 val (scene 06) without touching the committed SemanticKITTI
paths.

Key compatibility facts (verified against ai4ce/SSCBench + cvlibs KITTI-360):

* **Voxel grid is identical to SemanticKITTI**: 256x256x32 @ 0.2 m, front-facing
  region ``min=[0,-25.6,-2]`` -> ``max=[51.2,25.6,4.4]``. This matches SCPNet's
  ``semantickitti-multiscan.yaml`` (min/max_volume_space) EXACTLY, so SCPNet's
  ``prepare_input`` voxelizer needs no geometry change — KITTI-360 also uses the
  Velodyne HDL-64E.
* **GT files**: ``voxels/<frame>.label`` (uint16, flat, NOT bit-packed),
  ``voxels/<frame>.invalid`` (uint8, flat, NOT bit-packed),
  ``voxels/<frame>.bin`` (occupancy input). NB: SSCBench ``.invalid`` is RAW
  uint8 per-voxel, whereas SemanticKITTI ``.invalid`` is BIT-PACKED — the GSSC
  scorer (``external/semantic_kitti_api/evaluate_completion.py``) bit-unpacks
  ``.invalid``, so KITTI-360 ``.invalid`` must be repacked before scoring (see
  :func:`repack_invalid_for_semantic_kitti_api`).
* **GT label space** is the KITTI-360 19-class training space (0..18); predictions
  from the SemanticKITTI-trained pipeline must be projected with
  ``gssc.data.kitti360_class_map.remap_scpnet_to_kitti360`` and written back to
  KITTI-360 original (0-255) labels for the official scorer.
* **Val split** = sequence ``2013_05_28_drive_0006_sync`` (1,812 frames, sampled
  every 5 raw scans).

Raw LiDAR for SCPNet input lives at
``data_3d_raw/<sequence>/velodyne_points/data/{original_id:010d}.bin`` (float32
x,y,z,intensity), where ``original_id`` is mapped from the 6-digit voxel
``frame_id`` via the SSCBench frame-match table (``kitti_360_match.txt``).
SSCBench also ships a pre-voxelized occupancy ``voxels/<frame>.bin`` which the
S²D² stage consumes directly as the binary LiDAR conditioning grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SemanticKITTI / KITTI-360 share this grid + region (HDL-64E, 0.2 m voxels).
GRID_SIZE: tuple[int, int, int] = (256, 256, 32)
MIN_VOLUME_SPACE: tuple[float, float, float] = (0.0, -25.6, -2.0)
MAX_VOLUME_SPACE: tuple[float, float, float] = (51.2, 25.6, 4.4)

# Default SSCBench-KITTI360 split sequences (cvlibs KITTI-360 drive names).
KITTI360_SPLITS: dict[str, list[str]] = {
    "train": [
        "2013_05_28_drive_0000_sync",
        "2013_05_28_drive_0002_sync",
        "2013_05_28_drive_0003_sync",
        "2013_05_28_drive_0004_sync",
        "2013_05_28_drive_0005_sync",
        "2013_05_28_drive_0007_sync",
        "2013_05_28_drive_0010_sync",
    ],
    "val": ["2013_05_28_drive_0006_sync"],
    "test": ["2013_05_28_drive_0009_sync"],
}

NUM_KITTI360_VOXELS = GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]  # 2,097,152


def load_frame_match_table(match_path: str | os.PathLike) -> dict[str, dict[str, str]]:
    """Parse the SSCBench frame-id <-> raw-velodyne-id match table.

    Each line is ``<sequence> <raw_id>.png <voxel_frame_id>.png`` (the PaSCo /
    SSCBench ``kitti_360_match.txt`` format). We key by the 6-digit voxel frame
    id and store the zero-padded 10-digit raw id used to index ``data_3d_raw``.

    Returns:
        ``{sequence: {voxel_frame_id: raw_velodyne_id}}``.
    """
    table: dict[str, dict[str, str]] = {}
    with open(match_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            seq, raw_png, voxel_png = parts
            raw_id = os.path.splitext(raw_png)[0]      # e.g. 0000000009
            voxel_id = os.path.splitext(voxel_png)[0]  # e.g. 000000
            table.setdefault(seq, {})[voxel_id] = raw_id
    return table


def read_kitti360_label(label_path: str | os.PathLike) -> np.ndarray:
    """Read a KITTI-360 voxel ``.label`` file (flat uint16, 256x256x32)."""
    labels = np.fromfile(label_path, dtype=np.uint16)
    if labels.size != NUM_KITTI360_VOXELS:
        raise ValueError(
            f"{label_path}: expected {NUM_KITTI360_VOXELS} uint16 voxels, "
            f"got {labels.size}"
        )
    return labels.reshape(GRID_SIZE)


def read_kitti360_invalid(invalid_path: str | os.PathLike) -> np.ndarray:
    """Read a KITTI-360 ``.invalid`` mask (RAW uint8 per voxel, NOT bit-packed)."""
    invalid = np.fromfile(invalid_path, dtype=np.uint8)
    if invalid.size != NUM_KITTI360_VOXELS:
        raise ValueError(
            f"{invalid_path}: expected {NUM_KITTI360_VOXELS} uint8 voxels, "
            f"got {invalid.size} (KITTI-360 .invalid is raw, not bit-packed)"
        )
    return invalid.reshape(GRID_SIZE)


def pack_binary_mask(flat: np.ndarray) -> np.ndarray:
    """Bit-pack a flat 0/1 uint8 mask into the SemanticKITTI byte format."""
    assert flat.shape[0] % 8 == 0, "mask length must be divisible by 8"
    out = np.zeros(flat.shape[0] // 8, dtype=np.uint8)
    for i in range(8):
        out += (flat[i::8] & 1) << (7 - i)
    return out


def repack_invalid_for_semantic_kitti_api(
    invalid_grid: np.ndarray, out_path: str | os.PathLike
) -> None:
    """Convert a raw KITTI-360 ``.invalid`` grid to the bit-packed form the
    GSSC scorer expects, and write it next to the prediction.

    ``external/semantic_kitti_api/evaluate_completion.py`` reads
    ``<frame>.invalid`` with ``unpack(np.fromfile(..., uint8))`` (bit-packed,
    262,144 bytes -> 2,097,152 voxels). KITTI-360 ships it raw (2,097,152
    bytes). This repacks so the scorer's eval-mask is correct.
    """
    flat = np.asarray(invalid_grid).reshape(-1).astype(np.uint8)
    pack_binary_mask(flat).tofile(out_path)


def load_lidar_raw(velodyne_path: str | os.PathLike) -> np.ndarray:
    """Load a KITTI-360 raw Velodyne scan -> ``(N, 4)`` float32 (x, y, z, i).

    SCPNet's ``prepare_input`` voxelizer takes exactly this ``(N, 4)`` array and
    voxelizes it into the 256x256x32 grid.
    """
    pc = np.fromfile(velodyne_path, dtype=np.float32)
    return pc.reshape(-1, 4)


@dataclass(frozen=True)
class Kitti360EvalConfig:
    """Immutable config for the SSCBench-KITTI360 zero-shot eval.

    Attributes:
        kitti360_root: Root containing ``data_3d_raw/<seq>/velodyne_points/data``
            and ``data_2d_raw/<seq>/voxels`` (SSCBench-KITTI360 layout after
            unsquashfs).
        match_table: Path to ``kitti_360_match.txt``.
        split: One of ``{"train", "val", "test"}``. ``"val"`` = scene 06.
        sample_stride: SSCBench samples every 5 raw scans; the shipped voxel
            frames already encode this, so the loader just enumerates the
            ``voxels/*.label`` files.
    """

    kitti360_root: str
    match_table: str
    split: str = "val"
    sample_stride: int = 5


class Kitti360SSCDataset:
    """Enumerates SSCBench-KITTI360 frames for the zero-shot SCPNet+S²D² eval.

    This is a thin index over the on-disk layout; it does NOT load tensors
    eagerly. ``generate_predictions``-style consumers iterate ``frames`` and
    pull raw LiDAR (for SCPNet) + the occupancy ``voxels/.bin`` (for S²D²
    conditioning) + the GT ``.label``/``.invalid`` (for scoring) per frame.
    """

    def __init__(self, cfg: Kitti360EvalConfig) -> None:
        self.cfg = cfg
        if cfg.split not in KITTI360_SPLITS:
            raise ValueError(f"Unknown split {cfg.split!r}; choose from {list(KITTI360_SPLITS)}")
        self.sequences = KITTI360_SPLITS[cfg.split]
        self.match = load_frame_match_table(cfg.match_table)
        self.frames: list[dict[str, str]] = self._index_frames()

    def _voxels_dir(self, sequence: str) -> Path:
        # SSCBench-KITTI360 nests voxels under data_2d_raw/<seq>/voxels.
        return Path(self.cfg.kitti360_root) / "data_2d_raw" / sequence / "voxels"

    def _velodyne_dir(self, sequence: str) -> Path:
        return Path(self.cfg.kitti360_root) / "data_3d_raw" / sequence / "velodyne_points" / "data"

    def _index_frames(self) -> list[dict[str, str]]:
        frames: list[dict[str, str]] = []
        for seq in self.sequences:
            vox_dir = self._voxels_dir(seq)
            velo_dir = self._velodyne_dir(seq)
            label_files = sorted(vox_dir.glob("*.label"))
            for lf in label_files:
                frame_id = lf.stem  # 6-digit voxel id

                # VERIFIED (this release): the SSCBench-KITTI360 velodyne scans are
                # repackaged with 6-digit filenames that align 1:1 with the voxel
                # frame ids (velodyne/<frame_id>.bin <-> voxels/<frame_id>.*). The
                # 10-digit raw ids in kitti_360_match.txt are NOT used as filenames
                # here. Occupancy-IoU sweep over +/-6 frame offsets confirmed offset
                # 0 (direct same-id) is the unique best alignment at every probe
                # frame, so we index velodyne directly by frame_id. The match-table
                # raw_id is still recorded for provenance.
                direct_velo = velo_dir / f"{frame_id}.bin"
                raw_id = self.match.get(seq, {}).get(frame_id, frame_id.zfill(10))
                if direct_velo.exists():
                    velodyne_path = str(direct_velo)
                else:
                    # Fallback for an upstream layout that keeps 10-digit raw names.
                    velodyne_path = str(velo_dir / f"{raw_id}.bin")

                frames.append(
                    {
                        "sequence": seq,
                        "frame_id": frame_id,
                        "raw_id": raw_id,
                        "label_path": str(lf),
                        "invalid_path": str(lf.with_suffix(".invalid")),
                        "occupancy_bin_path": str(lf.with_suffix(".bin")),
                        "velodyne_path": velodyne_path,
                    }
                )
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.frames[idx]
