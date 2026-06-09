"""SemanticPOSS zero-shot SSC evaluation: dataset index + voxelizer + GT builder.

Lets the SemanticKITTI-trained SCPNet+S²D² pipeline run zero-shot on
SemanticPOSS val (sequence 02) WITHOUT touching the committed SemanticKITTI
paths, replicating the TALoS (jang2024talos, NeurIPS 2024) Table 4
cross-dataset protocol.

PROTOCOL (verified against the TALoS repo github.com/blue-531/talos and paper
Appendix A):

* **Voxel grid is the SemanticKITTI SSC grid**: 256x256x32 @ 0.2 m, front-facing
  region ``min=[0,-25.6,-2]`` -> ``max=[51.2,25.6,4.4]`` (TALoS
  ``semantickitti-tta-val.yaml`` min/max_volume_space, "256x256x32 following
  SCPNet [3]", Appendix A). SCPNet's ``prepare_input`` voxelizer needs NO geometry
  change for the SCPNet stage; this module reuses the SAME bounds for the GT.
* **GT construction** (the key detail): TALoS scores POSS seq 02 through its
  ``SemKITTI_sk_multiscan`` dataset with ``multiscan = 0`` (single scan, no
  accumulation; ``dataloader/pc_dataset.py`` L151). For ``imageset='val'`` it
  reads a PRE-EXISTING ``voxels/<frame>.label`` (uint16, flat, 256x256x32;
  L330) and a ``voxels/<frame>.invalid`` bit-packed occlusion mask. SemanticPOSS
  ships ONLY ``velodyne/<frame>.bin`` (float32 x,y,z,remission) + per-point
  ``labels/<frame>.label`` (uint32; low 16 bits = semantic, TALoS ``& 0xFFFF``).
  It does NOT ship voxelized ``voxels/`` SSC GT — so :func:`build_poss_voxel_gt`
  reproduces it: project each labelled point into the grid, per-voxel
  majority-vote class, and a ray-cast free-space ``.invalid`` mask (the standard
  single-scan SemanticKITTI voxelizer convention). Output ``.label``/``.invalid``
  are written into a ``voxels/`` dir so the SCPNet runner + scorer consume them
  exactly like SemanticKITTI.
* **GT label space** is POSS ORIGINAL labels (0-255). The scorer
  (``external/semantic_kitti_api/evaluate_completion.py``) remaps both GT and
  predictions via the POSS datacfg ``learning_map`` (11 classes), then scores
  cIoU (completion IoU) + mIoU. Predictions are projected SemanticKITTI->POSS via
  ``gssc.data.poss_class_map`` and written back in POSS original labels.
* **Val split** = sequence ``02`` (the only POSS eval seq, TALoS Sec. 4.5 /
  jang2024talos; JS3C-Net SemanticPOSS.yaml valid:[2]).

NOTE on GT comparability: TALoS does not release its POSS ``voxels/`` pack, so the
reproduced GT here is a *best-effort replica* of the single-scan SemanticKITTI
voxelizer. The ``.invalid`` ray-cast and the majority-vote tie-breaking are the
two places where a small numeric gap vs the TALoS number can arise, so the
resulting row should be reported as "our voxelization of POSS GT" rather than
claimed byte-identical to TALoS's.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# SemanticKITTI SSC grid + front-facing region (matches SCPNet / TALoS exactly).
GRID_SIZE: tuple[int, int, int] = (256, 256, 32)
MIN_VOLUME_SPACE: tuple[float, float, float] = (0.0, -25.6, -2.0)
MAX_VOLUME_SPACE: tuple[float, float, float] = (51.2, 25.6, 4.4)
VOXEL_SIZE: float = 0.2  # (51.2-0)/256 = 25.6*2/256 = (4.4-(-2))/32 = 0.2 m isotropic.

NUM_POSS_VOXELS = GRID_SIZE[0] * GRID_SIZE[1] * GRID_SIZE[2]  # 2,097,152

# SemanticPOSS split (JS3C-Net SemanticPOSS.yaml / TALoS semantic-poss-multiscan.yaml).
POSS_SPLITS: dict[str, list[str]] = {
    "train": ["00", "01", "03", "04", "05"],
    "val": ["02"],
    "test": ["02"],
}

# POSS raw label (low 16 bits) -> POSS 11-class learning id. From JS3C-Net
# SemanticPOSS.yaml ``learning_map`` (identical to TALoS semantic-poss-multiscan).
POSS_LEARNING_MAP: dict[int, int] = {
    0: 0,   # unlabeled
    4: 1,   # 1 person  -> people
    5: 1,   # 2+ person -> people
    6: 2,   # rider
    7: 3,   # car
    8: 4,   # trunk
    9: 5,   # plants
    10: 6,  # traffic sign 1 -> traffic-sign
    11: 6,  # traffic sign 2 -> traffic-sign
    12: 6,  # traffic sign 3 -> traffic-sign
    13: 7,  # pole
    14: 0,  # trashcan -> unlabeled
    15: 8,  # building
    16: 0,  # cone/stone -> unlabeled
    17: 9,  # fence
    21: 10, # bike
    22: 11, # ground
}

# POSS learning id -> POSS original label (for writing GT .label in 0-255 space).
from gssc.data.poss_class_map import SEMANTICPOSS_LEARNING_MAP_INV  # noqa: E402


def _build_poss_remap_lut() -> np.ndarray:
    """LUT mapping POSS raw (low-16-bit) label -> POSS learning id (0-11)."""
    maxkey = max(POSS_LEARNING_MAP.keys())
    lut = np.zeros(maxkey + 1, dtype=np.uint8)
    for k, v in POSS_LEARNING_MAP.items():
        lut[k] = v
    return lut


POSS_REMAP_LUT: np.ndarray = _build_poss_remap_lut()


def load_poss_scan(velodyne_path: str | os.PathLike) -> np.ndarray:
    """Load a SemanticPOSS Velodyne scan -> ``(N, 4)`` float32 (x, y, z, remission).

    Same format as SemanticKITTI. SCPNet's ``prepare_input`` consumes this (N,4)
    array directly for the SCPNet zero-shot stage.
    """
    pc = np.fromfile(velodyne_path, dtype=np.float32)
    return pc.reshape(-1, 4)


def load_poss_point_labels(label_path: str | os.PathLike) -> np.ndarray:
    """Load SemanticPOSS per-point labels -> ``(N,)`` POSS learning ids (0-11).

    POSS ``.label`` is uint32 per point; the low 16 bits hold the semantic class
    (TALoS: ``annotated_data & 0xFFFF``). We mask then remap to the 11-class
    learning space.
    """
    raw = np.fromfile(label_path, dtype=np.uint32)
    sem = (raw & 0xFFFF).astype(np.int64)
    sem = np.clip(sem, 0, POSS_REMAP_LUT.shape[0] - 1)
    return POSS_REMAP_LUT[sem]


def _points_to_grid_indices(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map metric xyz -> integer voxel indices in the SemanticKITTI SSC grid.

    Returns the in-bounds voxel indices ``[M, 3]`` (int64) and the boolean
    in-bounds mask ``[N]`` so per-point labels can be subset consistently.
    """
    mn = np.asarray(MIN_VOLUME_SPACE, dtype=np.float64)
    mx = np.asarray(MAX_VOLUME_SPACE, dtype=np.float64)
    inb = np.all((xyz >= mn) & (xyz < mx), axis=1)
    idx = np.floor((xyz[inb] - mn) / VOXEL_SIZE).astype(np.int64)
    # Clamp the rare boundary point that lands exactly on the upper edge.
    for c in range(3):
        np.clip(idx[:, c], 0, GRID_SIZE[c] - 1, out=idx[:, c])
    return idx, inb


def voxelize_labels_majority(xyz: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
    """Per-voxel majority-vote semantic label -> ``[256,256,32]`` POSS learning ids.

    For each occupied voxel, the winning class is the most frequent NON-empty
    point label among the points falling in it (ignore-class 0 points still mark
    occupancy but only win a voxel if no semantic point is present). Empty voxels
    stay 0. This is the standard single-scan SemanticKITTI SSC GT convention.

    Args:
        xyz: ``[N, 3]`` metric point coordinates.
        point_labels: ``[N]`` POSS learning ids (0-11) for each point.

    Returns:
        ``[256, 256, 32]`` uint8 voxel grid of POSS learning ids (0 = empty).
    """
    idx, inb = _points_to_grid_indices(xyz)
    lab = point_labels[inb]
    flat = (idx[:, 0] * GRID_SIZE[1] * GRID_SIZE[2]
            + idx[:, 1] * GRID_SIZE[2]
            + idx[:, 2])

    num_classes = int(SEMANTICPOSS_LEARNING_MAP_INV.shape[0])  # 12 (0..11)
    # votes[v, c] = #points of class c in voxel v. Bincount over flat*C + class.
    combo = flat * num_classes + lab
    counts = np.bincount(combo, minlength=NUM_POSS_VOXELS * num_classes)
    counts = counts.reshape(NUM_POSS_VOXELS, num_classes)
    # Zero out the empty/ignore column so a voxel only takes class 0 if it has
    # NO semantic points; otherwise argmax picks the dominant semantic class.
    semantic_counts = counts.copy()
    semantic_counts[:, 0] = 0
    has_semantic = semantic_counts.sum(axis=1) > 0
    winner = np.zeros(NUM_POSS_VOXELS, dtype=np.uint8)
    winner[has_semantic] = semantic_counts[has_semantic].argmax(axis=1).astype(np.uint8)
    return winner.reshape(GRID_SIZE)


def build_invalid_mask(xyz: np.ndarray) -> np.ndarray:
    """Build the ``.invalid`` mask (1 = ignore in eval) via per-column ray casting.

    SemanticKITTI marks a voxel INVALID when it lies BEYOND the deepest laser
    return along the sensor line of sight (never observed -> excluded from cIoU /
    mIoU). A faithful replica casts a ray per point; here we use the column
    approximation that matches the front-facing grid well: within each (x,y)
    BEV column, voxels ABOVE the highest occupied voxel in that column AND
    columns with NO returns are marked invalid. Voxels at/below the highest
    return are valid (observed free space or occupied).

    NB: this column approximation is the principal source of any small numeric
    gap vs TALoS's exact ray-traced ``.invalid``. For a byte-faithful replica,
    swap this for a Bresenham 3D ray-cast from the sensor origin (voxel index of
    metric (0,0,0)) to each point; until then the reported number should be
    labelled "our voxelization".

    Returns:
        ``[256, 256, 32]`` uint8 mask, 1 = invalid (ignored), 0 = valid.
    """
    idx, _ = _points_to_grid_indices(xyz)
    invalid = np.ones(GRID_SIZE, dtype=np.uint8)  # default: unobserved -> invalid
    occ = np.zeros(GRID_SIZE, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    # Highest occupied z per (x,y) column; -1 if column empty.
    z_any = occ.any(axis=2)
    # argmax of reversed z gives topmost occupied; map back.
    top_from_rev = np.argmax(occ[:, :, ::-1], axis=2)
    z_top = np.where(z_any, GRID_SIZE[2] - 1 - top_from_rev, -1)
    zz = np.arange(GRID_SIZE[2])[None, None, :]
    valid = z_any[:, :, None] & (zz <= z_top[:, :, None])
    invalid[valid] = 0
    return invalid


def pack_binary_mask(flat: np.ndarray) -> np.ndarray:
    """Bit-pack a flat 0/1 uint8 mask into the SemanticKITTI byte format."""
    assert flat.shape[0] % 8 == 0, "mask length must be divisible by 8"
    out = np.zeros(flat.shape[0] // 8, dtype=np.uint8)
    for i in range(8):
        out += (flat[i::8] & 1) << (7 - i)
    return out


def pack_occupancy_bin(occ_grid: np.ndarray) -> np.ndarray:
    """Bit-pack a 0/1 occupancy grid into the SemanticKITTI ``voxels/.bin`` format.

    The SCPNet runner and the S²D² LiDAR encoder both enumerate / consume a
    bit-packed ``voxels/<frame>.bin``; this lets the POSS frames slot into both
    consumers unchanged.
    """
    return pack_binary_mask(np.asarray(occ_grid).reshape(-1).astype(np.uint8))


def build_poss_voxel_gt(
    velodyne_path: str | os.PathLike,
    point_label_path: str | os.PathLike,
    out_voxels_dir: str | os.PathLike,
    frame_id: str,
) -> None:
    """Materialise POSS single-scan SSC GT for one frame into ``out_voxels_dir``.

    Writes three SemanticKITTI-format files the SCPNet runner + scorer consume:
      * ``<frame>.label``   uint16 flat (POSS ORIGINAL 0-255 labels), 256x256x32
      * ``<frame>.invalid`` bit-packed occlusion mask (scorer bit-unpacks it)
      * ``<frame>.bin``     bit-packed occupancy (frame list + S²D² conditioning)

    Args:
        velodyne_path: POSS ``velodyne/<frame>.bin`` (float32 x,y,z,remission).
        point_label_path: POSS ``labels/<frame>.label`` (uint32 per point).
        out_voxels_dir: destination ``.../sequences/02/voxels``.
        frame_id: 6-digit frame id (e.g. ``"000000"``).
    """
    out_dir = Path(out_voxels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = load_poss_scan(velodyne_path)
    xyz = scan[:, :3]
    plabels = load_poss_point_labels(point_label_path)  # [N], POSS learning ids

    # GT voxel labels in POSS learning space, then to POSS original (0-255).
    vox_learn = voxelize_labels_majority(xyz, plabels)            # [H,W,D], 0-11
    vox_original = SEMANTICPOSS_LEARNING_MAP_INV[vox_learn.astype(np.int64)]
    vox_original.reshape(-1).astype(np.uint16).tofile(out_dir / f"{frame_id}.label")

    # Invalid (occlusion) mask -> bit-packed.
    invalid = build_invalid_mask(xyz)
    pack_binary_mask(invalid.reshape(-1).astype(np.uint8)).tofile(
        out_dir / f"{frame_id}.invalid"
    )

    # Occupancy input -> bit-packed voxels/.bin (drives the frame list + S²D²).
    idx, _ = _points_to_grid_indices(xyz)
    occ = np.zeros(GRID_SIZE, dtype=np.uint8)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    pack_occupancy_bin(occ).tofile(out_dir / f"{frame_id}.bin")


def read_poss_invalid_packed(invalid_path: str | os.PathLike) -> np.ndarray:
    """Bit-unpack a POSS ``.invalid`` written by :func:`build_poss_voxel_gt`."""
    compressed = np.fromfile(invalid_path, dtype=np.uint8)
    out = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = (compressed >> (7 - i)) & 1
    return out[:NUM_POSS_VOXELS].reshape(GRID_SIZE)


def load_occupancy_bin(bin_path: str | os.PathLike):
    """Load a bit-packed POSS occupancy ``voxels/.bin`` -> [1,1,256,256,32] tensor.

    Mirrors ``gssc.inference.generate_predictions.load_lidar_voxels`` so the S²D²
    LiDAR-conditioning path is byte-identical between SemanticKITTI and POSS.
    """
    import torch

    compressed = np.fromfile(bin_path, dtype=np.uint8)
    out = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    for i in range(8):
        out[i::8] = (compressed >> (7 - i)) & 1
    binary = out[:NUM_POSS_VOXELS].reshape(GRID_SIZE).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)


@dataclass(frozen=True)
class SemanticPossEvalConfig:
    """Immutable config for the SemanticPOSS zero-shot eval.

    Attributes:
        poss_root: Root containing ``sequences/<seq>/{velodyne,labels}`` (the raw
            SemanticPOSS release).
        voxel_gt_root: Where the materialised SSC GT (``sequences/<seq>/voxels``)
            is written by :func:`build_poss_voxel_gt`. May equal ``poss_root``.
        split: One of ``{"train", "val", "test"}``. ``"val"`` = seq 02.
    """

    poss_root: str
    voxel_gt_root: str
    split: str = "val"
    sequences: tuple[str, ...] = field(default=())

    def resolved_sequences(self) -> list[str]:
        return list(self.sequences) if self.sequences else POSS_SPLITS[self.split]


class SemanticPossSSCDataset:
    """Index over SemanticPOSS frames for the zero-shot SCPNet+S²D² eval.

    Thin on-disk index (no eager tensor loading). Consumers iterate ``frames``
    and pull: raw LiDAR (SCPNet stage), occupancy ``voxels/.bin`` (S²D²
    conditioning), and GT ``.label``/``.invalid`` (scoring). The ``voxels/`` files
    must be materialised first via :func:`materialise_gt`.
    """

    def __init__(self, cfg: SemanticPossEvalConfig) -> None:
        self.cfg = cfg
        self.sequences = cfg.resolved_sequences()
        self.frames: list[dict[str, str]] = self._index_frames()

    def _velodyne_dir(self, seq: str) -> Path:
        return Path(self.cfg.poss_root) / "sequences" / seq / "velodyne"

    def _labels_dir(self, seq: str) -> Path:
        return Path(self.cfg.poss_root) / "sequences" / seq / "labels"

    def _voxels_dir(self, seq: str) -> Path:
        return Path(self.cfg.voxel_gt_root) / "sequences" / seq / "voxels"

    def _index_frames(self) -> list[dict[str, str]]:
        frames: list[dict[str, str]] = []
        for seq in self.sequences:
            vel_dir = self._velodyne_dir(seq)
            if not vel_dir.is_dir():
                raise FileNotFoundError(
                    f"SemanticPOSS velodyne dir missing: {vel_dir}. "
                    "Provision the raw SemanticPOSS release first (see "
                    "configs/eval/semanticposs_seq02.yaml)."
                )
            for vf in sorted(vel_dir.glob("*.bin")):
                frame_id = vf.stem
                frames.append(
                    {
                        "sequence": seq,
                        "frame_id": frame_id,
                        "velodyne_path": str(vf),
                        "point_label_path": str(self._labels_dir(seq) / f"{frame_id}.label"),
                        "label_path": str(self._voxels_dir(seq) / f"{frame_id}.label"),
                        "invalid_path": str(self._voxels_dir(seq) / f"{frame_id}.invalid"),
                        "occupancy_bin_path": str(self._voxels_dir(seq) / f"{frame_id}.bin"),
                    }
                )
        return frames

    def materialise_gt(self, skip_existing: bool = True) -> int:
        """Voxelise every frame's POSS GT into ``voxel_gt_root`` (one-time prep).

        Returns the number of frames written. Idempotent when ``skip_existing``.
        """
        written = 0
        for f in self.frames:
            if skip_existing and os.path.exists(f["label_path"]) \
                    and os.path.exists(f["invalid_path"]) \
                    and os.path.exists(f["occupancy_bin_path"]):
                continue
            build_poss_voxel_gt(
                velodyne_path=f["velodyne_path"],
                point_label_path=f["point_label_path"],
                out_voxels_dir=Path(f["label_path"]).parent,
                frame_id=f["frame_id"],
            )
            written += 1
        return written

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.frames[idx]
