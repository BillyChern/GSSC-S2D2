"""Class-space bridging between SemanticKITTI (20-class) and SSCBench-KITTI360 (19-class).

Supports the zero-shot SSCBench-KITTI360 eval for GSSC.

The SemanticKITTI-trained SCPNet+S²D² pipeline emits predictions in the
SemanticKITTI **training space** (20 indices, 0 = empty, 1..19 semantic), with
the index ordering defined by ``configs/.../semantic-kitti.yaml`` ``learning_map``:

    0 empty | 1 car | 2 bicycle | 3 motorcycle | 4 truck | 5 other-vehicle
    6 person | 7 bicyclist | 8 motorcyclist | 9 road | 10 parking
    11 sidewalk | 12 other-ground | 13 building | 14 fence | 15 vegetation
    16 trunk | 17 terrain | 18 pole | 19 traffic-sign

SSCBench-KITTI360 uses a **different 19-class training space** (0 = empty,
1..18 semantic), per ``dataset/configs/kitti360.yaml`` ``learning_map`` in the
ai4ce/SSCBench repo:

    0 empty | 1 car | 2 bicycle | 3 motorcycle | 4 truck | 5 other-vehicle
    6 person | 7 road | 8 parking | 9 sidewalk | 10 other-ground
    11 building | 12 fence | 13 vegetation | 14 terrain | 15 pole
    16 traffic-sign | 17 other-structure | 18 other-object

The two spaces are NOT a renumbering of the same classes:

  * SSCBench-KITTI360 DROPS three SemanticKITTI classes:
        bicyclist (sk 7), motorcyclist (sk 8), trunk (sk 16)
    They are folded into "unlabeled" by the official KITTI360 learning_map,
    so they are absent from KITTI360 GT and ignored in scoring.
  * SSCBench-KITTI360 ADDS two classes that SemanticKITTI's training space
    does not predict:
        other-structure (k360 17), other-object (k360 18)
    A SemanticKITTI-trained model can NEVER emit these, so their IoU is
    structurally 0 (a real zero-shot domain-gap penalty, not a bug).

This module provides the projection SemanticKITTI-train-space -> KITTI360-train-space
so the SCPNet+S²D² output can be scored against SSCBench-KITTI360 GT under the
KITTI360 16-shared-class protocol. The mapping is by *semantic identity*:
shared classes map across; SemanticKITTI-only classes (bicyclist, motorcyclist,
trunk) map to 0 (empty/ignore) because KITTI360 GT has no such voxels to match.

The 16 classes that survive the intersection (and therefore actually score):
    car, bicycle, motorcycle, truck, other-vehicle, person, road, parking,
    sidewalk, other-ground, building, fence, vegetation, terrain, pole,
    traffic-sign.
This is the standard cross-dataset SemanticKITTI->KITTI360 evaluation subset.

References
----------
* ai4ce/SSCBench ``dataset/configs/kitti360.yaml`` (learning_map / learning_map_inv).
* SSCBench: A Large-Scale 3D Semantic Scene Completion Benchmark for Autonomous
  Driving (Li et al., 2023), Tab. 1 class list.
"""
from __future__ import annotations

import numpy as np

# --- SemanticKITTI training-space index -> class name (the GSSC/SCPNet space). ---
SEMANTICKITTI_TRAIN_NAMES: dict[int, str] = {
    0: "empty",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "bicyclist",
    8: "motorcyclist",
    9: "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",
}

# --- SSCBench-KITTI360 training-space index -> class name (the GT/scoring space). ---
KITTI360_TRAIN_NAMES: dict[int, str] = {
    0: "empty",
    1: "car",
    2: "bicycle",
    3: "motorcycle",
    4: "truck",
    5: "other-vehicle",
    6: "person",
    7: "road",
    8: "parking",
    9: "sidewalk",
    10: "other-ground",
    11: "building",
    12: "fence",
    13: "vegetation",
    14: "terrain",
    15: "pole",
    16: "traffic-sign",
    17: "other-structure",
    18: "other-object",
}

# Reverse lookup: KITTI360 class-name -> KITTI360 train index.
_K360_NAME_TO_IDX: dict[str, int] = {v: k for k, v in KITTI360_TRAIN_NAMES.items()}

# SemanticKITTI train classes that KITTI360 does not keep -> fold to empty (0).
_SK_ONLY_TO_EMPTY = {"bicyclist", "motorcyclist", "trunk"}


def build_sk2k360_lut() -> np.ndarray:
    """Build a 20-entry LUT: SemanticKITTI train idx -> KITTI360 train idx.

    SemanticKITTI-only classes (bicyclist, motorcyclist, trunk) and empty map
    to 0; every other class maps to its KITTI360 index by semantic identity.

    Returns:
        ``np.ndarray`` of shape (20,), dtype uint8, indexable by a 0-19 array
        of SemanticKITTI-train-space predictions.
    """
    lut = np.zeros(20, dtype=np.uint8)
    for sk_idx, name in SEMANTICKITTI_TRAIN_NAMES.items():
        if name == "empty" or name in _SK_ONLY_TO_EMPTY:
            lut[sk_idx] = 0
        else:
            lut[sk_idx] = _K360_NAME_TO_IDX[name]
    return lut


SK2K360_LUT: np.ndarray = build_sk2k360_lut()

# KITTI360 train indices that a SemanticKITTI-trained model can actually emit
# (i.e. the 16-class shared intersection). Used for reporting which classes are
# scorable vs structurally-zero (other-structure=17, other-object=18).
SCORABLE_K360_CLASSES: tuple[int, ...] = tuple(
    sorted({int(i) for i in SK2K360_LUT if i != 0})
)
STRUCTURAL_ZERO_K360_CLASSES: tuple[int, ...] = tuple(
    k for k in range(1, 19) if k not in SCORABLE_K360_CLASSES
)


def remap_scpnet_to_kitti360(pred_sk_train: np.ndarray) -> np.ndarray:
    """Project a SemanticKITTI-train-space voxel grid into KITTI360 train space.

    Args:
        pred_sk_train: integer array, values in [0, 19] (SemanticKITTI train).

    Returns:
        Array of the same shape with values in [0, 18] (KITTI360 train).
    """
    p = np.asarray(pred_sk_train)
    if p.min() < 0 or p.max() > 19:
        raise ValueError(
            f"SemanticKITTI-train pred out of range [0,19]: "
            f"min={p.min()}, max={p.max()}"
        )
    return SK2K360_LUT[p.astype(np.int64)]


# KITTI360 train idx -> KITTI360 *original* (0-255) label, for writing .label
# files the official scorer can read against KITTI360 GT. This is the
# ``learning_map_inv`` from dataset/configs/kitti360.yaml.
KITTI360_LEARNING_MAP_INV: np.ndarray = np.array(
    [
        0,    # 0 empty
        10,   # 1 car
        11,   # 2 bicycle
        15,   # 3 motorcycle
        18,   # 4 truck
        20,   # 5 other-vehicle
        30,   # 6 person
        40,   # 7 road
        44,   # 8 parking
        48,   # 9 sidewalk
        49,   # 10 other-ground
        50,   # 11 building
        51,   # 12 fence
        70,   # 13 vegetation
        72,   # 14 terrain
        80,   # 15 pole
        81,   # 16 traffic-sign
        52,   # 17 other-structure
        99,   # 18 other-object
    ],
    dtype=np.uint16,
)
