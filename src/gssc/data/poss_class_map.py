"""Class-space bridging between SemanticKITTI (20-class) and SemanticPOSS (11-class).

Supports the zero-shot SemanticKITTI->SemanticPOSS eval for GSSC, replicating the
TALoS (jang2024talos, NeurIPS 2024) Table 4 cross-dataset protocol.

The SemanticKITTI-trained SCPNet+S²D² pipeline emits predictions in the
SemanticKITTI **training space** (20 indices, 0 = empty, 1..19 semantic), with
the index ordering defined by ``external/semantic_kitti_api/config/semantic-kitti.yaml``
``learning_map``::

    0 empty | 1 car | 2 bicycle | 3 motorcycle | 4 truck | 5 other-vehicle
    6 person | 7 bicyclist | 8 motorcyclist | 9 road | 10 parking
    11 sidewalk | 12 other-ground | 13 building | 14 fence | 15 vegetation
    16 trunk | 17 terrain | 18 pole | 19 traffic-sign

SemanticPOSS uses an 11-class training space (0 = empty/ignore, 1..11 semantic),
per the official JS3C-Net ``SemanticPOSS.yaml`` / TALoS ``semantic-poss-multiscan.yaml``
``learning_map`` (both identical)::

    0 unlabeled | 1 people | 2 rider | 3 car | 4 trunk | 5 plants
    6 traffic-sign | 7 pole | 8 building | 9 fence | 10 bike | 11 ground

The SemanticKITTI->SemanticPOSS mapping is TALoS Table A.1 ("following [41,42]",
Kim et al. CVPR 2023 single-domain-generalization + Alonso et al. 2020):

    KITTI                       -> POSS
    car                         -> car
    bicycle                     -> bike
    person                      -> person/people
    bicyclist                   -> rider
    road, sidewalk              -> ground
    building                    -> building
    fence                       -> fence
    vegetation                  -> plants
    trunk                       -> trunk
    pole                        -> pole
    traffic sign                -> traffic-sign

Every other SemanticKITTI class (motorcycle, truck, other-vehicle, motorcyclist,
parking, other-ground, terrain) has NO POSS equivalent in Table A.1 and folds to
0 (empty/ignore): POSS GT has no such voxels, so they cannot match and are not
scored. This is a real cross-dataset label-set mismatch (it lowers the ceiling),
not a bug — exactly the regime TALoS reports ("mIoU is actually low ... due to the
severe drastic gap").

The 11 POSS classes that survive the intersection and can actually be emitted by
the SemanticKITTI-trained model: every POSS class 1..11 is reachable EXCEPT none
are structurally unreachable (all 11 have at least one KITTI source class), so all
11 are scorable. (Compare to KITTI-360 where other-structure/other-object are
structurally zero.)

References
----------
* TALoS (Jang et al., NeurIPS 2024), Table 4 + Table A.1 + Appendix A.
* JS3C-Net ``opt/SemanticPOSS.yaml`` (learning_map / learning_map_inv) — the
  same 11-class POSS map TALoS uses (``semantic-poss-multiscan.yaml``).
* Kim et al., "Single Domain Generalization for LiDAR Semantic Segmentation",
  CVPR 2023 (TALoS ref [41]); Alonso et al., arXiv:2010.12239 (ref [42]).
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

# --- SemanticPOSS training-space index -> class name (the GT/scoring space). ---
# Identical ordering to JS3C-Net SemanticPOSS.yaml / TALoS semantic-poss-multiscan.yaml.
SEMANTICPOSS_TRAIN_NAMES: dict[int, str] = {
    0: "unlabeled",
    1: "people",
    2: "rider",
    3: "car",
    4: "trunk",
    5: "plants",
    6: "traffic-sign",
    7: "pole",
    8: "building",
    9: "fence",
    10: "bike",
    11: "ground",
}

# Reverse lookup: POSS class-name -> POSS train index.
_POSS_NAME_TO_IDX: dict[str, int] = {v: k for k, v in SEMANTICPOSS_TRAIN_NAMES.items()}

# TALoS Table A.1: SemanticKITTI class name -> SemanticPOSS class name.
# "person" maps to POSS "people"; KITTI classes absent from Table A.1 fold to empty.
_SK2POSS_NAME: dict[str, str] = {
    "car": "car",
    "bicycle": "bike",
    "person": "people",
    "bicyclist": "rider",
    "road": "ground",
    "sidewalk": "ground",
    "building": "building",
    "fence": "fence",
    "vegetation": "plants",
    "trunk": "trunk",
    "pole": "pole",
    "traffic-sign": "traffic-sign",
}

# SemanticKITTI train classes with NO POSS counterpart in Table A.1 -> empty (0).
_SK_ONLY_TO_EMPTY = {
    "motorcycle",
    "truck",
    "other-vehicle",
    "motorcyclist",
    "parking",
    "other-ground",
    "terrain",
}


def build_sk2poss_lut() -> np.ndarray:
    """Build a 20-entry LUT: SemanticKITTI train idx -> SemanticPOSS train idx.

    Maps each SemanticKITTI training class to its SemanticPOSS counterpart per
    TALoS Table A.1; SemanticKITTI-only classes and empty map to 0.

    Returns:
        ``np.ndarray`` of shape (20,), dtype uint8, indexable by a 0-19 array
        of SemanticKITTI-train-space predictions.
    """
    lut = np.zeros(20, dtype=np.uint8)
    for sk_idx, name in SEMANTICKITTI_TRAIN_NAMES.items():
        if name == "empty" or name in _SK_ONLY_TO_EMPTY:
            lut[sk_idx] = 0
        else:
            lut[sk_idx] = _POSS_NAME_TO_IDX[_SK2POSS_NAME[name]]
    return lut


SK2POSS_LUT: np.ndarray = build_sk2poss_lut()

# POSS train indices a SemanticKITTI-trained model can actually emit. All 11 POSS
# classes have >=1 KITTI source in Table A.1, so all are scorable (none are
# structurally zero, unlike KITTI-360 other-structure/other-object).
SCORABLE_POSS_CLASSES: tuple[int, ...] = tuple(
    sorted({int(i) for i in SK2POSS_LUT if i != 0})
)
STRUCTURAL_ZERO_POSS_CLASSES: tuple[int, ...] = tuple(
    k for k in range(1, 12) if k not in SCORABLE_POSS_CLASSES
)


def remap_scpnet_to_poss(pred_sk_train: np.ndarray) -> np.ndarray:
    """Project a SemanticKITTI-train-space voxel grid into SemanticPOSS train space.

    Args:
        pred_sk_train: integer array, values in [0, 19] (SemanticKITTI train).

    Returns:
        Array of the same shape with values in [0, 11] (SemanticPOSS train).
    """
    p = np.asarray(pred_sk_train)
    if p.min() < 0 or p.max() > 19:
        raise ValueError(
            f"SemanticKITTI-train pred out of range [0,19]: "
            f"min={p.min()}, max={p.max()}"
        )
    return SK2POSS_LUT[p.astype(np.int64)]


# POSS train idx -> POSS *original* (0-255) label, for writing .label files the
# official scorer reads against POSS GT. This is the ``learning_map_inv`` from
# JS3C-Net SemanticPOSS.yaml (and configs/.../semantic-poss.yaml in this repo).
SEMANTICPOSS_LEARNING_MAP_INV: np.ndarray = np.array(
    [
        0,    # 0 unlabeled
        4,    # 1 people   (POSS raw "1 person")
        6,    # 2 rider
        7,    # 3 car
        8,    # 4 trunk
        9,    # 5 plants
        10,   # 6 traffic-sign (POSS raw "traffic sign 1")
        13,   # 7 pole
        15,   # 8 building
        17,   # 9 fence
        21,   # 10 bike
        22,   # 11 ground
    ],
    dtype=np.uint16,
)
