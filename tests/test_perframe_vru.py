"""Pin the two constructions that made the per-frame VRU scorer read 0.7 pp low.

Both bugs are silent: they produce plausible numbers that miss the published cells by
under a point, which is exactly why the scorer carries an aggregate gate. These tests
assert the LUT semantics without needing any dataset on disk.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "perframe_vru", pathlib.Path(__file__).resolve().parents[1] / "scripts/perframe_vru.py"
)
assert _SPEC and _SPEC.loader
perframe_vru = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perframe_vru)

DATACFG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "external/semantic_kitti_api/config/semantic-kitti.yaml"
)

pytestmark = pytest.mark.skipif(
    not DATACFG.exists(), reason="semantic-kitti-api config not vendored"
)


@pytest.fixture(scope="module")
def lut() -> np.ndarray:
    return perframe_vru.build_lut(str(DATACFG))


def test_moving_variants_fold_into_their_static_class(lut: np.ndarray) -> None:
    """253/254/255 are moving bicyclist/person/other; dropping them loses real VRU voxels."""
    assert lut[30] == 6 and lut[254] == 6, "person must include moving-person"
    assert lut[31] == 7 and lut[253] == 7, "bicyclist must include moving-bicyclist"
    assert lut[32] == 8 and lut[255] == 8, "motorcyclist must include moving-motorcyclist"


def test_empty_is_class_zero_but_outliers_are_ignored(lut: np.ndarray) -> None:
    """The official evaluator's ignore construction, which is easy to get backwards.

    Original label 0 is *empty*, a real class in completion. Every OTHER label that the
    learning map sends to 0 (outlier and friends) becomes 255 and is excluded. Folding
    those into class 0 inflates false positives on every class.
    """
    assert lut[0] == 0, "original 0 is 'empty', a scored class"
    assert lut[1] == 255, "'outlier' must be excluded, not merged into 'empty'"


def test_no_label_maps_to_zero_except_zero(lut: np.ndarray) -> None:
    zeros = np.flatnonzero(lut == 0)
    assert zeros.tolist() == [0], f"only original 0 may map to 0, got {zeros.tolist()}"


def test_iou_matches_hand_computation() -> None:
    """Gate the confusion-matrix reduction itself on a tiny known answer."""
    n = perframe_vru.NCLS
    conf = np.zeros(n * n, dtype=np.int64)
    # 8 voxels of GT class 6 -> 5 predicted 6, 3 predicted 0; plus 2 voxels of GT 0 -> 6.
    conf[6 * n + 6] = 5
    conf[6 * n + 0] = 3
    conf[0 * n + 6] = 2
    got = perframe_vru.iou(conf)[6]
    assert got == pytest.approx(5 / (5 + 2 + 3)), "IoU must count GT-empty predictions as FP"
