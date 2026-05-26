"""Tests for LMSCNet cross-base support (paper Tab. III row 90).

Mirrors :mod:`tests.test_js3c_base`. Coverage:
1. LMSCNet predictions reader: shape, dtype, value-range validation.
2. Shape-mismatch, value-range, and missing-file error paths.
3. ``--base_kind lmscnet`` is a recognised value in the dataset constructor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gssc.models.lmscnet_base import (
    EXPECTED_SHAPE,
    NUM_CLASSES,
    load_lmscnet_predictions,
)


def _make_synth_pred(seed: int = 0) -> np.ndarray:
    """A synthetic [256, 256, 32] int64 prediction in the legal label range."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, NUM_CLASSES, size=EXPECTED_SHAPE, dtype=np.int64)


def test_lmscnet_prediction_loading_shape_dtype(tmp_path: Path) -> None:
    seq = "08"
    frame_id = "000000"
    (tmp_path / seq).mkdir(parents=True)
    expected = _make_synth_pred()
    np.save(tmp_path / seq / f"{frame_id}_pred.npy", expected)

    got = load_lmscnet_predictions(tmp_path, seq, frame_id)
    assert got.shape == EXPECTED_SHAPE
    assert got.dtype == np.int64
    assert int(got.min()) >= 0
    assert int(got.max()) < NUM_CLASSES
    np.testing.assert_array_equal(got, expected)


def test_lmscnet_prediction_loading_validates(tmp_path: Path) -> None:
    seq = "08"
    (tmp_path / seq).mkdir(parents=True)

    wrong_shape = np.zeros((128, 128, 32), dtype=np.int64)
    np.save(tmp_path / seq / "000001_pred.npy", wrong_shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_lmscnet_predictions(tmp_path, seq, "000001")

    out_of_range = _make_synth_pred()
    out_of_range[0, 0, 0] = 25
    np.save(tmp_path / seq / "000002_pred.npy", out_of_range)
    with pytest.raises(ValueError, match="value range"):
        load_lmscnet_predictions(tmp_path, seq, "000002")

    with pytest.raises(FileNotFoundError, match="lmscnet prediction missing"):
        load_lmscnet_predictions(tmp_path, seq, "999999")


def test_lmscnet_uint8_cast_to_int64(tmp_path: Path) -> None:
    """Predictions dumped as ``uint8`` (one byte per voxel) must be accepted.

    Dump pipeline space-optimises predictions to ``uint8`` since the 20-class
    label space fits in one byte; the reader is contractually required to
    upcast losslessly to ``int64`` for downstream pipeline kernels.
    """
    seq = "08"
    (tmp_path / seq).mkdir(parents=True)
    arr_u8 = _make_synth_pred().astype(np.uint8, copy=False)
    np.save(tmp_path / seq / "000003_pred.npy", arr_u8)

    got = load_lmscnet_predictions(tmp_path, seq, "000003")
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, arr_u8.astype(np.int64))


def test_base_kind_lmscnet_is_accepted_literal() -> None:
    """``S3DSKDDataset(base_kind="lmscnet")`` is in the Literal type.

    The dataset stores ``base_kind`` as informational metadata; the actual
    loading uses ``base_pred_dir`` directly. This test guards against future
    regressions that might drop ``lmscnet`` from the accepted Literal.
    """
    from typing import get_args, get_type_hints

    semantickitti = pytest.importorskip("gssc.data.semantickitti")
    init = semantickitti.S3DSKDDataset.__init__
    hints = get_type_hints(init)
    base_kind_type = hints.get("base_kind")
    if base_kind_type is None:
        pytest.skip("base_kind type hint not exposed on S3DSKDDataset.__init__")
    allowed = get_args(base_kind_type)
    assert "lmscnet" in allowed, (
        f'base_kind Literal must include "lmscnet"; got {allowed}'
    )
    assert "scpnet" in allowed and "js3c" in allowed, (
        f'base_kind Literal must also keep prior values; got {allowed}'
    )
