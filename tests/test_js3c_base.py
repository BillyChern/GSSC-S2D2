"""Tests for JS3C-Net cross-base support (paper Tab. III rows 90-91).

Coverage:
1. JS3C predictions reader: shape, dtype, value-range validation.
2. D4 TTA symmetry on JS3C predictions (base-agnostic flip+rot pipeline).
3. Cold-diffusion forward determinism (required for cross-base eval).
4. Backwards-compat ``scpnet_pred_dir`` alias emits ``DeprecationWarning``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gssc.inference.d4_tta import D4_ELEMENTS, apply_d4, derive_bev, invert_d4
from gssc.models.js3c_base import (
    EXPECTED_SHAPE,
    NUM_CLASSES,
    load_js3c_predictions,
)
from gssc.utils.compat import resolve_base_pred_dir


def _make_synth_pred(seed: int = 0) -> np.ndarray:
    """A synthetic [256, 256, 32] int64 prediction in the legal label range."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, NUM_CLASSES, size=EXPECTED_SHAPE, dtype=np.int64)
    return arr


def test_js3c_prediction_loading_shape_dtype(tmp_path: Path) -> None:
    seq = "08"
    frame_id = "000000"
    (tmp_path / seq).mkdir(parents=True)
    expected = _make_synth_pred()
    np.save(tmp_path / seq / f"{frame_id}_pred.npy", expected)

    got = load_js3c_predictions(tmp_path, seq, frame_id)
    assert got.shape == EXPECTED_SHAPE
    assert got.dtype == np.int64
    assert got.min() >= 0
    assert got.max() < NUM_CLASSES
    np.testing.assert_array_equal(got, expected)


def test_js3c_prediction_loading_validates(tmp_path: Path) -> None:
    seq = "08"
    (tmp_path / seq).mkdir(parents=True)

    wrong_shape = np.zeros((128, 128, 32), dtype=np.int64)
    np.save(tmp_path / seq / "000001_pred.npy", wrong_shape)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_js3c_predictions(tmp_path, seq, "000001")

    out_of_range = _make_synth_pred()
    out_of_range[0, 0, 0] = 25
    np.save(tmp_path / seq / "000002_pred.npy", out_of_range)
    with pytest.raises(ValueError, match="value range"):
        load_js3c_predictions(tmp_path, seq, "000002")

    with pytest.raises(FileNotFoundError, match="js3c prediction missing"):
        load_js3c_predictions(tmp_path, seq, "999999")


def test_d4_tta_symmetry_on_js3c_predictions() -> None:
    """``invert_d4 . apply_d4`` is the identity on every D4 element.

    Critical for the v1.1.0 cross-base reproduction: the same D4 pipeline
    must round-trip JS3C predictions losslessly, otherwise the +3.99 pp val
    delta is contaminated by TTA-noise rather than the correction signal.
    """
    lidar = torch.rand(1, 1, 256, 256, 32)
    base = torch.from_numpy(_make_synth_pred()).unsqueeze(0)
    bev_in = torch.zeros(1, 256, 256, dtype=torch.long)

    for fx, fy, rk in D4_ELEMENTS:
        lidar_t, base_t, _ = apply_d4(lidar, base, bev_in, fx, fy, rk)
        bev = derive_bev(base_t)
        soft = torch.nn.functional.one_hot(base_t.long(), NUM_CLASSES).float()
        soft = soft.permute(0, 4, 1, 2, 3)
        soft_back = invert_d4(soft, fx, fy, rk)
        back = soft_back.argmax(dim=1)[0]
        torch.testing.assert_close(back, base[0])
        assert bev.shape == (1, 256, 256)
        assert int(bev.min()) >= 0
        assert int(bev.max()) < NUM_CLASSES


def test_scpnet_pred_dir_alias_emits_deprecation_warning() -> None:
    """``S3DSKDDataset(scpnet_pred_dir=...)`` still works but warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolved = resolve_base_pred_dir(scpnet_pred_dir="data/scpnet_predictions")
    assert resolved == "data/scpnet_predictions"
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "scpnet_pred_dir alias must emit DeprecationWarning"
    assert any("scpnet_pred_dir" in str(w.message) for w in deprecations)


def test_resolve_base_pred_dir_prefers_new_kwarg() -> None:
    """When both names are supplied, the canonical ``base_pred_dir`` wins."""
    resolved = resolve_base_pred_dir(
        base_pred_dir="data/js3cnet_predictions",
        scpnet_pred_dir="data/scpnet_predictions",
    )
    assert resolved == "data/js3cnet_predictions"


def test_cold_diffusion_deterministic_source() -> None:
    """Cold-diffusion (paper supp §H) replaces the uniform-noise D3PM target.

    The forward *distribution* :math:`q(x_t | x_0)` produced by
    :meth:`MultinomialDiffusion3DV2.q_probs` must be deterministic in
    ``(x_0, t, x_scpnet)``: ``q_sample`` then draws a categorical sample
    from those probabilities, but the probabilities themselves are
    deterministic. This is the contract the JS3C+S²D² eval path relies on
    (paper supp § H sets ``cold_diffusion=true``); without determinism the
    cross-base eval becomes seed-dependent.
    """
    multinomial = pytest.importorskip("gssc.diffusion.multinomial")
    diffusion = multinomial.MultinomialDiffusion3DV2(
        num_classes=NUM_CLASSES,
        num_timesteps=100,
        beta_max=0.1,
    )
    x_0 = torch.from_numpy(_make_synth_pred(seed=1)).unsqueeze(0)
    t = torch.zeros(1, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(x_0.long(), NUM_CLASSES).float()
    one_hot = one_hot.permute(0, 4, 1, 2, 3).contiguous()

    if not hasattr(diffusion, "q_probs"):
        pytest.skip("q_probs not exposed on MultinomialDiffusion3DV2")

    # Cold-diffusion: pass x_scpnet=x_0 (source equals target -> 100% peaked).
    probs1 = diffusion.q_probs(one_hot, t, x_scpnet=one_hot)
    probs2 = diffusion.q_probs(one_hot, t, x_scpnet=one_hot)
    torch.testing.assert_close(probs1, probs2)
    # Standard (uniform-noise) forward is also deterministic in (x_0, t).
    std1 = diffusion.q_probs(one_hot, t)
    std2 = diffusion.q_probs(one_hot, t)
    torch.testing.assert_close(std1, std2)
