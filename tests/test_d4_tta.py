"""Tests for the D4 (dihedral) TTA helpers in :mod:`gssc.inference.d4_tta`.

These pin the symmetry contract: ``invert_d4(apply_d4(x))`` must be the
identity on every group element. A regression here (axis swap, wrong
``rot90`` sign) silently drops 0.2 -- 0.5 mIoU at inference time, so the
test is cheap insurance.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from gssc.inference.d4_tta import (
    D4_ELEMENTS,
    apply_d4,
    derive_bev,
    invert_d4,
    unpack_voxels,
)

# ---------------------------------------------------------------------------
# unpack_voxels: kitti's bit-packed binary occupancy decoder.
# ---------------------------------------------------------------------------

def test_unpack_voxels_zeros_stay_zero() -> None:
    """All-empty input -> all-empty output, no spurious 1s."""
    compressed = np.zeros(32, dtype=np.uint8)
    out = unpack_voxels(compressed)
    assert out.shape == (32 * 8,)
    assert out.sum() == 0


def test_unpack_voxels_msb_first() -> None:
    """Byte 0b10000000 unpacks to [1,0,0,0,0,0,0,0] (MSB first)."""
    compressed = np.array([0b10000000], dtype=np.uint8)
    assert list(unpack_voxels(compressed)) == [1, 0, 0, 0, 0, 0, 0, 0]


def test_unpack_voxels_lsb_set() -> None:
    """Byte 0b00000001 unpacks to [0,0,0,0,0,0,0,1]."""
    compressed = np.array([0b00000001], dtype=np.uint8)
    assert list(unpack_voxels(compressed)) == [0, 0, 0, 0, 0, 0, 0, 1]


def test_unpack_voxels_dtype_is_uint8() -> None:
    """Output dtype is uint8 — float casts happen downstream."""
    compressed = np.array([0xFF], dtype=np.uint8)
    out = unpack_voxels(compressed)
    assert out.dtype == np.uint8
    assert out.sum() == 8  # all bits set


# ---------------------------------------------------------------------------
# apply_d4 / invert_d4 round-trip.
# ---------------------------------------------------------------------------

def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic asymmetric inputs so any axis swap is detectable."""
    torch.manual_seed(0)
    # Lidar: [B=1, C=1, H=8, W=4, D=2] — distinct values per cell.
    lidar = torch.arange(1 * 1 * 8 * 4 * 2, dtype=torch.float32).reshape(1, 1, 8, 4, 2)
    # SCPNet pred: [B=1, H=8, W=4, D=2] long tensor.
    scp = torch.arange(8 * 4 * 2, dtype=torch.long).reshape(1, 8, 4, 2)
    # BEV: [B=1, H=8, W=4] long.
    bev = torch.arange(8 * 4, dtype=torch.long).reshape(1, 8, 4)
    return lidar, scp, bev


def test_apply_d4_identity_is_noop() -> None:
    """The (False, False, 0) element should not modify any input."""
    lidar, scp, bev = _make_inputs()
    lidar2, scp2, bev2 = apply_d4(lidar, scp, bev, False, False, 0)
    assert torch.equal(lidar, lidar2)
    assert torch.equal(scp, scp2)
    assert torch.equal(bev, bev2)


def test_apply_d4_flip_x_negates_axis() -> None:
    """flip_x mirrors along H (lidar axis 2 / scp+bev axis 1)."""
    lidar, scp, bev = _make_inputs()
    lidar2, scp2, bev2 = apply_d4(lidar, scp, bev, True, False, 0)
    assert torch.equal(lidar2, torch.flip(lidar, dims=[2]))
    assert torch.equal(scp2, torch.flip(scp, dims=[1]))
    assert torch.equal(bev2, torch.flip(bev, dims=[1]))


def test_apply_d4_flip_y_negates_axis() -> None:
    """flip_y mirrors along W (lidar axis 3 / scp+bev axis 2)."""
    lidar, scp, bev = _make_inputs()
    lidar2, scp2, bev2 = apply_d4(lidar, scp, bev, False, True, 0)
    assert torch.equal(lidar2, torch.flip(lidar, dims=[3]))
    assert torch.equal(scp2, torch.flip(scp, dims=[2]))
    assert torch.equal(bev2, torch.flip(bev, dims=[2]))


def test_apply_d4_rot90_k1() -> None:
    """rot_k=1 rotates 90° on the H,W plane."""
    lidar, scp, bev = _make_inputs()
    lidar2, scp2, bev2 = apply_d4(lidar, scp, bev, False, False, 1)
    assert torch.equal(lidar2, torch.rot90(lidar, k=1, dims=[2, 3]))
    assert torch.equal(scp2, torch.rot90(scp, k=1, dims=[1, 2]))
    assert torch.equal(bev2, torch.rot90(bev, k=1, dims=[1, 2]))


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_apply_d4_invert_d4_round_trip(element: tuple[bool, bool, int]) -> None:
    """For every D4 element: invert ∘ apply on the softmax should be identity.

    apply_d4 transforms (lidar, scp, bev); invert_d4 transforms a softmax
    [1, K, H, W, D]. We construct a synthetic softmax that has the same
    H, W footprint as scp/bev, run forward then inverse, and check exact
    equality.
    """
    flip_x, flip_y, rot_k = element
    torch.manual_seed(7)
    soft = torch.randn(1, 20, 8, 4, 2)

    # Apply (forward) to softmax: same convention as inference path - flip
    # axes 2 (H) and 3 (W), rot90 on (2,3).
    forward = soft.clone()
    if flip_x:
        forward = torch.flip(forward, dims=[2])
    if flip_y:
        forward = torch.flip(forward, dims=[3])
    if rot_k > 0:
        forward = torch.rot90(forward, k=rot_k, dims=[2, 3])

    back = invert_d4(forward, flip_x, flip_y, rot_k)
    assert torch.allclose(back, soft, atol=0)


def test_d4_elements_count() -> None:
    """The TTA group has exactly 8 elements (D4 = ⟨flip, rot⟩)."""
    assert len(D4_ELEMENTS) == 8
    # Each element is a unique (bool, bool, int) triple.
    assert len(set(D4_ELEMENTS)) == 8


# ---------------------------------------------------------------------------
# derive_bev: topmost-non-empty-class projection.
# ---------------------------------------------------------------------------

def test_derive_bev_topmost_non_empty() -> None:
    """For each (x, y) column the highest-z non-empty class wins.

    ``derive_bev`` hardcodes the canonical SemanticKITTI grid (256x256x32),
    so we build inputs at that resolution and probe a few columns.
    """
    scp = torch.zeros(1, 256, 256, 32, dtype=torch.long)
    scp[0, 0, 0, 0] = 5
    scp[0, 0, 0, 31] = 7  # topmost wins
    scp[0, 1, 1, 15] = 3  # only one non-empty in this column
    bev = derive_bev(scp)
    assert bev.shape == (1, 256, 256)
    assert bev[0, 0, 0].item() == 7
    assert bev[0, 1, 1].item() == 3
    assert bev[0, 2, 2].item() == 0  # all empty -> 0


def test_derive_bev_all_empty_returns_zeros() -> None:
    """An all-empty scene yields an all-empty BEV (no spurious classes)."""
    scp = torch.zeros(1, 256, 256, 32, dtype=torch.long)
    bev = derive_bev(scp)
    assert bev.sum().item() == 0
