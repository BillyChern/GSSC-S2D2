"""Tests for the helpers in :mod:`gssc.inference.generate_predictions`.

These cover the binary-occupancy decoder and the .bin voxel loader so that
a regression in either silently corrupts the entire 4071-frame validation
sweep — caught here in a millisecond instead of after a 5-min generation.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# torch alone is not a sufficient guard here. The module under test reaches
# gssc.models.s2d2_unet -> gssc.models.sparse_lidar_encoder:17, which does
# `import spconv.pytorch` at module scope, and spconv ships only CUDA-specific
# wheels -- so "torch installed, spconv not" is the ordinary contributor setup.
# Without this line that combination does not skip: the file dies AT COLLECTION
# and aborts the entire run (measured on a clean py3.10 venv with CPU torch 2.4.0
# and no spconv: `3 errors during collection`, exit 2, zero tests executed).
pytest.importorskip("spconv")

from gssc.inference.generate_predictions import (
    LEARNING_MAP_INV,
    SPLIT_SEQUENCES,
    load_lidar_voxels,
    unpack,
)


def test_unpack_msb_first() -> None:
    """SemanticKITTI .bin files pack 8 bits per byte, MSB first."""
    compressed = np.array([0b10100110], dtype=np.uint8)
    expected = [1, 0, 1, 0, 0, 1, 1, 0]
    assert list(unpack(compressed)) == expected


def test_unpack_dtype_and_shape() -> None:
    """Output dtype is uint8 and length is 8x the input."""
    compressed = np.zeros(100, dtype=np.uint8)
    out = unpack(compressed)
    assert out.dtype == np.uint8
    assert out.shape == (800,)


def test_load_lidar_voxels_shape_and_dtype(tmp_path) -> None:
    """A canonical 256x256x32 grid round-trips through the bin loader."""
    # Pack a known density into the bit-encoded bin file.
    n_voxels = 256 * 256 * 32
    binary = np.zeros(n_voxels, dtype=np.uint8)
    binary[::100] = 1  # ~1% occupancy, like real LiDAR
    # Pack 8 bits per byte (MSB first).
    packed = np.packbits(binary)
    bin_path = tmp_path / "000000.bin"
    bin_path.write_bytes(packed.tobytes())

    voxels = load_lidar_voxels(str(bin_path))
    assert voxels.shape == (1, 1, 256, 256, 32)
    assert voxels.dtype == torch.float32
    # Round-trip occupancy count.
    assert voxels.sum().item() == binary.sum()


def test_learning_map_inv_size_and_values() -> None:
    """The training-space → original-label LUT must have one row per class."""
    assert LEARNING_MAP_INV.shape == (20,)
    assert LEARNING_MAP_INV.dtype == np.uint16
    # Spot-check a few canonical mappings (cf. semantic-kitti.yaml).
    assert LEARNING_MAP_INV[0] == 0     # unlabeled
    assert LEARNING_MAP_INV[1] == 10    # car
    assert LEARNING_MAP_INV[8] == 32    # motorcyclist
    assert LEARNING_MAP_INV[18] == 80   # pole
    assert LEARNING_MAP_INV[19] == 81   # traffic-sign


def test_split_sequences_val_is_seq08() -> None:
    """The val split is exactly seq 08 (paper convention)."""
    assert SPLIT_SEQUENCES["valid"] == ["08"]


def test_split_sequences_test_is_11_through_21() -> None:
    """The test split is the 11 hidden sequences for the leaderboard."""
    test = SPLIT_SEQUENCES["test"]
    assert test == [f"{i:02d}" for i in range(11, 22)]
    assert len(test) == 11
