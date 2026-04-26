"""
SemanticKITTI data format utilities for data augmentation pipeline.
Handles binary file I/O and voxel format conversions.
"""
import os

import numpy as np

# Official SemanticKITTI learning_map from config/semantic-kitti.yaml
SEMANTICKITTI_LEARNING_MAP = {
    0: 0,     # "unlabeled"
    1: 0,     # "outlier" mapped to "unlabeled"
    10: 1,    # "car"
    11: 2,    # "bicycle"
    13: 5,    # "bus" mapped to "other-vehicle"
    15: 3,    # "motorcycle"
    16: 5,    # "on-rails" mapped to "other-vehicle"
    18: 4,    # "truck"
    20: 5,    # "other-vehicle"
    30: 6,    # "person"
    31: 7,    # "bicyclist"
    32: 8,    # "motorcyclist"
    40: 9,    # "road"
    44: 10,   # "parking"
    48: 11,   # "sidewalk"
    49: 12,   # "other-ground"
    50: 13,   # "building"
    51: 14,   # "fence"
    52: 0,    # "other-structure" mapped to "unlabeled"
    60: 9,    # "lane-marking" to "road"
    70: 15,   # "vegetation"
    71: 16,   # "trunk"
    72: 17,   # "terrain"
    80: 18,   # "pole"
    81: 19,   # "traffic-sign"
    99: 0,    # "other-object" to "unlabeled"
    252: 1,   # "moving-car" to "car"
    253: 7,   # "moving-bicyclist" to "bicyclist"
    254: 6,   # "moving-person" to "person"
    255: 8,   # "moving-motorcyclist" to "motorcyclist"
    256: 5,   # "moving-on-rails" mapped to "other-vehicle"
    257: 5,   # "moving-bus" mapped to "other-vehicle"
    258: 4,   # "moving-truck" to "truck"
    259: 5,   # "moving-other-vehicle" to "other-vehicle"
}

# Create vectorized mapping function
LEARNING_MAP_ARRAY = np.zeros(260, dtype=np.uint8)  # Max label is 259
for original, mapped in SEMANTICKITTI_LEARNING_MAP.items():
    LEARNING_MAP_ARRAY[original] = mapped


def unpack_binary_mask(compressed: np.ndarray) -> np.ndarray:
    """Convert bit-packed binary mask to full boolean array."""
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed


def pack_binary_mask(uncompressed: np.ndarray) -> np.ndarray:
    """Pack boolean array to bit-compressed format."""
    assert uncompressed.shape[0] % 8 == 0, "Array size must be divisible by 8"
    compressed_size = uncompressed.shape[0] // 8
    compressed = np.zeros(compressed_size, dtype=np.uint8)

    for i in range(8):
        compressed += (uncompressed[i::8] & 1) << (7 - i)

    return compressed


def load_semantickitti_voxels(base_path: str, frame_id: str,
                            voxel_size: tuple[int, int, int] = (256, 256, 32)) -> dict:
    """
    Load SemanticKITTI voxel data for a single frame.

    Args:
        base_path: Path to voxels directory (e.g., sequences/00/voxels)
        frame_id: Frame identifier (e.g., '000000')
        voxel_size: Expected voxel grid dimensions

    Returns:
        dict with keys: 'input', 'labels', 'invalid', 'occluded'
    """
    frame_path = os.path.join(base_path, frame_id)
    H, W, D = voxel_size
    total_voxels = H * W * D

    # Load input (sparse voxels)
    input_data = unpack_binary_mask(
        np.fromfile(f"{frame_path}.bin", dtype=np.uint8)
    ).reshape(voxel_size)

    # Load ground truth labels
    labels = np.fromfile(f"{frame_path}.label", dtype=np.uint16).reshape(voxel_size)

    # Apply SemanticKITTI learning_map (convert to 0-19 classes)
    # Handle labels > 259 by mapping to unlabeled (0)
    labels = np.where(labels < 260, LEARNING_MAP_ARRAY[labels], 0).astype(np.uint8)

    # Load invalid mask
    invalid = unpack_binary_mask(
        np.fromfile(f"{frame_path}.invalid", dtype=np.uint8)
    ).reshape(voxel_size)

    # Load occluded mask
    occluded = unpack_binary_mask(
        np.fromfile(f"{frame_path}.occluded", dtype=np.uint8)
    ).reshape(voxel_size)

    return {
        'input': input_data,
        'labels': labels,
        'invalid': invalid,
        'occluded': occluded
    }


def save_semantickitti_voxels(data: dict, base_path: str, frame_id: str):
    """Save voxel data in SemanticKITTI format."""
    frame_path = os.path.join(base_path, frame_id)

    # Save input (sparse voxels)
    input_packed = pack_binary_mask(data['input'].flatten())
    input_packed.tofile(f"{frame_path}.bin")

    # Save labels
    data['labels'].astype(np.uint16).tofile(f"{frame_path}.label")

    # Save invalid mask
    invalid_packed = pack_binary_mask(data['invalid'].flatten())
    invalid_packed.tofile(f"{frame_path}.invalid")

    # Save occluded mask
    occluded_packed = pack_binary_mask(data['occluded'].flatten())
    occluded_packed.tofile(f"{frame_path}.occluded")


def validate_training_sequence(sequence_id: int) -> bool:
    """Ensure sequence is in training set (0-10 only)."""
    return 0 <= sequence_id <= 10


def get_sequence_frames(sequence_path: str) -> list:
    """Get all frame IDs in a sequence directory."""
    voxel_path = os.path.join(sequence_path, 'voxels')
    if not os.path.exists(voxel_path):
        return []

    frames = []
    for file in os.listdir(voxel_path):
        if file.endswith('.bin'):
            frames.append(file[:-4])  # Remove .bin extension

    return sorted(frames)


def _get_max_occurrence_label(block: np.ndarray) -> int:
    """
    Get the most frequent NON-ZERO class in a block.

    EXACT logic from pyramid-discrete-diffusion/Tools/data_process/carla_process.py
    This is MODE POOLING, not max pooling - it finds the most frequent non-zero class.

    Args:
        block: Voxel block to analyze

    Returns:
        Most frequent non-zero class, or 0 if all voxels are empty
    """
    flat = block.flatten().astype(np.int32)
    if flat.max() == 0:
        return 0

    counts = np.bincount(flat, minlength=20)
    nonzero_counts = counts[1:]  # Exclude class 0 (empty/unlabeled)

    if nonzero_counts.sum() > 0:
        max_label = np.argmax(nonzero_counts) + 1
        return max_label
    else:
        return 0


def convert_to_pyramid_format(voxel_data: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    """
    Convert SemanticKITTI voxel data to pyramid diffusion format using MODE POOLING.

    Mode pooling finds the most frequent NON-ZERO class in each block, preserving
    occupancy much better than nearest neighbor interpolation.

    This matches the original pyramid-discrete-diffusion approach:
    - Tools/data_process/carla_process.py: resample() with get_max_occurrence_label()

    Args:
        voxel_data: Input voxel grid (H, W, D) with class labels 0-19
        target_size: Target size tuple (H', W', D')

    Returns:
        Downsampled voxel grid using mode pooling
    """
    current_size = voxel_data.shape

    # If same size, return copy
    if current_size == target_size:
        return voxel_data.copy()

    # Calculate block size for each dimension
    scale_x = current_size[0] // target_size[0]
    scale_y = current_size[1] // target_size[1]
    scale_z = current_size[2] // target_size[2]

    result = np.zeros(target_size, dtype=voxel_data.dtype)

    # Mode pooling: for each output voxel, find most frequent non-zero class in the corresponding block
    for i in range(target_size[0]):
        for j in range(target_size[1]):
            for k in range(target_size[2]):
                block = voxel_data[i*scale_x:(i+1)*scale_x,
                                   j*scale_y:(j+1)*scale_y,
                                   k*scale_z:(k+1)*scale_z]
                result[i, j, k] = _get_max_occurrence_label(block)

    return result


def convert_from_pyramid_format(pyramid_data: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    """Convert pyramid diffusion output back to SemanticKITTI format."""
    from scipy import ndimage

    current_size = pyramid_data.shape
    zoom_factors = [t/c for t, c in zip(target_size, current_size)]

    return ndimage.zoom(pyramid_data, zoom_factors, order=0)  # Nearest neighbor