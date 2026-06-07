"""
SemanticKITTI dataset loader for data augmentation pipeline.
Enforces train/eval split and provides pyramid-compatible data format.

Supports two loading modes:
1. Pre-quantized mode (fast): Loads pre-processed .npy files from quantized directory
2. On-the-fly mode (slow): Loads full-resolution .label files and downsamples using mode pooling
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from gssc.data.semantic_kitti_utils import (
    convert_to_pyramid_format,
    get_sequence_frames,
    load_semantickitti_voxels,
    validate_training_sequence,
)

# Check for scipy availability
try:
    import scipy.ndimage  # noqa: F401  # availability probe
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SemanticKITTIDataset(Dataset):
    """
    SemanticKITTI dataset for training data augmentation models.
    Only allows access to training sequences (0-10).

    Supports two loading modes:
    1. Pre-quantized mode: Load pre-processed .npy files (fast, recommended)
    2. On-the-fly mode: Load full-resolution and downsample (slow, fallback)
    """

    # Map pyramid sizes to stage names for pre-quantized file lookup
    SIZE_TO_STAGE = {
        (32, 32, 4): 's1',
        (64, 64, 8): 's2',
    }

    def __init__(self,
                 dataset_root: str,
                 sequences: list[int] | None = None,
                 pyramid_size: tuple[int, int, int] = (256, 256, 16),
                 semantickitti_size: tuple[int, int, int] = (256, 256, 32),
                 quantized_root: str | None = None,
                 augment: bool = False):
        """
        Args:
            dataset_root: Path to SemanticKITTI dataset root (sequences/)
            sequences: List of sequence IDs to use (default: all training sequences)
            pyramid_size: Target size for pyramid diffusion model
            semantickitti_size: Original SemanticKITTI voxel size
            quantized_root: Path to pre-quantized data (optional, auto-detected if None)
            augment: Whether to apply data augmentation (flip + rotation)
        """
        self.dataset_root = dataset_root
        self.pyramid_size = pyramid_size
        self.semantickitti_size = semantickitti_size
        self.augment = augment

        # Determine stage name from pyramid size
        self.stage = self.SIZE_TO_STAGE.get(tuple(pyramid_size), None)

        # Auto-detect quantized root if not provided
        if quantized_root is None:
            # Try common locations relative to dataset_root
            parent_dir = os.path.dirname(dataset_root.rstrip('/'))
            possible_paths = [
                os.path.join(parent_dir, 'SemanticKITTI_quantized'),
                os.path.join(parent_dir, 'quantized'),
                'datasets/SemanticKITTI_quantized',
            ]
            for path in possible_paths:
                if os.path.exists(path) and self.stage:
                    stage_path = os.path.join(path, self.stage, 'sequences')
                    if os.path.exists(stage_path):
                        quantized_root = path
                        break

        self.quantized_root = quantized_root
        self.use_quantized = False

        # Check if pre-quantized data is available
        if self.quantized_root and self.stage:
            quantized_stage_path = os.path.join(self.quantized_root, self.stage, 'sequences')
            if os.path.exists(quantized_stage_path):
                self.use_quantized = True
                self.quantized_stage_path = quantized_stage_path
                print(f"[SemanticKITTIDataset] Using pre-quantized data from: {quantized_stage_path}")
            else:
                print(f"[SemanticKITTIDataset] Pre-quantized path not found: {quantized_stage_path}")
                print("[SemanticKITTIDataset] Falling back to on-the-fly mode pooling")
        else:
            print("[SemanticKITTIDataset] Using on-the-fly mode pooling (no pre-quantized data)")

        # Default to all training sequences (EXCLUDING sequence 08 - used as validation)
        if sequences is None:
            sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]  # 0-10 EXCEPT 08

        # Validate sequences are in training set
        for seq_id in sequences:
            if not validate_training_sequence(seq_id):
                raise ValueError(f"Sequence {seq_id} is not in training set (0-10)")

        # Build frame list
        self.frame_list = []
        for seq_id in sequences:
            if self.use_quantized:
                # Use quantized directory structure
                seq_path = os.path.join(self.quantized_stage_path, f"{seq_id:02d}")
                if os.path.exists(seq_path):
                    frames = sorted([f.replace('.npy', '') for f in os.listdir(seq_path) if f.endswith('.npy')])
                    for frame_id in frames:
                        self.frame_list.append((seq_id, frame_id))
            else:
                # Use original directory structure
                seq_path = os.path.join(dataset_root, f"{seq_id:02d}")
                if not os.path.exists(seq_path):
                    continue
                frames = get_sequence_frames(seq_path)
                for frame_id in frames:
                    self.frame_list.append((seq_id, frame_id))

        if not self.frame_list:
            raise ValueError("No valid frames found in specified sequences")

    def __len__(self) -> int:
        return len(self.frame_list)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - 'input': Sparse voxel input (pyramid_size)
                - 'labels': Ground truth labels (pyramid_size)
                - 'invalid': Invalid voxel mask (pyramid_size)
                - 'sequence': Sequence ID
                - 'frame': Frame ID
        """
        seq_id, frame_id = self.frame_list[idx]

        if self.use_quantized:
            # Fast path: load pre-quantized .npy file
            quantized_path = os.path.join(
                self.quantized_stage_path, f"{seq_id:02d}", f"{frame_id}.npy"
            )
            labels = np.load(quantized_path)

            # For pre-quantized data, we only have labels
            # Input and invalid masks are not needed for unconditional generation training
            pyramid_data = {
                'input': torch.zeros(self.pyramid_size, dtype=torch.float32),
                'labels': torch.from_numpy(labels).long(),
                'invalid': torch.zeros(self.pyramid_size, dtype=torch.bool),
                'sequence': seq_id,
                'frame': frame_id
            }
        else:
            # Slow path: load full-resolution and downsample with mode pooling
            voxel_path = os.path.join(self.dataset_root, f"{seq_id:02d}", "voxels")

            # Load original data
            voxel_data = load_semantickitti_voxels(
                voxel_path, frame_id, self.semantickitti_size
            )

            # Convert to pyramid format using mode pooling
            pyramid_data = {
                'input': convert_to_pyramid_format(voxel_data['input'], self.pyramid_size),
                'labels': convert_to_pyramid_format(voxel_data['labels'], self.pyramid_size),
                'invalid': convert_to_pyramid_format(voxel_data['invalid'], self.pyramid_size),
                'sequence': seq_id,
                'frame': frame_id
            }

            # Convert to tensors
            pyramid_data['input'] = torch.from_numpy(pyramid_data['input']).float()
            pyramid_data['labels'] = torch.from_numpy(pyramid_data['labels']).long()
            pyramid_data['invalid'] = torch.from_numpy(pyramid_data['invalid']).bool()

        # Apply data augmentation if enabled
        if self.augment:
            pyramid_data = self._apply_augmentation(pyramid_data)

        return pyramid_data

    def _apply_augmentation(self, data: dict) -> dict:
        """
        Apply data augmentation: random flip (X, Y) and rotation (0°, 90°, 180°, 270°).
        Matches the reference pyramid-discrete-diffusion implementation.
        """
        labels = data['labels'].numpy() if isinstance(data['labels'], torch.Tensor) else data['labels']

        # Random flip along X axis (50% chance)
        if np.random.randint(2):
            labels = np.flip(labels, axis=0).copy()

        # Random flip along Y axis (50% chance)
        if np.random.randint(2):
            labels = np.flip(labels, axis=1).copy()

        # Random rotation in XY plane (0°, 90°, 180°, 270°)
        k = np.random.randint(4)
        if k > 0:
            labels = np.rot90(labels, k, axes=(0, 1)).copy()

        data['labels'] = torch.from_numpy(labels).long()
        return data

    def get_frame_info(self, idx: int) -> tuple[int, str]:
        """Get sequence and frame ID for given index."""
        return self.frame_list[idx]


class SemanticKITTIAugmentationDataset(Dataset):
    """Dataset that would combine original and synthetic SemanticKITTI data.

    This is a stub: the on-disk loading of the synthetic (sparse, complete)
    pool is not wired up here. The headline runs consume the synthetic pool
    through the dump/retrain scripts (see ``docs/DATASET.md`` "Synthetic pool"),
    not through this class, so this dataset only proxies the original data and
    explicitly refuses any request that would require synthetic samples.
    """

    def __init__(self,
                 original_dataset: SemanticKITTIDataset,
                 augmented_data_path: str | None = None,
                 augmentation_ratio: float = 1.0):
        """
        Args:
            original_dataset: Original SemanticKITTI dataset
            augmented_data_path: Path to augmented data. Synthetic-pool loading
                is not implemented here; passing a non-None value raises
                NotImplementedError rather than silently ignoring the path.
            augmentation_ratio: Ratio of augmented to original data
        """
        if augmented_data_path is not None:
            raise NotImplementedError(
                "Synthetic-pool loading is not wired into "
                "SemanticKITTIAugmentationDataset. Provision the synthetic "
                "(sparse, complete) pool via the dump/retrain scripts described "
                "in docs/DATASET.md ('Synthetic pool'); this class only proxies "
                "the original dataset."
            )
        self.original_dataset = original_dataset
        self.augmented_data_path = augmented_data_path
        self.augmentation_ratio = augmentation_ratio

        # Synthetic samples are not loaded here, so the length matches the
        # original dataset exactly (see class docstring / docs/DATASET.md).
        self.total_samples = len(original_dataset)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        # Only original indices are valid: total_samples == len(original_dataset)
        # because synthetic loading is not implemented (see class docstring).
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        raise NotImplementedError(
            "Synthetic-augmented sampling is not implemented in "
            "SemanticKITTIAugmentationDataset; index "
            f"{idx} exceeds the original dataset. See docs/DATASET.md "
            "('Synthetic pool') for the supported augmentation path."
        )