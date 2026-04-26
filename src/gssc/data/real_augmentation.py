"""
Real Data Augmentation Pipeline (DA0)

Augments real SemanticKITTI scenes WITHOUT relying on S3 generation:
1. Geometric transforms (flip, rotation)
2. Multi-viewpoint LiDAR re-simulation
3. Rare object copy-paste from Object Bank

This creates diverse (sparse, complete) pairs from existing real scenes.

Usage:
    from gssc.data.real_augmentation import RealDataAugmentor

    augmentor = RealDataAugmentor(object_bank_path="datasets/object_bank.pkl")

    # Augment single scene
    augmented_pairs = augmentor.augment_scene(complete_scene)

    # Batch processing
    augmentor.process_dataset(
        ssc_root="datasets/dataset_SemanticKITTI_SSC",
        output_dir="datasets/augmented_pairs",
        sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    )

Author: SSC Project
Date: December 2024
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from gssc.data.lidar_resampler_v2 import (
    create_multi_return_resampler,
    pack_voxels,
)
from gssc.data.object_bank import RARE_CLASS_IDS, RARE_CLASSES, ObjectBank

logger = logging.getLogger(__name__)


# SemanticKITTI learning map
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}

# Inverse learning map (for saving)
LEARNING_MAP_INV = {
    0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
    10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81,
}

# Ground classes for placement validation
GROUND_CLASSES = {9, 10, 11, 12, 17}  # road, parking, sidewalk, other-ground, terrain


@dataclass
class AugmentationConfig:
    """
    Configuration for real data augmentation.

    IMPORTANT: Training already applies flip/rotation, so we should:
    1. DISABLE geometric transforms here (avoid redundancy)
    2. FOCUS on copy-paste (adds NEW content, not just transforms)
    3. FOCUS on multi-viewpoint LiDAR (creates NEW observation patterns)

    The unique value of real_augmentation vs training augmentation:
    - Training: transforms existing data (same content, different orientation)
    - Real Aug: ADDS new semantic content + creates novel LiDAR observations
    """

    # Geometric transforms (DISABLED - training already does this)
    enable_flip_x: bool = False  # Training handles this
    enable_flip_y: bool = False  # Training handles this
    enable_rotation: bool = False  # Training handles this
    rotation_angles: list[int] = None

    # Multi-viewpoint LiDAR (UNIQUE VALUE #1)
    # Creates different (sparse, complete) pairs from same scene
    enable_multi_viewpoint: bool = True
    viewpoint_offsets: list[tuple[int, int, int]] = None  # [(dx, dy, dz), ...]
    num_viewpoints: int = 4  # How many viewpoints per scene

    # Copy-paste (UNIQUE VALUE #2 - Class Rebalancing)
    # AGGRESSIVE settings to ensure ALL rare classes present and balanced
    enable_copy_paste: bool = True
    voxel_budget_ratio: float = 0.15  # Max 15% of scene voxels can be added
    paste_probability: float = 1.0  # Always paste
    ensure_all_classes: bool = True  # CRITICAL: Ensure ALL rare classes present
    min_voxels_per_class: int = 50  # Minimum voxels required for each rare class

    # 3D object bank settings
    use_3d_bank: bool = True  # Use 3D object bank (not BEV)
    object_bank_3d_path: str = "datasets/object_bank_3d"  # Path to 3D bank

    # LiDAR simulation
    lidar_h_resolution: int = 2048
    lidar_max_returns: int = 2
    lidar_samples_per_return: int = 2

    def __post_init__(self):
        if self.rotation_angles is None:
            self.rotation_angles = [90, 180, 270]
        if self.viewpoint_offsets is None:
            # Offsets in voxel coordinates (x, y, z)
            # Original sensor is at (0, 128, 10)
            self.viewpoint_offsets = [
                (0, 0, 0),      # Original
                (3, 0, 0),      # Forward
                (-3, 0, 0),     # Backward
                (0, 3, 0),      # Right
                (0, -3, 0),     # Left
                (3, 3, 0),      # Forward-right
                (3, -3, 0),     # Forward-left
                (-3, 3, 0),     # Backward-right
            ]


class GeometricAugmentor:
    """Apply geometric transformations to 3D voxel scenes."""

    @staticmethod
    def flip_x(scene: np.ndarray) -> np.ndarray:
        """Flip scene along X axis (forward-backward)."""
        return np.flip(scene, axis=0).copy()

    @staticmethod
    def flip_y(scene: np.ndarray) -> np.ndarray:
        """Flip scene along Y axis (left-right)."""
        return np.flip(scene, axis=1).copy()

    @staticmethod
    def rotate_90(scene: np.ndarray) -> np.ndarray:
        """Rotate scene 90 degrees around Z axis."""
        return np.rot90(scene, k=1, axes=(0, 1)).copy()

    @staticmethod
    def rotate_180(scene: np.ndarray) -> np.ndarray:
        """Rotate scene 180 degrees around Z axis."""
        return np.rot90(scene, k=2, axes=(0, 1)).copy()

    @staticmethod
    def rotate_270(scene: np.ndarray) -> np.ndarray:
        """Rotate scene 270 degrees around Z axis."""
        return np.rot90(scene, k=3, axes=(0, 1)).copy()

    def apply_transform(self, scene: np.ndarray, transform_name: str) -> np.ndarray:
        """Apply named transform to scene."""
        transforms = {
            'original': lambda x: x.copy(),
            'flip_x': self.flip_x,
            'flip_y': self.flip_y,
            'rot90': self.rotate_90,
            'rot180': self.rotate_180,
            'rot270': self.rotate_270,
            'flip_x_rot90': lambda x: self.rotate_90(self.flip_x(x)),
            'flip_y_rot90': lambda x: self.rotate_90(self.flip_y(x)),
        }
        if transform_name not in transforms:
            raise ValueError(f"Unknown transform: {transform_name}")
        return transforms[transform_name](scene)

    def get_all_transforms(self) -> list[str]:
        """Get list of all available transforms."""
        return ['original', 'flip_x', 'flip_y', 'rot90', 'rot180', 'rot270']


class CopyPasteAugmentor:
    """Copy-paste rare objects into scenes with overlap prevention."""

    def __init__(self, object_bank_path: str):
        """
        Initialize with object bank.

        Args:
            object_bank_path: Path to object bank directory
        """
        self.object_bank = ObjectBank.load(object_bank_path)
        logger.info(f"Loaded object bank with {self.object_bank.total_objects} objects")

    def is_valid_placement(
        self,
        scene: np.ndarray,
        obj_voxels: np.ndarray,
        obj_labels: np.ndarray,
        position: tuple[int, int, int],
    ) -> bool:
        """
        Check if placement is valid (no class conflicts, grounded).

        Args:
            scene: Current scene (256, 256, 32)
            obj_voxels: Object voxel coordinates [N, 3]
            obj_labels: Object voxel labels [N]
            position: Placement position (x, y, z)

        Returns:
            True if valid placement
        """
        px, py, pz = int(position[0]), int(position[1]), int(position[2])

        # Compute target positions (ensure integer indices)
        target_coords = (obj_voxels + np.array([px, py, pz])).astype(np.int32)

        # Check bounds
        if (target_coords[:, 0].min() < 0 or target_coords[:, 0].max() >= 256 or
            target_coords[:, 1].min() < 0 or target_coords[:, 1].max() >= 256 or
            target_coords[:, 2].min() < 0 or target_coords[:, 2].max() >= 32):
            return False

        # Check for class conflicts
        for i, (x, y, z) in enumerate(target_coords):
            existing_class = scene[x, y, z]
            new_class = obj_labels[i]

            if existing_class > 0 and existing_class != new_class:
                # Don't overwrite different non-empty classes
                # Exception: can overwrite unlabeled (0)
                return False

        # Check grounding: object should touch ground or be on vehicle
        min_z = target_coords[:, 2].min()
        if min_z > 2:  # Object floating too high
            # Check if there's ground below
            x_range = (target_coords[:, 0].min(), target_coords[:, 0].max())
            y_range = (target_coords[:, 1].min(), target_coords[:, 1].max())
            ground_below = scene[x_range[0]:x_range[1]+1, y_range[0]:y_range[1]+1, :min_z]
            ground_classes_present = np.isin(ground_below, list(GROUND_CLASSES))
            if not ground_classes_present.any():
                return False

        return True

    def find_valid_position(
        self,
        scene: np.ndarray,
        obj_voxels: np.ndarray,
        obj_labels: np.ndarray,
        max_attempts: int = 50,
    ) -> tuple[int, int, int] | None:
        """
        Find a valid position for placing the object.

        Strategy: Place on road/ground surfaces where they exist.
        """
        # Find ground voxels
        ground_mask = np.isin(scene, list(GROUND_CLASSES))
        ground_coords = np.argwhere(ground_mask)

        if len(ground_coords) == 0:
            return None

        obj_voxels.max(axis=0) - obj_voxels.min(axis=0)

        for _ in range(max_attempts):
            # Random ground position
            idx = np.random.randint(len(ground_coords))
            gx, gy, gz = ground_coords[idx]

            # Place object just above ground
            offset_z = gz + 1 - obj_voxels[:, 2].min()
            position = (gx - obj_voxels[:, 0].min(), gy - obj_voxels[:, 1].min(), offset_z)

            if self.is_valid_placement(scene, obj_voxels, obj_labels, position):
                return position

        return None

    def paste_object(
        self,
        scene: np.ndarray,
        class_id: int,
    ) -> tuple[np.ndarray, bool]:
        """
        Paste a random object of given class into scene.

        Returns:
            Modified scene and success flag
        """
        # Get random object from bank
        objects = self.object_bank.get_objects(class_id=class_id, num_samples=1)
        if not objects:
            return scene, False
        obj = objects[0]

        # Find valid position
        position = self.find_valid_position(scene, obj.voxels, obj.labels)
        if position is None:
            return scene, False

        # Paste object
        scene_copy = scene.copy()
        px, py, pz = int(position[0]), int(position[1]), int(position[2])
        for i, (x, y, z) in enumerate(obj.voxels):
            tx, ty, tz = int(x + px), int(y + py), int(z + pz)
            if 0 <= tx < 256 and 0 <= ty < 256 and 0 <= tz < 32:
                scene_copy[tx, ty, tz] = obj.labels[i]

        return scene_copy, True

    def augment_scene(
        self,
        scene: np.ndarray,
        num_objects: int = 2,
        target_classes: list[int] = None,
    ) -> np.ndarray:
        """
        Augment scene with multiple pasted objects.

        Args:
            scene: Input scene (256, 256, 32)
            num_objects: Number of objects to paste
            target_classes: Classes to paste (default: all rare classes)

        Returns:
            Augmented scene
        """
        if target_classes is None:
            target_classes = RARE_CLASSES

        augmented = scene.copy()
        pasted = 0

        for _ in range(num_objects * 3):  # Try multiple times
            if pasted >= num_objects:
                break

            # Random rare class
            class_id = np.random.choice(target_classes)
            augmented, success = self.paste_object(augmented, class_id)
            if success:
                pasted += 1

        return augmented


class RealDataAugmentor:
    """
    Complete real data augmentation pipeline.

    Combines geometric transforms, multi-viewpoint LiDAR, and copy-paste.
    """

    def __init__(
        self,
        object_bank_path: str = None,
        config: AugmentationConfig = None,
    ):
        """
        Initialize augmentor.

        Args:
            object_bank_path: Path to object bank (for copy-paste)
            config: Augmentation configuration
        """
        self.config = config or AugmentationConfig()
        self.geo_aug = GeometricAugmentor()

        # Initialize LiDAR simulator
        self.lidar_sim = create_multi_return_resampler(
            max_returns=self.config.lidar_max_returns,
            samples_per_return=self.config.lidar_samples_per_return,
        )

        # Initialize copy-paste if object bank provided
        self.copy_paste = None
        if object_bank_path:
            bank_path = Path(object_bank_path)
            if bank_path.exists():
                try:
                    # Handle directory with .pkl suffix (legacy naming)
                    if bank_path.is_dir():
                        self.copy_paste = CopyPasteAugmentor(str(bank_path))
                    else:
                        self.copy_paste = CopyPasteAugmentor(object_bank_path)
                except Exception as e:
                    logger.warning(f"Failed to load object bank: {e}")

    def simulate_lidar(
        self,
        scene: np.ndarray,
        viewpoint_offset: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Simulate LiDAR observation from viewpoint.

        For viewpoint offsets, we shift the scene in the opposite direction
        to simulate sensor being at a different position.

        Args:
            scene: Complete scene (256, 256, 32)
            viewpoint_offset: Offset from default sensor position (voxel coords)

        Returns:
            Sparse observation mask (256, 256, 32)
        """
        # If there's a viewpoint offset, shift scene in opposite direction
        if viewpoint_offset != (0, 0, 0):
            dx, dy, dz = viewpoint_offset
            shifted_scene = np.zeros_like(scene)

            # Compute valid ranges for the shift
            src_x = slice(max(0, dx), min(256, 256 + dx))
            src_y = slice(max(0, dy), min(256, 256 + dy))
            src_z = slice(max(0, dz), min(32, 32 + dz))

            dst_x = slice(max(0, -dx), min(256, 256 - dx))
            dst_y = slice(max(0, -dy), min(256, 256 - dy))
            dst_z = slice(max(0, -dz), min(32, 32 - dz))

            shifted_scene[dst_x, dst_y, dst_z] = scene[src_x, src_y, src_z]
            scene_to_sim = shifted_scene
        else:
            scene_to_sim = scene

        # Simulate LiDAR (returns sparse_mask, point_cloud, stats)
        sparse_mask, _, _ = self.lidar_sim.simulate(scene_to_sim)

        # If we shifted, shift the result back
        if viewpoint_offset != (0, 0, 0):
            dx, dy, dz = viewpoint_offset
            result = np.zeros_like(sparse_mask)

            src_x = slice(max(0, -dx), min(256, 256 - dx))
            src_y = slice(max(0, -dy), min(256, 256 - dy))
            src_z = slice(max(0, -dz), min(32, 32 - dz))

            dst_x = slice(max(0, dx), min(256, 256 + dx))
            dst_y = slice(max(0, dy), min(256, 256 + dy))
            dst_z = slice(max(0, dz), min(32, 32 + dz))

            result[dst_x, dst_y, dst_z] = sparse_mask[src_x, src_y, src_z]
            return result

        return sparse_mask

    def augment_scene(
        self,
        complete_scene: np.ndarray,
        source_name: str = "unknown",
    ) -> list[dict]:
        """
        Generate multiple augmented (sparse, complete) pairs from one scene.

        Args:
            complete_scene: Original complete scene (256, 256, 32)
            source_name: Source identifier for tracking

        Returns:
            List of dicts with 'sparse', 'complete', 'transform', 'viewpoint'
        """
        results = []

        # Select transforms
        transforms = ['original']
        if self.config.enable_flip_x:
            transforms.append('flip_x')
        if self.config.enable_flip_y:
            transforms.append('flip_y')
        if self.config.enable_rotation:
            for angle in self.config.rotation_angles:
                transforms.append(f'rot{angle}')

        # Select viewpoints
        num_vp = min(self.config.num_viewpoints, len(self.config.viewpoint_offsets))
        viewpoints = self.config.viewpoint_offsets[:num_vp]

        for transform in transforms:
            # Apply geometric transform
            transformed = self.geo_aug.apply_transform(complete_scene, transform)

            # Apply copy-paste to ensure ALL rare classes present and balanced
            if (self.config.enable_copy_paste and
                self.copy_paste is not None and
                np.random.random() < self.config.paste_probability):

                objects_added = 0
                objects_per_class = {}

                # PHASE 1: Ensure ALL rare classes present
                if self.config.ensure_all_classes:
                    for cls_id in RARE_CLASS_IDS[:8]:
                        current_count = np.sum(transformed == cls_id)
                        attempts = 0
                        max_attempts = 50  # Try harder to place each class

                        while (current_count < self.config.min_voxels_per_class and
                               attempts < max_attempts and
                               objects_added < self.config.max_paste_per_scene):
                            new_scene, success = self.copy_paste.paste_object(transformed, cls_id)
                            attempts += 1
                            if success:
                                transformed = new_scene
                                objects_added += 1
                                objects_per_class[cls_id] = objects_per_class.get(cls_id, 0) + 1
                                current_count = np.sum(transformed == cls_id)

                # PHASE 2: Balance - add more to reach target
                target_objects = np.random.randint(
                    self.config.min_paste_per_scene,
                    self.config.max_paste_per_scene + 1
                )

                max_rounds = 10
                for _ in range(max_rounds):
                    if objects_added >= target_objects:
                        break

                    # Sort classes by count (lowest first)
                    class_counts = [(cls_id, np.sum(transformed == cls_id))
                                    for cls_id in RARE_CLASS_IDS[:8]]
                    sorted_classes = sorted(class_counts, key=lambda x: x[1])

                    for cls_id, _ in sorted_classes:
                        if objects_added >= target_objects:
                            break
                        if objects_per_class.get(cls_id, 0) >= self.config.max_objects_per_class:
                            continue

                        new_scene, success = self.copy_paste.paste_object(transformed, cls_id)
                        if success:
                            transformed = new_scene
                            objects_added += 1
                            objects_per_class[cls_id] = objects_per_class.get(cls_id, 0) + 1

            # Generate sparse observations from multiple viewpoints
            for vp_idx, vp_offset in enumerate(viewpoints):
                sparse_mask = self.simulate_lidar(transformed, vp_offset)

                results.append({
                    'sparse': sparse_mask,
                    'complete': transformed,
                    'transform': transform,
                    'viewpoint': vp_offset,
                    'viewpoint_idx': vp_idx,
                    'source': source_name,
                })

        return results

    def save_pair(
        self,
        sparse_mask: np.ndarray,
        complete_scene: np.ndarray,
        output_dir: Path,
        frame_id: str,
    ):
        """Save pair in SemanticKITTI format."""
        voxels_dir = output_dir / 'voxels'
        voxels_dir.mkdir(parents=True, exist_ok=True)

        # Pack sparse mask to binary
        sparse_packed = pack_voxels(sparse_mask)
        sparse_packed.tofile(voxels_dir / f'{frame_id}.bin')

        # Convert complete labels back to original format
        complete_orig = np.zeros_like(complete_scene, dtype=np.uint16)
        for train_id, orig_id in LEARNING_MAP_INV.items():
            complete_orig[complete_scene == train_id] = orig_id
        complete_orig.tofile(voxels_dir / f'{frame_id}.label')

        # Create invalid mask (zeros for synthetic)
        invalid = np.zeros_like(sparse_mask, dtype=np.uint8)
        invalid_packed = pack_voxels(invalid)
        invalid_packed.tofile(voxels_dir / f'{frame_id}.invalid')

    def process_dataset(
        self,
        ssc_root: str,
        output_dir: str,
        sequences: list[int] = None,
        max_scenes: int = None,
    ) -> dict:
        """
        Process entire dataset with augmentation.

        Args:
            ssc_root: Path to SemanticKITTI SSC dataset
            output_dir: Output directory
            sequences: Sequences to process (default: train sequences)
            max_scenes: Maximum scenes to process (for testing)

        Returns:
            Statistics dict
        """
        if sequences is None:
            sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

        ssc_root = Path(ssc_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'total_input_scenes': 0,
            'total_output_pairs': 0,
            'class_counts': {},
        }

        frame_counter = 0

        for seq in tqdm(sequences, desc='Sequences'):
            seq_dir = ssc_root / 'sequences' / f'{seq:02d}' / 'voxels'
            if not seq_dir.exists():
                logger.warning(f"Sequence {seq:02d} not found")
                continue

            label_files = sorted(seq_dir.glob('*.label'))

            for label_file in tqdm(label_files, desc=f'Seq {seq:02d}', leave=False):
                if max_scenes and stats['total_input_scenes'] >= max_scenes:
                    break

                # Load complete scene
                labels = np.fromfile(label_file, dtype=np.uint16).reshape(256, 256, 32)

                # Apply learning map
                complete = np.zeros_like(labels, dtype=np.uint8)
                for orig, target in LEARNING_MAP.items():
                    complete[labels == orig] = target

                # Augment scene
                pairs = self.augment_scene(
                    complete,
                    source_name=f'{seq:02d}_{label_file.stem}'
                )

                # Save pairs
                for pair in pairs:
                    frame_id = f'{frame_counter:06d}'
                    self.save_pair(
                        pair['sparse'],
                        pair['complete'],
                        output_dir / 'sequences' / 'augmented',
                        frame_id,
                    )
                    frame_counter += 1
                    stats['total_output_pairs'] += 1

                    # Update class counts
                    for cls in range(20):
                        cnt = (pair['complete'] == cls).sum()
                        stats['class_counts'][cls] = stats['class_counts'].get(cls, 0) + cnt

                stats['total_input_scenes'] += 1

        # Save stats
        with open(output_dir / 'augmentation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Generated {stats['total_output_pairs']} pairs from {stats['total_input_scenes']} scenes")

        return stats


def create_augmentor(
    object_bank_path: str = None,
    num_viewpoints: int = 4,
    enable_copy_paste: bool = True,
) -> RealDataAugmentor:
    """
    Factory function to create augmentor with common settings.

    Args:
        object_bank_path: Path to object bank
        num_viewpoints: Number of viewpoints per scene
        enable_copy_paste: Enable copy-paste augmentation

    Returns:
        Configured RealDataAugmentor
    """
    config = AugmentationConfig(
        num_viewpoints=num_viewpoints,
        enable_copy_paste=enable_copy_paste,
    )
    return RealDataAugmentor(object_bank_path, config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real Data Augmentation')
    parser.add_argument('--ssc_root', type=str, required=True,
                        help='SemanticKITTI SSC dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--object_bank', type=str, default='datasets/object_bank.pkl',
                        help='Object bank path')
    parser.add_argument('--sequences', type=int, nargs='+', default=None,
                        help='Sequences to process')
    parser.add_argument('--max_scenes', type=int, default=None,
                        help='Max scenes (for testing)')
    parser.add_argument('--num_viewpoints', type=int, default=4,
                        help='Viewpoints per scene')
    parser.add_argument('--no_copy_paste', action='store_true',
                        help='Disable copy-paste')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    augmentor = create_augmentor(
        object_bank_path=args.object_bank,
        num_viewpoints=args.num_viewpoints,
        enable_copy_paste=not args.no_copy_paste,
    )

    stats = augmentor.process_dataset(
        ssc_root=args.ssc_root,
        output_dir=args.output_dir,
        sequences=args.sequences,
        max_scenes=args.max_scenes,
    )

    print("\nAugmentation complete!")
    print(f"  Input scenes: {stats['total_input_scenes']}")
    print(f"  Output pairs: {stats['total_output_pairs']}")
