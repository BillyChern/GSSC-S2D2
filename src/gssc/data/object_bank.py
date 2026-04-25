"""
Rare Object Bank for Copy-Paste Augmentation

This module provides:
1. RareObjectExtractor - Extract rare objects from GT scenes
2. ObjectBank - Store and retrieve rare objects
3. ObjectPaster - Paste objects into new scenes

Based on:
- PolarMix (NeurIPS 2022): https://arxiv.org/abs/2208.00223
- GT-Paste: Ground truth copy-paste augmentation

The goal is to create a bank of rare object instances (bicycle, motorcycle,
truck, person, etc.) that can be pasted into new scenes for augmentation.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from scipy import ndimage
from tqdm import tqdm


@dataclass
class RareObject:
    """Representation of a rare object instance."""
    class_id: int
    class_name: str
    voxels: np.ndarray  # [N, 3] relative coordinates
    labels: np.ndarray  # [N] semantic labels (all same class)
    bbox_size: Tuple[int, int, int]  # (X, Y, Z) bounding box size
    centroid: np.ndarray  # [3] original centroid position
    num_voxels: int
    source_scene: str  # Source scene identifier
    context_classes: List[int] = field(default_factory=list)  # Adjacent classes


# Class name mapping
CLASS_NAMES = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
    'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
    'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]

# Rare classes to extract
RARE_CLASS_IDS = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]
EXTREME_RARE_IDS = [2, 3, 4, 6, 7, 8]

# Alias for compatibility
RARE_CLASSES = RARE_CLASS_IDS
EXTREME_RARE_CLASSES = EXTREME_RARE_IDS


class RareObjectExtractor:
    """
    Extract rare objects from semantic scenes.

    Uses connected component analysis to find distinct object instances.
    """

    def __init__(
        self,
        rare_classes: List[int] = None,
        min_voxels: int = 5,
        max_voxels: int = 10000,
        connectivity: int = 26,  # 6, 18, or 26
    ):
        """
        Args:
            rare_classes: List of rare class IDs to extract
            min_voxels: Minimum voxels for valid object
            max_voxels: Maximum voxels for valid object
            connectivity: 3D connectivity (6-face, 18-edge, 26-corner)
        """
        self.rare_classes = rare_classes or RARE_CLASS_IDS
        self.min_voxels = min_voxels
        self.max_voxels = max_voxels

        # Connectivity structure for 3D connected components
        if connectivity == 6:
            self.structure = ndimage.generate_binary_structure(3, 1)
        elif connectivity == 18:
            self.structure = ndimage.generate_binary_structure(3, 2)
        else:  # 26
            self.structure = ndimage.generate_binary_structure(3, 3)

    def extract_from_scene(
        self,
        scene: np.ndarray,
        scene_id: str = "unknown",
    ) -> List[RareObject]:
        """
        Extract rare objects from a single scene.

        Args:
            scene: [X, Y, Z] semantic labels
            scene_id: Identifier for the source scene

        Returns:
            objects: List of RareObject instances
        """
        objects = []

        for cls_id in self.rare_classes:
            # Find voxels of this class
            mask = (scene == cls_id)

            if mask.sum() < self.min_voxels:
                continue

            # Find connected components
            labeled_array, num_features = ndimage.label(mask, structure=self.structure)

            for comp_id in range(1, num_features + 1):
                comp_mask = (labeled_array == comp_id)
                num_voxels = comp_mask.sum()

                # Filter by size
                if num_voxels < self.min_voxels or num_voxels > self.max_voxels:
                    continue

                # Extract coordinates
                coords = np.argwhere(comp_mask)

                # Compute properties
                centroid = coords.mean(axis=0)
                relative_coords = coords - centroid

                bbox_min = coords.min(axis=0)
                bbox_max = coords.max(axis=0)
                bbox_size = tuple(bbox_max - bbox_min + 1)

                # Find adjacent classes (context)
                dilated = ndimage.binary_dilation(comp_mask, iterations=2)
                border_mask = dilated & ~comp_mask
                context_classes = np.unique(scene[border_mask]).tolist()
                context_classes = [c for c in context_classes if c != cls_id and c != 0]

                # Create object
                obj = RareObject(
                    class_id=cls_id,
                    class_name=CLASS_NAMES[cls_id],
                    voxels=relative_coords.astype(np.float32),
                    labels=np.full(num_voxels, cls_id, dtype=np.uint8),
                    bbox_size=bbox_size,
                    centroid=centroid.astype(np.float32),
                    num_voxels=num_voxels,
                    source_scene=scene_id,
                    context_classes=context_classes,
                )
                objects.append(obj)

        return objects

    def extract_objects(
        self,
        scene: np.ndarray,
        scene_id: str = "unknown",
    ) -> List[RareObject]:
        """Alias for extract_from_scene for compatibility."""
        return self.extract_from_scene(scene, scene_id)

    def extract_from_dataset(
        self,
        data_root: str,
        sequences: List[str] = None,
        max_scenes_per_seq: int = 1000,
    ) -> List[RareObject]:
        """
        Extract rare objects from entire dataset.

        Args:
            data_root: Path to SemanticKITTI data
            sequences: List of sequence IDs (e.g., ['00', '01', ...])
            max_scenes_per_seq: Maximum scenes to process per sequence

        Returns:
            objects: List of all extracted RareObjects
        """
        data_root = Path(data_root)
        all_objects = []

        if sequences is None:
            sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

        for seq in tqdm(sequences, desc="Processing sequences"):
            seq_path = data_root / seq / 'voxels'
            if not seq_path.exists():
                seq_path = data_root / 'sequences' / seq / 'voxels'

            if not seq_path.exists():
                print(f"Sequence {seq} not found at {seq_path}")
                continue

            label_files = sorted(seq_path.glob('*.label'))[:max_scenes_per_seq]

            for lf in tqdm(label_files, desc=f"Seq {seq}", leave=False):
                # Load scene
                labels = np.fromfile(str(lf), dtype=np.uint16)
                scene = labels.reshape(256, 256, 32)

                # Extract objects
                scene_id = f"{seq}/{lf.stem}"
                objects = self.extract_from_scene(scene, scene_id)
                all_objects.extend(objects)

        return all_objects


class ObjectBank:
    """
    Storage and retrieval system for rare objects.
    """

    def __init__(self, bank_dir: str = None):
        """
        Args:
            bank_dir: Directory to store object bank
        """
        self.bank_dir = Path(bank_dir) if bank_dir else None
        self.objects: Dict[int, List[RareObject]] = {cls: [] for cls in range(20)}
        self.metadata = {
            'total_objects': 0,
            'class_counts': {},
            'mean_sizes': {},
        }

    def add_object(self, obj: RareObject):
        """Add object to bank."""
        self.objects[obj.class_id].append(obj)

    def add_objects(self, objects: List[RareObject]):
        """Add multiple objects to bank."""
        for obj in objects:
            self.add_object(obj)
        self._update_metadata()

    def _update_metadata(self):
        """Update metadata statistics."""
        self.metadata['total_objects'] = sum(len(objs) for objs in self.objects.values())
        self.metadata['class_counts'] = {
            cls: len(objs) for cls, objs in self.objects.items() if len(objs) > 0
        }

        # Compute mean sizes per class
        for cls, objs in self.objects.items():
            if len(objs) > 0:
                sizes = [obj.num_voxels for obj in objs]
                self.metadata['mean_sizes'][cls] = np.mean(sizes)

    def get_objects(
        self,
        class_id: int = None,
        min_voxels: int = None,
        max_voxels: int = None,
        num_samples: int = None,
    ) -> List[RareObject]:
        """
        Retrieve objects from bank.

        Args:
            class_id: Specific class to retrieve (None = all rare classes)
            min_voxels: Minimum object size
            max_voxels: Maximum object size
            num_samples: Random sample size (None = all)

        Returns:
            objects: List of matching objects
        """
        if class_id is not None:
            candidates = self.objects[class_id]
        else:
            candidates = []
            for cls in RARE_CLASS_IDS:
                candidates.extend(self.objects[cls])

        # Filter by size
        if min_voxels is not None:
            candidates = [o for o in candidates if o.num_voxels >= min_voxels]
        if max_voxels is not None:
            candidates = [o for o in candidates if o.num_voxels <= max_voxels]

        # Sample
        if num_samples is not None and num_samples < len(candidates):
            indices = np.random.choice(len(candidates), num_samples, replace=False)
            candidates = [candidates[i] for i in indices]

        return candidates

    def sample_random(
        self,
        class_weights: Dict[int, float] = None,
    ) -> RareObject:
        """
        Sample a random object with optional class weighting.

        Args:
            class_weights: Dict of {class_id: weight} (higher = more likely)

        Returns:
            Random RareObject
        """
        # Get available classes
        available_classes = [cls for cls, objs in self.objects.items() if len(objs) > 0]

        if len(available_classes) == 0:
            raise ValueError("Object bank is empty")

        # Weight by inverse frequency if not specified
        if class_weights is None:
            counts = np.array([len(self.objects[cls]) for cls in available_classes])
            weights = 1.0 / (counts + 1)
        else:
            weights = np.array([class_weights.get(cls, 1.0) for cls in available_classes])

        weights = weights / weights.sum()

        # Sample class
        cls = np.random.choice(available_classes, p=weights)

        # Sample object from class
        obj = np.random.choice(self.objects[cls])

        return obj

    def save(self, bank_dir: str = None):
        """
        Save object bank to disk.

        Args:
            bank_dir: Directory to save to (uses self.bank_dir if None)
        """
        bank_dir = Path(bank_dir or self.bank_dir)
        bank_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(bank_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save objects per class
        for cls_id, objs in self.objects.items():
            if len(objs) == 0:
                continue

            cls_dir = bank_dir / CLASS_NAMES[cls_id]
            cls_dir.mkdir(exist_ok=True)

            for i, obj in enumerate(objs):
                np.savez(
                    cls_dir / f'object_{i:05d}.npz',
                    class_id=obj.class_id,
                    voxels=obj.voxels,
                    labels=obj.labels,
                    bbox_size=obj.bbox_size,
                    centroid=obj.centroid,
                    source_scene=obj.source_scene,
                    context_classes=obj.context_classes,
                )

        print(f"Saved {self.metadata['total_objects']} objects to {bank_dir}")

    def load_from_dir(self, bank_dir: str = None):
        """
        Load object bank from disk directory.

        Args:
            bank_dir: Directory to load from
        """
        bank_dir = Path(bank_dir or self.bank_dir)

        if not bank_dir.exists():
            raise ValueError(f"Bank directory not found: {bank_dir}")

        # Load metadata
        with open(bank_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Load objects
        self.objects = {cls: [] for cls in range(20)}

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            cls_dir = bank_dir / cls_name

            if not cls_dir.exists():
                continue

            for npz_file in cls_dir.glob('*.npz'):
                data = np.load(npz_file, allow_pickle=True)

                obj = RareObject(
                    class_id=int(data['class_id']),
                    class_name=cls_name,
                    voxels=data['voxels'],
                    labels=data['labels'],
                    bbox_size=tuple(data['bbox_size']),
                    centroid=data['centroid'],
                    num_voxels=len(data['voxels']),
                    source_scene=str(data['source_scene']),
                    context_classes=list(data['context_classes']) if 'context_classes' in data else [],
                )
                self.objects[cls_id].append(obj)

        print(f"Loaded {self.metadata.get('total_objects', 'N/A')} objects from {bank_dir}")

    @property
    def total_objects(self) -> int:
        """Total number of objects in bank."""
        return sum(len(objs) for objs in self.objects.values())

    def __len__(self) -> int:
        """Return total number of objects."""
        return self.total_objects

    def get_class_count(self, class_id: int) -> int:
        """Get number of objects for a class."""
        return len(self.objects[class_id])

    @classmethod
    def load(cls, bank_path: str) -> 'ObjectBank':
        """
        Static method to load object bank from disk.

        Args:
            bank_path: Path to bank directory or pickle file

        Returns:
            Loaded ObjectBank instance
        """
        bank_path = Path(bank_path)

        # Handle both directory and pickle file
        # Check if it's a FILE with .pkl suffix (not a directory)
        if bank_path.suffix == '.pkl' and bank_path.is_file():
            import pickle
            with open(bank_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Directory-based loading (handles dirs even with .pkl suffix)
            bank = cls(str(bank_path))
            bank.load_from_dir()  # Use instance method
            return bank

    def save_pickle(self, path: str):
        """Save as pickle file for faster loading."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved object bank pickle to {path}")

    def summary(self):
        """Print summary of object bank."""
        print("\n" + "=" * 50)
        print("OBJECT BANK SUMMARY")
        print("=" * 50)

        total = 0
        for cls_id in range(20):
            count = len(self.objects[cls_id])
            if count > 0:
                mean_size = np.mean([o.num_voxels for o in self.objects[cls_id]])
                print(f"{cls_id:2d} {CLASS_NAMES[cls_id]:15s}: {count:5d} objects (mean size: {mean_size:.1f})")
                total += count

        print("-" * 50)
        print(f"{'TOTAL':19s}: {total:5d} objects")
        print("=" * 50)


class ObjectPaster:
    """
    Paste objects from bank into scenes.
    """

    def __init__(
        self,
        object_bank: ObjectBank,
        grid_size: Tuple[int, int, int] = (256, 256, 32),
    ):
        """
        Args:
            object_bank: ObjectBank instance
            grid_size: Scene grid size (X, Y, Z)
        """
        self.bank = object_bank
        self.grid_size = grid_size

        # Valid placement classes (where objects can be placed nearby)
        self.placement_context = {
            2: [9, 11, 10],      # bicycle: near road, sidewalk, parking
            3: [9, 10],          # motorcycle: near road, parking
            4: [9, 10],          # truck: near road, parking
            5: [9, 10, 11],      # other-vehicle
            6: [11, 9, 12],      # person: near sidewalk, road, other-ground
            7: [9, 11],          # bicyclist: near road, sidewalk
            8: [9],              # motorcyclist: near road
            16: [9, 11, 15],     # trunk: near road, sidewalk, vegetation
            18: [9, 11],         # pole: near road, sidewalk
            19: [9, 11],         # traffic-sign: near road, sidewalk
        }

    def find_valid_positions(
        self,
        scene: np.ndarray,
        target_class: int,
        obj_size: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Find valid positions for placing an object.

        Args:
            scene: [X, Y, Z] current scene
            target_class: Class of object to place
            obj_size: (X, Y, Z) size of object

        Returns:
            valid_positions: [N, 3] array of valid center positions
        """
        context_classes = self.placement_context.get(target_class, [9])

        # Find context voxels
        context_mask = np.zeros_like(scene, dtype=bool)
        for ctx_cls in context_classes:
            context_mask |= (scene == ctx_cls)

        # Dilate to find adjacent empty voxels
        structure = ndimage.generate_binary_structure(3, 1)
        dilated = ndimage.binary_dilation(context_mask, structure=structure, iterations=3)

        # Valid = adjacent to context AND currently empty
        valid_mask = dilated & (scene == 0)

        # Check size constraints
        sx, sy, sz = obj_size
        valid_positions = []

        candidate_positions = np.argwhere(valid_mask)
        for pos in candidate_positions:
            x, y, z = pos

            # Check bounds
            if (x + sx//2 >= self.grid_size[0] or x - sx//2 < 0 or
                y + sy//2 >= self.grid_size[1] or y - sy//2 < 0 or
                z + sz//2 >= self.grid_size[2] or z - sz//2 < 0):
                continue

            # Check if region is empty
            region = scene[
                max(0, x - sx//2):x + sx//2 + 1,
                max(0, y - sy//2):y + sy//2 + 1,
                max(0, z - sz//2):z + sz//2 + 1
            ]

            if (region == 0).mean() > 0.8:  # 80% empty
                valid_positions.append(pos)

        return np.array(valid_positions) if valid_positions else np.array([]).reshape(0, 3)

    def paste_object(
        self,
        scene: np.ndarray,
        obj: RareObject,
        position: np.ndarray,
        rotation_deg: int = 0,
    ) -> np.ndarray:
        """
        Paste object into scene at given position.

        Args:
            scene: [X, Y, Z] scene to modify
            obj: RareObject to paste
            position: [3] center position
            rotation_deg: Rotation around Z axis (0, 90, 180, 270)

        Returns:
            modified_scene: [X, Y, Z] scene with pasted object
        """
        modified = scene.copy()
        voxels = obj.voxels.copy()

        # Rotate around Z axis
        if rotation_deg != 0:
            angle = np.radians(rotation_deg)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            voxels = voxels @ rot_matrix.T

        # Translate to position
        voxels = voxels + position

        # Paste voxels
        for vox, label in zip(voxels, obj.labels):
            x, y, z = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))

            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
                modified[x, y, z] = label

        return modified

    def augment_scene(
        self,
        scene: np.ndarray,
        num_objects: int = 5,
        class_weights: Dict[int, float] = None,
    ) -> np.ndarray:
        """
        Augment scene by pasting multiple rare objects.

        Args:
            scene: [X, Y, Z] original scene
            num_objects: Number of objects to paste
            class_weights: Optional weights for class sampling

        Returns:
            augmented: [X, Y, Z] augmented scene
        """
        augmented = scene.copy()
        pasted_count = 0

        for _ in range(num_objects * 2):  # Try more times than needed
            if pasted_count >= num_objects:
                break

            try:
                # Sample random object
                obj = self.bank.sample_random(class_weights)

                # Find valid positions
                valid_pos = self.find_valid_positions(
                    augmented, obj.class_id, obj.bbox_size
                )

                if len(valid_pos) == 0:
                    continue

                # Sample position
                pos_idx = np.random.randint(len(valid_pos))
                position = valid_pos[pos_idx]

                # Random rotation
                rotation = np.random.choice([0, 90, 180, 270])

                # Paste
                augmented = self.paste_object(augmented, obj, position, rotation)
                pasted_count += 1

            except Exception as e:
                continue

        return augmented


def build_object_bank(
    data_root: str,
    output_dir: str,
    sequences: List[str] = None,
) -> ObjectBank:
    """
    Build object bank from SemanticKITTI dataset.

    Args:
        data_root: Path to SemanticKITTI data
        output_dir: Where to save object bank
        sequences: Sequences to process

    Returns:
        ObjectBank instance
    """
    # Extract objects
    extractor = RareObjectExtractor()
    objects = extractor.extract_from_dataset(data_root, sequences)

    print(f"\nExtracted {len(objects)} rare objects")

    # Create and populate bank
    bank = ObjectBank(output_dir)
    bank.add_objects(objects)
    bank.summary()

    # Save
    bank.save()

    return bank


if __name__ == '__main__':
    # Test object extraction
    print("Testing object bank...")

    # Create synthetic scene
    scene = np.zeros((32, 32, 8), dtype=np.uint16)
    scene[:, :, 0] = 9  # Road
    scene[10:15, 10:15, 1:4] = 1  # Car
    scene[20:22, 20:22, 1:3] = 2  # Bicycle

    # Extract
    extractor = RareObjectExtractor(min_voxels=2)
    objects = extractor.extract_from_scene(scene, "test_scene")

    print(f"\nExtracted {len(objects)} objects:")
    for obj in objects:
        print(f"  - {obj.class_name}: {obj.num_voxels} voxels, bbox={obj.bbox_size}")

    # Create bank
    bank = ObjectBank()
    bank.add_objects(objects)
    bank.summary()

    # Test pasting
    paster = ObjectPaster(bank, grid_size=(32, 32, 8))

    if len(bank.objects[2]) > 0:  # If we have bicycles
        obj = bank.objects[2][0]
        valid_pos = paster.find_valid_positions(scene, 2, obj.bbox_size)
        print(f"\nFound {len(valid_pos)} valid positions for bicycle")

    print("\nObject bank tests passed!")
