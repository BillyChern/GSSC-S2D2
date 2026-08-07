"""
Post-Processing Pipeline for Cascaded S1→S2→S3 Generation

This module provides quality filtering, geometric cleaning, and enhancement
for synthetically generated scenes from the pyramid diffusion cascade.

Pipeline:
  Generated S3 Scene (256×256×32)
          │
          ↓
  Step 1: Quality Filtering
  ├── Reject if occupied < min_voxels
  ├── Reject if missing ground classes
  └── Reject if unlabeled ratio too high
          │
          ↓
  Step 2: Geometric Cleaning
  ├── Remove floating objects (disconnected from ground)
  ├── Morphological closing (fill small holes)
  └── Remove isolated noise (small connected components)
          │
          ↓
  Step 3: Rare Class Enhancement
  ├── Copy-paste from Object Bank
  ├── Inpainting-style insertion
  └── Class-aware placement
          │
          ↓
  Clean Scene → LiDAR Simulation → (sparse, complete) pair
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Import object bank for enhancement
try:
    from .object_bank import CLASS_NAMES, RARE_CLASS_IDS, ObjectBank
    from .real_augmentation import CopyPasteAugmentor
except ImportError:
    from object_bank import CLASS_NAMES, RARE_CLASS_IDS, ObjectBank
    from real_augmentation import CopyPasteAugmentor

logger = logging.getLogger(__name__)


# Ground/surface classes
GROUND_CLASSES = {9, 10, 11, 12, 17}  # road, parking, sidewalk, other-ground, terrain

# Structure classes (should be grounded)
STRUCTURE_CLASSES = {13, 14}  # building, fence

# Vegetation classes
VEGETATION_CLASSES = {15, 16}  # vegetation, trunk


@dataclass
class PostProcessConfig:
    """Configuration for cascade post-processing."""

    # Quality filtering thresholds
    min_occupied_voxels: int = 50000
    max_unlabeled_ratio: float = 0.5
    require_ground: bool = True
    min_ground_voxels: int = 5000

    # Geometric cleaning
    remove_floating: bool = True
    floating_height_threshold: int = 5  # voxels above this without support
    min_component_size: int = 5  # remove components smaller than this
    apply_closing: bool = True
    closing_iterations: int = 1

    # Rare class enhancement (AGGRESSIVE - ensure all classes present and balanced)
    enable_enhancement: bool = True
    voxel_budget_ratio: float = 0.15  # Max 15% of scene voxels can be added
    ensure_all_classes: bool = True  # CRITICAL: Ensure ALL rare classes present
    min_voxels_per_class: int = 50  # Minimum voxels required for each rare class
    target_rare_count: int = 200  # Target voxels per rare class after balancing
    # Paste-count budget for RareClassEnhancer. These four are read by enhance() at lines ~318,
    # 378, 396-397 and 416; they were dropped when the pipeline gained voxel_budget_ratio, which
    # left enhance() raising AttributeError on its own config for every scene. The values are the
    # ones tools/cascaded_generation.py passes, so caller and callee agree again.
    min_objects_to_add: int = 20      # lower bound on the per-scene paste budget
    max_objects_to_add: int = 50      # upper bound; also caps Phase 1 across all classes
    max_objects_per_class: int = 10   # per-class ceiling inside the budget
    balance_target_ratio: float = 1.0 # scales DATASET_AVERAGE_COUNTS when adaptive_balance is on
    enhancement_probability: float = 1.0  # Always enhance
    adaptive_balance: bool = True  # Enable adaptive per-scene balancing
    use_3d_bank: bool = True  # Use 3D object bank (not BEV)
    object_bank_3d_path: str = "datasets/object_bank_3d"  # Path to 3D bank

    # Inpainting options
    enable_inpainting: bool = False
    inpaint_anchor_classes: list[int] = None  # Classes to use as anchors

    def __post_init__(self):
        if self.inpaint_anchor_classes is None:
            self.inpaint_anchor_classes = [9, 11]  # road, sidewalk


class QualityFilter:
    """
    Step 1: Quality filtering to reject invalid generated scenes.
    """

    def __init__(self, config: PostProcessConfig):
        self.config = config

    def check(self, scene: np.ndarray) -> tuple[bool, dict]:
        """
        Check if scene passes quality filters.

        Returns:
            (passed, stats_dict)
        """
        stats = {}

        # Count occupied voxels
        occupied = np.sum(scene > 0)
        stats['occupied_voxels'] = occupied

        if occupied < self.config.min_occupied_voxels:
            stats['reject_reason'] = f'too_few_voxels ({occupied} < {self.config.min_occupied_voxels})'
            return False, stats

        # Check unlabeled ratio (class 0 among all voxels)
        total_voxels = scene.size
        unlabeled = np.sum(scene == 0)
        unlabeled_ratio = unlabeled / total_voxels
        stats['unlabeled_ratio'] = unlabeled_ratio

        # Note: This check might be too strict since most voxels are empty
        # Better to check ratio among occupied voxels

        # Check ground presence
        ground_mask = np.isin(scene, list(GROUND_CLASSES))
        ground_voxels = np.sum(ground_mask)
        stats['ground_voxels'] = ground_voxels

        if self.config.require_ground and ground_voxels < self.config.min_ground_voxels:
            stats['reject_reason'] = f'insufficient_ground ({ground_voxels} < {self.config.min_ground_voxels})'
            return False, stats

        # Check class diversity
        unique_classes = np.unique(scene)
        stats['num_classes'] = len(unique_classes)

        if len(unique_classes) < 3:  # Need at least a few classes
            stats['reject_reason'] = f'low_diversity ({len(unique_classes)} classes)'
            return False, stats

        stats['passed'] = True
        return True, stats


class GeometricCleaner:
    """
    Step 2: Geometric cleaning to remove artifacts.
    """

    def __init__(self, config: PostProcessConfig):
        self.config = config
        # 3D connectivity structure (26-connected)
        self.structure = ndimage.generate_binary_structure(3, 3)

    def clean(self, scene: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply geometric cleaning operations.

        Returns:
            (cleaned_scene, stats_dict)
        """
        cleaned = scene.copy()
        stats = {'original_occupied': np.sum(scene > 0)}

        # 1. Remove floating objects
        if self.config.remove_floating:
            cleaned, removed = self._remove_floating(cleaned)
            stats['floating_removed'] = removed

        # 2. Remove small isolated components
        if self.config.min_component_size > 1:
            cleaned, removed = self._remove_small_components(cleaned)
            stats['small_components_removed'] = removed

        # 3. Morphological closing (fill small holes)
        if self.config.apply_closing:
            cleaned = self._apply_closing(cleaned)

        stats['final_occupied'] = np.sum(cleaned > 0)
        stats['voxels_removed'] = stats['original_occupied'] - stats['final_occupied']

        return cleaned, stats

    def _remove_floating(self, scene: np.ndarray) -> tuple[np.ndarray, int]:
        """Remove objects not connected to ground."""
        # Find ground layer (z=0 to z=2)
        ground_mask = np.zeros_like(scene, dtype=bool)
        for z in range(min(3, scene.shape[2])):
            ground_mask[:, :, z] = scene[:, :, z] > 0

        # Also include known ground classes at any height
        ground_classes_mask = np.isin(scene, list(GROUND_CLASSES))
        ground_mask |= ground_classes_mask

        # Find all voxels connected to ground
        occupied = scene > 0
        labeled, num_features = ndimage.label(occupied, structure=self.structure)

        # Find which components touch ground
        ground_components = set()
        for comp_id in range(1, num_features + 1):
            if np.any(ground_mask & (labeled == comp_id)):
                ground_components.add(comp_id)

        # Keep only grounded components
        cleaned = scene.copy()
        removed = 0
        for comp_id in range(1, num_features + 1):
            if comp_id not in ground_components:
                comp_mask = labeled == comp_id
                # Check if this is a high floating component
                z_coords = np.where(comp_mask)[2]
                if len(z_coords) > 0 and z_coords.min() > self.config.floating_height_threshold:
                    cleaned[comp_mask] = 0
                    removed += np.sum(comp_mask)

        return cleaned, removed

    def _remove_small_components(self, scene: np.ndarray) -> tuple[np.ndarray, int]:
        """Remove small isolated components."""
        occupied = scene > 0
        labeled, num_features = ndimage.label(occupied, structure=self.structure)

        cleaned = scene.copy()
        removed = 0

        for comp_id in range(1, num_features + 1):
            comp_mask = labeled == comp_id
            size = np.sum(comp_mask)
            if size < self.config.min_component_size:
                cleaned[comp_mask] = 0
                removed += size

        return cleaned, removed

    def _apply_closing(self, scene: np.ndarray) -> np.ndarray:
        """Apply morphological closing per class to fill holes."""
        closed = scene.copy()

        # Small closing structure
        struct = ndimage.generate_binary_structure(3, 1)

        # Apply closing per major class
        for cls in [1, 9, 13, 15]:  # car, road, building, vegetation
            mask = scene == cls
            if np.sum(mask) > 100:
                closed_mask = ndimage.binary_closing(
                    mask, structure=struct,
                    iterations=self.config.closing_iterations
                )
                # Only fill holes, don't remove existing voxels
                new_voxels = closed_mask & ~mask & (closed == 0)
                closed[new_voxels] = cls

        return closed


class RareClassEnhancer:
    """
    Step 3: Enhance rare class presence using Object Bank.

    Enhanced version with adaptive per-scene class balancing:
    - Analyzes which classes are underrepresented in each scene
    - Prioritizes adding objects of the rarest classes first
    - Adds more objects (up to 20 per scene) for better balance
    """

    # Average voxel counts per class from SemanticKITTI dataset (approximate)
    DATASET_AVERAGE_COUNTS = {
        1: 2500,   # car
        2: 150,    # bicycle
        3: 200,    # motorcycle
        4: 500,    # truck
        5: 300,    # other-vehicle
        6: 100,    # person
        7: 80,     # bicyclist
        8: 50,     # motorcyclist
    }

    def __init__(self, config: PostProcessConfig, object_bank_path: str = None):
        self.config = config
        self.object_bank = None
        self.copy_paste = None

        if object_bank_path and config.enable_enhancement:
            try:
                self.copy_paste = CopyPasteAugmentor(object_bank_path)
                self.object_bank = self.copy_paste.object_bank
            except Exception as e:
                logger.warning("Could not load object bank: %s", e)

    def _compute_class_deficits(self, scene: np.ndarray) -> dict[int, float]:
        """
        Compute how underrepresented each class is in the scene.

        Returns:
            Dict mapping class_id -> deficit_score (higher = more underrepresented)
        """
        deficits = {}
        np.sum(scene > 0)

        for cls_id in RARE_CLASS_IDS[:8]:  # Dynamic objects
            current_count = np.sum(scene == cls_id)

            # Method 1: Compare to fixed target
            target = self.config.target_rare_count
            deficit_fixed = max(0, target - current_count) / target

            # Method 2: Compare to dataset average (if adaptive balance enabled)
            if self.config.adaptive_balance and cls_id in self.DATASET_AVERAGE_COUNTS:
                avg_count = self.DATASET_AVERAGE_COUNTS[cls_id]
                target_balanced = avg_count * self.config.balance_target_ratio
                deficit_adaptive = max(0, target_balanced - current_count) / max(1, target_balanced)
            else:
                deficit_adaptive = deficit_fixed

            # Combined deficit score (prioritize classes with zero presence)
            if current_count == 0:
                deficit_score = 2.0  # High priority for completely missing classes
            else:
                deficit_score = max(deficit_fixed, deficit_adaptive)

            deficits[cls_id] = deficit_score

        return deficits

    def enhance(self, scene: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Enhance scene to ensure ALL rare classes are present and balanced.

        Two-phase enhancement:
        Phase 1: ENSURE ALL CLASSES PRESENT - add objects until every rare class
                 has at least min_voxels_per_class voxels
        Phase 2: BALANCE - continue adding until we reach min_objects_to_add,
                 prioritizing underrepresented classes

        Returns:
            (enhanced_scene, stats_dict)
        """
        stats = {
            'objects_added': 0,
            'classes_enhanced': [],
            'class_counts_before': {},
            'class_counts_after': {},
            'voxels_added_per_class': {},
            'missing_classes_filled': 0,
            'all_classes_present': False,
        }

        if self.copy_paste is None:
            return scene, stats

        enhanced = scene.copy()

        # Record initial counts
        for cls_id in RARE_CLASS_IDS[:8]:
            stats['class_counts_before'][cls_id] = int(np.sum(scene == cls_id))

        total_objects_added = 0
        objects_per_class = {cls_id: 0 for cls_id in RARE_CLASS_IDS[:8]}
        max_attempts_per_class = 50  # Max attempts to place object for each class

        # ===== PHASE 1: Ensure ALL classes present =====
        if self.config.ensure_all_classes:
            for cls_id in RARE_CLASS_IDS[:8]:
                current_count = np.sum(enhanced == cls_id)

                # Keep adding until class has minimum voxels
                attempts = 0
                while (current_count < self.config.min_voxels_per_class and
                       attempts < max_attempts_per_class and
                       total_objects_added < self.config.max_objects_to_add):

                    enhanced_new, success = self.copy_paste.paste_object(enhanced, cls_id)
                    attempts += 1

                    if success:
                        enhanced = enhanced_new
                        total_objects_added += 1
                        objects_per_class[cls_id] += 1
                        current_count = np.sum(enhanced == cls_id)

                        if cls_id not in stats['classes_enhanced']:
                            stats['classes_enhanced'].append(cls_id)
                            stats['missing_classes_filled'] += 1

        # ===== PHASE 2: Balance to reach min_objects_to_add =====
        # Determine how many more objects to add
        target_objects = np.random.randint(
            self.config.min_objects_to_add,
            self.config.max_objects_to_add + 1
        )
        max(0, target_objects - total_objects_added)

        # Round-robin: add objects prioritizing classes with lowest counts
        max_rounds = 10
        for _round_num in range(max_rounds):
            if total_objects_added >= target_objects:
                break

            # Sort classes by current count (lowest first)
            class_counts = [(cls_id, np.sum(enhanced == cls_id)) for cls_id in RARE_CLASS_IDS[:8]]
            sorted_classes = sorted(class_counts, key=lambda x: x[1])

            added_this_round = False
            for cls_id, _count in sorted_classes:
                if total_objects_added >= target_objects:
                    break

                if objects_per_class[cls_id] >= self.config.max_objects_per_class:
                    continue

                # Try to add one object
                enhanced_new, success = self.copy_paste.paste_object(enhanced, cls_id)
                if success:
                    enhanced = enhanced_new
                    total_objects_added += 1
                    objects_per_class[cls_id] += 1
                    added_this_round = True

                    if cls_id not in stats['classes_enhanced']:
                        stats['classes_enhanced'].append(cls_id)

            # If no objects added in a round, stop
            if not added_this_round:
                break

        stats['objects_added'] = total_objects_added

        # Record final counts and check if all classes present
        all_present = True
        for cls_id in RARE_CLASS_IDS[:8]:
            final_count = int(np.sum(enhanced == cls_id))
            stats['class_counts_after'][cls_id] = final_count
            stats['voxels_added_per_class'][cls_id] = final_count - stats['class_counts_before'][cls_id]
            if final_count < self.config.min_voxels_per_class:
                all_present = False

        stats['all_classes_present'] = all_present

        return enhanced, stats


class InpaintingEnhancer:
    """
    Optional: Inpainting-style enhancement.

    This uses anchor points (road, sidewalk) to guide where objects
    should be placed, similar to RePaint inpainting.
    """

    def __init__(self, config: PostProcessConfig, object_bank_path: str = None):
        self.config = config
        self.object_bank = None

        if object_bank_path and config.enable_inpainting:
            try:
                self.object_bank = ObjectBank.load(object_bank_path)
            except Exception as e:
                logger.warning("Could not load object bank for inpainting: %s", e)

    def enhance(self, scene: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Apply inpainting-style enhancement.

        Strategy:
        1. Find anchor regions (road edges, sidewalks)
        2. Create "inpainting masks" for potential object locations
        3. Sample objects from bank and place at anchors
        """
        stats = {'anchor_points_found': 0, 'objects_inpainted': 0}

        if self.object_bank is None:
            return scene, stats

        enhanced = scene.copy()

        # Find road edges (good places for pedestrians, bikes)
        road_mask = scene == 9

        # Dilate to find edges
        struct = ndimage.generate_binary_structure(3, 1)
        road_dilated = ndimage.binary_dilation(road_mask, structure=struct)
        road_edge = road_dilated & ~road_mask & (scene == 0)

        # Find anchor points at road edges
        anchor_coords = np.argwhere(road_edge)
        stats['anchor_points_found'] = len(anchor_coords)

        if len(anchor_coords) == 0:
            return enhanced, stats

        # Sample a few anchor points and try to place objects
        num_placements = min(3, len(anchor_coords) // 100)

        for _ in range(num_placements):
            # Pick random anchor
            idx = np.random.randint(len(anchor_coords))
            ax, ay, az = anchor_coords[idx]

            # Pick random pedestrian/cyclist class
            cls_id = np.random.choice([6, 7, 2])  # person, bicyclist, bicycle

            # Get object from bank
            objects = self.object_bank.get_objects(class_id=cls_id, num_samples=1)
            if not objects:
                continue

            obj = objects[0]

            # Try to place at anchor
            px, py, pz = ax, ay, az + 1  # Just above anchor

            # Check bounds and validity
            valid = True
            for vx, vy, vz in obj.voxels:
                tx, ty, tz = int(vx + px), int(vy + py), int(vz + pz)
                if not (0 <= tx < 256 and 0 <= ty < 256 and 0 <= tz < 32):
                    valid = False
                    break
                if enhanced[tx, ty, tz] > 0:
                    valid = False
                    break

            if valid:
                for i, (vx, vy, vz) in enumerate(obj.voxels):
                    tx, ty, tz = int(vx + px), int(vy + py), int(vz + pz)
                    enhanced[tx, ty, tz] = obj.labels[i]
                stats['objects_inpainted'] += 1

        return enhanced, stats


class CascadePostProcessor:
    """
    Complete post-processing pipeline for cascaded generation.
    """

    def __init__(
        self,
        config: PostProcessConfig = None,
        object_bank_path: str = None,
    ):
        self.config = config or PostProcessConfig()

        self.quality_filter = QualityFilter(self.config)
        self.geometric_cleaner = GeometricCleaner(self.config)
        self.rare_enhancer = RareClassEnhancer(self.config, object_bank_path)
        self.inpainting_enhancer = InpaintingEnhancer(self.config, object_bank_path)

    def process(
        self,
        scene: np.ndarray,
        skip_quality_check: bool = False,
    ) -> tuple[np.ndarray | None, dict]:
        """
        Apply full post-processing pipeline.

        Args:
            scene: Generated scene (256, 256, 32)
            skip_quality_check: If True, don't reject based on quality

        Returns:
            (processed_scene or None if rejected, stats_dict)
        """
        all_stats = {}

        # Step 1: Quality filtering
        if not skip_quality_check:
            passed, filter_stats = self.quality_filter.check(scene)
            all_stats['quality_filter'] = filter_stats

            if not passed:
                return None, all_stats

        # Step 2: Geometric cleaning
        cleaned, clean_stats = self.geometric_cleaner.clean(scene)
        all_stats['geometric_clean'] = clean_stats

        # Step 3: Rare class enhancement
        if self.config.enable_enhancement:
            enhanced, enhance_stats = self.rare_enhancer.enhance(cleaned)
            all_stats['rare_enhancement'] = enhance_stats
        else:
            enhanced = cleaned

        # Step 4 (optional): Inpainting enhancement
        if self.config.enable_inpainting:
            enhanced, inpaint_stats = self.inpainting_enhancer.enhance(enhanced)
            all_stats['inpainting'] = inpaint_stats

        # Final stats
        all_stats['final'] = {
            'occupied_voxels': int(np.sum(enhanced > 0)),
            'num_classes': len(np.unique(enhanced)),
        }

        return enhanced, all_stats

    def process_batch(
        self,
        scenes: list[np.ndarray],
        skip_quality_check: bool = False,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Process multiple scenes.

        Returns:
            (list of processed scenes, aggregate stats)
        """
        processed = []
        rejected = 0
        total_stats = {
            'objects_added': 0,
            'voxels_removed': 0,
        }

        for scene in scenes:
            result, stats = self.process(scene, skip_quality_check)

            if result is not None:
                processed.append(result)
                if 'geometric_clean' in stats:
                    total_stats['voxels_removed'] += stats['geometric_clean'].get('voxels_removed', 0)
                if 'rare_enhancement' in stats:
                    total_stats['objects_added'] += stats['rare_enhancement'].get('objects_added', 0)
            else:
                rejected += 1

        total_stats['input_scenes'] = len(scenes)
        total_stats['output_scenes'] = len(processed)
        total_stats['rejected'] = rejected
        total_stats['accept_rate'] = len(processed) / len(scenes) if scenes else 0

        return processed, total_stats


def test_postprocessor():
    """Test the post-processing pipeline."""
    print("=== Testing Cascade Post-Processor ===\n")

    # Load a sample scene
    scene = np.load('datasets/SemanticKITTI_3D/256/00/000100_gt_scene.npy')
    print(f"Input scene: {scene.shape}, {np.sum(scene > 0)} occupied voxels")

    # Create post-processor
    config = PostProcessConfig(
        enable_enhancement=True,
        enable_inpainting=True,
        max_objects_to_add=3,
    )

    processor = CascadePostProcessor(
        config=config,
        object_bank_path='datasets/object_bank.pkl',
    )

    # Process
    result, stats = processor.process(scene, skip_quality_check=True)

    print("\n=== Results ===")
    print(f"Quality filter: {stats.get('quality_filter', 'skipped')}")
    print(f"Geometric cleaning: {stats.get('geometric_clean', {})}")
    print(f"Rare enhancement: {stats.get('rare_enhancement', {})}")
    print(f"Inpainting: {stats.get('inpainting', {})}")
    print(f"Final: {stats.get('final', {})}")

    if result is not None:
        print(f"\nOutput scene: {np.sum(result > 0)} occupied voxels")

        # Compare rare classes
        print("\nRare class comparison:")
        for cls_id in RARE_CLASS_IDS[:8]:
            orig = np.sum(scene == cls_id)
            final = np.sum(result == cls_id)
            if orig > 0 or final > 0:
                print(f"  {CLASS_NAMES[cls_id]}: {orig} -> {final} ({final-orig:+d})")


if __name__ == '__main__':
    test_postprocessor()
