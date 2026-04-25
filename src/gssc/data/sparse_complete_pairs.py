"""
Sparse-Complete Pair Generation using LiDAR Ray-Tracing Simulation

This module provides the complete pipeline for creating synthetic (sparse, complete)
training pairs from generated complete 3D semantic scenes using realistic LiDAR
ray-tracing simulation.

Pipeline:
    1. Generated complete scene (256×256×32 semantic labels) from S1→S2→S3
    2. LiDAR ray-tracing → sparse observation mask (binary visibility)
    3. Apply semantic labels from complete scene to visible voxels
    4. Save (sparse, complete) pairs in SemanticKITTI-compatible format

Key Features:
    - MultiReturn LiDAR simulation (best accuracy, ~0.27 IoU with GT)
    - Realistic Velodyne HDL-64E calibration data
    - Beam divergence and noise modeling
    - Batch processing with progress tracking
    - SemanticKITTI-compatible output format (.bin, .label)

Usage:
    from gssc.data.sparse_complete_pairs import SparseCompletePairGenerator

    generator = SparseCompletePairGenerator()

    # Single scene
    sparse, complete = generator.generate_pair(complete_scene)

    # Batch processing
    generator.process_batch(
        complete_scenes,
        output_dir="datasets/synthetic_pairs",
    )

Author: SSC Project
Date: December 2024
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import logging
import json

# Import LiDAR simulation modules
from gssc.data.lidar_simulator_v2 import (
    MultiReturnLiDARSimulator,
    create_multi_return_resampler,
    DensityAwareLiDARSimulator,
    create_density_aware_resampler,
    VELODYNE_HDL64E_VERTICAL_ANGLES,
    pack_voxels,
    unpack_voxels,
)

logger = logging.getLogger(__name__)


@dataclass
class PairGenerationConfig:
    """Configuration for sparse-complete pair generation."""

    # LiDAR simulation method
    # 'multi_return': Best accuracy (~0.27 IoU with GT)
    # 'density_aware': Standard ray-tracing
    simulation_method: str = 'multi_return'

    # MultiReturn LiDAR parameters
    max_returns: int = 2
    samples_per_return: int = 2
    range_noise_std: float = 0.03  # meters
    angular_noise_std: float = 0.002  # radians

    # Horizontal resolution (points per 180° sweep)
    h_resolution: int = 2048

    # Output format
    output_format: str = 'semantickitti'  # 'semantickitti' or 'npz'
    save_point_cloud: bool = False  # Also save raw point clouds

    # Processing
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.simulation_method in ['multi_return', 'density_aware'], \
            f"Unknown simulation method: {self.simulation_method}"
        assert self.output_format in ['semantickitti', 'npz'], \
            f"Unknown output format: {self.output_format}"


class SparseCompletePairGenerator:
    """
    Generate synthetic (sparse, complete) pairs using LiDAR ray-tracing.

    This class provides the complete pipeline for creating training pairs:
    1. Take a complete 3D semantic scene (256×256×32)
    2. Simulate LiDAR observation using ray-tracing
    3. Create sparse observation with semantic labels
    4. Return (sparse, complete) pair for training

    The sparse observation represents what a Velodyne HDL-64E LiDAR would
    observe from the ego-vehicle position, with realistic beam patterns,
    noise, and occlusion handling.
    """

    def __init__(self, config: PairGenerationConfig = None):
        """
        Initialize the pair generator.

        Args:
            config: Generation configuration (uses defaults if None)
        """
        self.config = config or PairGenerationConfig()
        self._init_simulator()

    def _init_simulator(self):
        """Initialize the appropriate LiDAR simulator."""
        if self.config.simulation_method == 'multi_return':
            self.simulator = MultiReturnLiDARSimulator(
                vertical_angles=VELODYNE_HDL64E_VERTICAL_ANGLES,
                h_resolution=self.config.h_resolution,
                max_returns=self.config.max_returns,
                samples_per_return=self.config.samples_per_return,
                range_noise_std=self.config.range_noise_std,
                angular_noise_std=self.config.angular_noise_std,
            )
            logger.info("Initialized MultiReturnLiDARSimulator")
        else:
            self.simulator = create_density_aware_resampler()
            logger.info("Initialized DensityAwareLiDARSimulator")

    def generate_pair(
        self,
        complete_scene: np.ndarray,
        return_point_cloud: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate a sparse-complete pair from a complete scene.

        Args:
            complete_scene: (256, 256, 32) semantic label array (uint8/uint16)
            return_point_cloud: Also return the raw point cloud

        Returns:
            sparse_labels: (256, 256, 32) sparse semantic labels (0 = unobserved/empty)
            complete_labels: (256, 256, 32) complete semantic labels (same as input)
            [optional] point_cloud: (N, 3) world coordinates of observed points

        Notes:
            - sparse_labels contains semantic labels ONLY for visible voxels
            - Invisible/occluded voxels are marked as 0 (unlabeled)
            - complete_labels is the input scene (returned for convenience)
        """
        assert complete_scene.shape == (256, 256, 32), \
            f"Expected (256, 256, 32), got {complete_scene.shape}"

        # Ensure correct dtype
        complete = complete_scene.astype(np.uint8)

        # Simulate LiDAR observation
        if self.config.simulation_method == 'multi_return':
            sparse_mask, point_cloud, stats = self.simulator.simulate(
                complete,
                verbose=False
            )
        else:
            sparse_mask, stats = self.simulator.simulate(
                complete,
                verbose=False
            )
            point_cloud = None

        # Apply semantic labels to visible voxels
        # sparse_labels = labels where sparse_mask=1, else 0
        sparse_labels = complete * sparse_mask

        if return_point_cloud and point_cloud is not None:
            return sparse_labels, complete, point_cloud
        else:
            return sparse_labels, complete

    def generate_binary_mask(
        self,
        complete_scene: np.ndarray,
    ) -> np.ndarray:
        """
        Generate only the binary observation mask (for compatibility with .bin format).

        Args:
            complete_scene: (256, 256, 32) semantic labels

        Returns:
            sparse_mask: (256, 256, 32) binary observation mask (uint8)
        """
        assert complete_scene.shape == (256, 256, 32)
        complete = complete_scene.astype(np.uint8)

        if self.config.simulation_method == 'multi_return':
            sparse_mask, _, _ = self.simulator.simulate(complete, verbose=False)
        else:
            sparse_mask, _ = self.simulator.simulate(complete, verbose=False)

        return sparse_mask.astype(np.uint8)

    def process_batch(
        self,
        complete_scenes: List[np.ndarray],
        output_dir: str,
        start_index: int = 0,
        sequence_id: str = "synthetic",
    ) -> Dict[str, any]:
        """
        Process a batch of complete scenes and save pairs to disk.

        Args:
            complete_scenes: List of (256, 256, 32) complete scenes
            output_dir: Directory to save output files
            start_index: Starting frame index for naming
            sequence_id: Sequence identifier for file naming

        Returns:
            stats: Dictionary with processing statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for SemanticKITTI format
        voxels_dir = output_path / "sequences" / sequence_id / "voxels"
        voxels_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'num_scenes': len(complete_scenes),
            'total_observed': 0,
            'total_complete': 0,
            'observation_rate': 0.0,
        }

        iterator = tqdm(
            enumerate(complete_scenes, start=start_index),
            total=len(complete_scenes),
            desc="Generating pairs",
            disable=not self.config.verbose,
        )

        for idx, complete in iterator:
            # Generate pair
            sparse_labels, complete_labels = self.generate_pair(complete)
            sparse_mask = (sparse_labels > 0).astype(np.uint8)

            # Update stats
            stats['total_observed'] += sparse_mask.sum()
            stats['total_complete'] += (complete > 0).sum()

            # Save in appropriate format
            frame_id = f"{idx:06d}"

            if self.config.output_format == 'semantickitti':
                # Save sparse observation (.bin) - bit-packed binary mask
                bin_path = voxels_dir / f"{frame_id}.bin"
                packed = pack_voxels(sparse_mask)
                packed.tofile(bin_path)

                # Save complete labels (.label) - uint16 labels
                label_path = voxels_dir / f"{frame_id}.label"
                complete_labels.astype(np.uint16).tofile(label_path)

                # Save invalid mask (.invalid) - occluded/beyond LoS occupied voxels
                invalid_mask = ((complete > 0) & (sparse_mask == 0)).astype(np.uint8)
                invalid_path = voxels_dir / f"{frame_id}.invalid"
                pack_voxels(invalid_mask).tofile(invalid_path)

            else:  # npz format
                npz_path = output_path / f"{frame_id}.npz"
                np.savez_compressed(
                    npz_path,
                    sparse=sparse_labels,
                    complete=complete_labels,
                    mask=sparse_mask,
                )

            # Update progress bar
            if self.config.verbose:
                obs_rate = 100 * stats['total_observed'] / max(stats['total_complete'], 1)
                iterator.set_postfix({'obs_rate': f'{obs_rate:.1f}%'})

        # Final stats
        stats['observation_rate'] = stats['total_observed'] / max(stats['total_complete'], 1)

        # Save stats
        stats_path = output_path / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Processed {stats['num_scenes']} scenes")
        logger.info(f"Average observation rate: {100*stats['observation_rate']:.1f}%")

        return stats

    def process_npz_directory(
        self,
        input_dir: str,
        output_dir: str,
        scene_key: str = 'scenes',
    ) -> Dict[str, any]:
        """
        Process complete scenes from NPZ files in a directory.

        Args:
            input_dir: Directory containing NPZ files with complete scenes
            output_dir: Directory to save output pairs
            scene_key: Key in NPZ files containing scene data

        Returns:
            stats: Processing statistics
        """
        input_path = Path(input_dir)
        npz_files = sorted(input_path.glob("*.npz"))

        if not npz_files:
            raise ValueError(f"No NPZ files found in {input_dir}")

        logger.info(f"Found {len(npz_files)} NPZ files to process")

        all_scenes = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            if scene_key in data:
                scenes = data[scene_key]
                if scenes.ndim == 4:  # Multiple scenes: [N, 256, 256, 32]
                    all_scenes.extend([s for s in scenes])
                else:  # Single scene: [256, 256, 32]
                    all_scenes.append(scenes)

        logger.info(f"Loaded {len(all_scenes)} complete scenes")

        return self.process_batch(all_scenes, output_dir)


class SparseCompleteDataset:
    """
    Dataset class for loading generated (sparse, complete) pairs.

    Compatible with PyTorch DataLoader.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_id: str = "synthetic",
        transform=None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing sequences
            sequence_id: Which sequence to load
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir)
        self.sequence_id = sequence_id
        self.transform = transform

        # Find all frames
        self.voxels_dir = self.data_dir / "sequences" / sequence_id / "voxels"
        self.frame_ids = sorted([
            f.stem for f in self.voxels_dir.glob("*.label")
        ])

        logger.info(f"Found {len(self.frame_ids)} frames in {self.voxels_dir}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a sparse-complete pair.

        Returns:
            sparse_mask: (256, 256, 32) binary observation mask
            complete_labels: (256, 256, 32) semantic labels
        """
        frame_id = self.frame_ids[idx]

        # Load sparse mask
        bin_path = self.voxels_dir / f"{frame_id}.bin"
        sparse_packed = np.fromfile(bin_path, dtype=np.uint8)
        sparse_mask = unpack_voxels(sparse_packed)

        # Load complete labels
        label_path = self.voxels_dir / f"{frame_id}.label"
        complete_labels = np.fromfile(label_path, dtype=np.uint16).reshape(256, 256, 32)

        if self.transform:
            sparse_mask, complete_labels = self.transform(sparse_mask, complete_labels)

        return sparse_mask, complete_labels

    def load_with_invalid(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load pair with invalid mask."""
        frame_id = self.frame_ids[idx]

        sparse_mask, complete_labels = self[idx]

        # Load invalid mask
        invalid_path = self.voxels_dir / f"{frame_id}.invalid"
        if invalid_path.exists():
            invalid_packed = np.fromfile(invalid_path, dtype=np.uint8)
            invalid_mask = unpack_voxels(invalid_packed)
        else:
            # Compute from sparse and complete
            invalid_mask = ((complete_labels > 0) & (sparse_mask == 0)).astype(np.uint8)

        return sparse_mask, complete_labels, invalid_mask


def create_pair_generator(
    method: str = 'multi_return',
    max_returns: int = 2,
    samples_per_return: int = 2,
) -> SparseCompletePairGenerator:
    """
    Factory function to create a pair generator with common configurations.

    Args:
        method: 'multi_return' (best) or 'density_aware'
        max_returns: For multi_return method
        samples_per_return: For multi_return method

    Returns:
        Configured SparseCompletePairGenerator
    """
    config = PairGenerationConfig(
        simulation_method=method,
        max_returns=max_returns,
        samples_per_return=samples_per_return,
    )
    return SparseCompletePairGenerator(config)


def generate_pair_from_complete(
    complete_scene: np.ndarray,
    method: str = 'multi_return',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate a single pair.

    Args:
        complete_scene: (256, 256, 32) complete semantic scene
        method: LiDAR simulation method

    Returns:
        sparse_labels: (256, 256, 32) sparse observed labels
        complete_labels: (256, 256, 32) complete labels
    """
    generator = create_pair_generator(method=method)
    return generator.generate_pair(complete_scene)


# =============================================================================
# Integration with Synthetic Generator
# =============================================================================

class SyntheticPairPipeline:
    """
    Complete pipeline for generating synthetic (sparse, complete) pairs
    from the S1→S2→S3 pyramid diffusion model.

    This integrates:
    1. SyntheticSceneGenerator - generates complete scenes with rare class balancing
    2. SparseCompletePairGenerator - creates sparse observations via ray-tracing
    """

    def __init__(
        self,
        scene_generator=None,  # SyntheticSceneGenerator
        pair_config: PairGenerationConfig = None,
    ):
        """
        Initialize the pipeline.

        Args:
            scene_generator: Initialized SyntheticSceneGenerator (or None to create later)
            pair_config: Configuration for pair generation
        """
        self.scene_generator = scene_generator
        self.pair_generator = SparseCompletePairGenerator(
            pair_config or PairGenerationConfig()
        )

        self.stats = {
            'scenes_generated': 0,
            'pairs_created': 0,
            'class_counts': {i: 0 for i in range(20)},
            'observation_rates': [],
        }

    def generate_and_pair(
        self,
        num_scenes: int = 100,
        output_dir: str = "datasets/synthetic_pairs",
        strategy: str = 'auto',
    ) -> Dict[str, any]:
        """
        Generate complete scenes and their sparse observations.

        Args:
            num_scenes: Number of scene pairs to generate
            output_dir: Output directory
            strategy: Generation strategy ('auto', 'standard', 'copy_paste', 'inpaint')

        Returns:
            stats: Generation and pair creation statistics
        """
        if self.scene_generator is None:
            raise ValueError("Scene generator not initialized")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create output directories
        voxels_dir = output_path / "sequences" / "synthetic" / "voxels"
        voxels_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_scenes} synthetic pairs...")

        for i in tqdm(range(num_scenes), desc="Generating pairs"):
            # Generate complete scene
            complete = self.scene_generator.generate_single_scene(strategy=strategy)
            self.stats['scenes_generated'] += 1

            # Generate sparse observation
            sparse_labels, complete_labels = self.pair_generator.generate_pair(complete)
            sparse_mask = (sparse_labels > 0).astype(np.uint8)

            # Update class statistics
            for cls in range(20):
                self.stats['class_counts'][cls] += (complete == cls).sum()

            # Track observation rate
            occupied = (complete > 0).sum()
            observed = sparse_mask.sum()
            if occupied > 0:
                self.stats['observation_rates'].append(observed / occupied)

            # Save pair
            frame_id = f"{i:06d}"
            pack_voxels(sparse_mask).tofile(voxels_dir / f"{frame_id}.bin")
            complete_labels.astype(np.uint16).tofile(voxels_dir / f"{frame_id}.label")

            invalid = ((complete > 0) & (sparse_mask == 0)).astype(np.uint8)
            pack_voxels(invalid).tofile(voxels_dir / f"{frame_id}.invalid")

            self.stats['pairs_created'] += 1

        # Compute final statistics
        self.stats['avg_observation_rate'] = np.mean(self.stats['observation_rates'])

        # Save statistics
        stats_path = output_path / "pipeline_stats.json"
        stats_copy = {k: v for k, v in self.stats.items() if k != 'observation_rates'}
        stats_copy['avg_observation_rate'] = float(self.stats['avg_observation_rate'])
        with open(stats_path, 'w') as f:
            json.dump(stats_copy, f, indent=2)

        logger.info(f"Generated {self.stats['pairs_created']} pairs")
        logger.info(f"Average observation rate: {100*self.stats['avg_observation_rate']:.1f}%")

        return self.stats


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sparse-Complete Pair Generator - Test")
    print("=" * 60)

    # Create test complete scene
    print("\n1. Creating test complete scene...")
    complete_scene = np.zeros((256, 256, 32), dtype=np.uint8)

    # Ground plane (road class = 9)
    complete_scene[:, :, 8:10] = 9

    # Building (class = 13)
    complete_scene[50:80, 100:160, 10:25] = 13

    # Car (class = 1)
    complete_scene[100:110, 120:130, 10:12] = 1

    # Vegetation (class = 15)
    complete_scene[150:180, 50:90, 10:18] = 15

    print(f"   Complete scene: {complete_scene.shape}")
    print(f"   Occupied voxels: {(complete_scene > 0).sum():,}")

    # Test pair generation
    print("\n2. Testing pair generation...")
    generator = create_pair_generator(
        method='multi_return',
        max_returns=2,
        samples_per_return=2,
    )

    sparse_labels, complete_labels = generator.generate_pair(complete_scene)
    sparse_mask = (sparse_labels > 0).astype(np.uint8)

    print(f"   Sparse labels: {sparse_labels.shape}")
    print(f"   Observed voxels: {sparse_mask.sum():,}")
    print(f"   Observation rate: {100 * sparse_mask.sum() / (complete_scene > 0).sum():.1f}%")

    # Test class preservation
    print("\n3. Checking class preservation...")
    for cls in [9, 13, 1, 15]:
        complete_count = (complete_scene == cls).sum()
        sparse_count = (sparse_labels == cls).sum()
        print(f"   Class {cls}: {complete_count:,} complete → {sparse_count:,} visible")

    # Test batch processing
    print("\n4. Testing batch processing...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a few test scenes
        test_scenes = [complete_scene.copy() for _ in range(3)]

        stats = generator.process_batch(
            test_scenes,
            output_dir=tmpdir,
            sequence_id="test",
        )

        print(f"   Processed {stats['num_scenes']} scenes")
        print(f"   Output files created in {tmpdir}")

        # Verify files
        voxels_dir = Path(tmpdir) / "sequences" / "test" / "voxels"
        bin_files = list(voxels_dir.glob("*.bin"))
        label_files = list(voxels_dir.glob("*.label"))
        print(f"   .bin files: {len(bin_files)}")
        print(f"   .label files: {len(label_files)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
