"""
BEV Diffusion Training Pipeline

Supports running multiple experiments in parallel with different
hyperparameters and model configurations.
"""

import os
import sys
import json
import logging
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gssc.models.bev_diffusion_model import BEVDiffusionModel, create_bev_diffusion_model


# SemanticKITTI class names and frequencies (approximate)
SEMANTICKITTI_CLASSES = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
    'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
    'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'
]

# Approximate class frequencies (from SemanticKITTI statistics)
SEMANTICKITTI_FREQUENCIES = torch.tensor([
    0.40,   # unlabeled (empty)
    0.02,   # car
    0.001,  # bicycle
    0.001,  # motorcycle
    0.005,  # truck
    0.003,  # other-vehicle
    0.002,  # person
    0.001,  # bicyclist
    0.001,  # motorcyclist
    0.25,   # road
    0.02,   # parking
    0.05,   # sidewalk
    0.01,   # other-ground
    0.08,   # building
    0.02,   # fence
    0.10,   # vegetation
    0.01,   # trunk
    0.03,   # terrain
    0.005,  # pole
    0.002,  # traffic-sign
])


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    # Experiment identification
    name: str = "bev_diffusion_exp"
    description: str = ""

    # Model architecture
    model_config: str = "base"  # 'lightweight', 'base', 'large'
    num_classes: int = 20
    num_timesteps: int = 1000
    lidar_channels: int = 128
    unet_base_channels: int = 128
    use_self_conditioning: bool = True
    use_cross_attention: bool = True
    diffusion_type: str = "absorbing"  # 'absorbing' or 'uniform'
    schedule_type: str = "cosine"  # 'linear' or 'cosine'

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # Loss
    use_class_weights: bool = True
    aux_loss_weight: float = 0.001

    # Data
    data_root: str = "<repo>/datasets/dataset_SemanticKITTI_SSC"
    train_sequences: List[str] = field(default_factory=lambda: ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])
    val_sequences: List[str] = field(default_factory=lambda: ['08'])
    num_workers: int = 4
    use_rare_class_sampling: bool = True

    # Checkpointing
    save_every: int = 10
    output_dir: str = "<repo>/outputs/bev_experiments"

    # Logging
    log_every: int = 50

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperimentConfig':
        return cls(**d)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class SemanticKITTIBEVDataset(Dataset):
    """
    Dataset for BEV semantic segmentation from SemanticKITTI.

    Loads:
        - Sparse LiDAR voxels (.bin files)
        - Complete semantic labels (.label files)
        - Creates BEV labels by max-pooling along height
    """

    def __init__(
        self,
        data_root: str,
        sequences: List[str],
        split: str = 'train',
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.sequences = sequences
        self.split = split

        # Collect all samples
        self.samples = []
        for seq in sequences:
            seq_path = self.data_root / 'sequences' / seq
            voxel_path = seq_path / 'voxels'

            if not voxel_path.exists():
                logging.warning(f"Sequence {seq} not found at {voxel_path}")
                continue

            # Find all .bin files (sparse LiDAR)
            bin_files = sorted(voxel_path.glob('*.bin'))
            for bin_file in bin_files:
                frame_id = bin_file.stem
                label_file = voxel_path / f"{frame_id}.label"

                if label_file.exists():
                    self.samples.append({
                        'sequence': seq,
                        'frame_id': frame_id,
                        'bin_path': str(bin_file),
                        'label_path': str(label_file),
                    })

        logging.info(f"Found {len(self.samples)} samples for {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load sparse LiDAR voxels
        # .bin file contains packed binary mask
        with open(sample['bin_path'], 'rb') as f:
            bin_data = np.fromfile(f, dtype=np.uint8)

        # Unpack bits to voxel grid (256×256×32)
        voxels = np.unpackbits(bin_data).reshape(256, 256, 32).astype(np.float32)

        # Load complete semantic labels
        with open(sample['label_path'], 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint16).reshape(256, 256, 32)

        # Apply learning map to convert to 0-19 classes
        labels = self._apply_learning_map(labels)

        # Create BEV labels by taking the max (highest priority class) along height
        # Actually, we want the "most prominent" class at each (x,y) location
        # Use mode (most frequent) or max (highest class ID)
        bev_labels = self._create_bev_labels(labels)

        # Convert to tensors
        voxels = torch.from_numpy(voxels).unsqueeze(0)  # [1, X, Y, Z]
        bev_labels = torch.from_numpy(bev_labels).long()  # [H, W]

        return {
            'voxels': voxels,
            'bev_labels': bev_labels,
            'sequence': sample['sequence'],
            'frame_id': sample['frame_id'],
        }

    def _apply_learning_map(self, labels: np.ndarray) -> np.ndarray:
        """Apply SemanticKITTI learning map to convert labels to 0-19."""
        # Learning map from SemanticKITTI
        learning_map = {
            0: 0,    # unlabeled
            1: 0,    # outlier
            10: 1,   # car
            11: 2,   # bicycle
            13: 5,   # bus
            15: 3,   # motorcycle
            16: 5,   # on-rails
            18: 4,   # truck
            20: 5,   # other-vehicle
            30: 6,   # person
            31: 7,   # bicyclist
            32: 8,   # motorcyclist
            40: 9,   # road
            44: 10,  # parking
            48: 11,  # sidewalk
            49: 12,  # other-ground
            50: 13,  # building
            51: 14,  # fence
            52: 0,   # other-structure
            60: 9,   # lane-marking
            70: 15,  # vegetation
            71: 16,  # trunk
            72: 17,  # terrain
            80: 18,  # pole
            81: 19,  # traffic-sign
            99: 0,   # other-object
            252: 1,  # moving-car
            253: 7,  # moving-bicyclist
            254: 6,  # moving-person
            255: 8,  # moving-motorcyclist
            256: 5,  # moving-other-vehicle
            257: 5,  # moving-bus
            258: 4,  # moving-truck
            259: 5,  # moving-other-vehicle
        }

        mapped = np.zeros_like(labels)
        for orig, target in learning_map.items():
            mapped[labels == orig] = target

        return mapped

    def _create_bev_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Create BEV labels from 3D labels.

        Strategy: For each (x, y), take the non-zero class that appears most frequently
        along the z-axis. If all zeros, the result is 0 (unlabeled).
        """
        H, W, Z = labels.shape
        bev = np.zeros((H, W), dtype=np.int64)

        for x in range(H):
            for y in range(W):
                column = labels[x, y, :]
                # Get non-zero classes
                non_zero = column[column > 0]
                if len(non_zero) > 0:
                    # Take mode (most frequent)
                    values, counts = np.unique(non_zero, return_counts=True)
                    bev[x, y] = values[counts.argmax()]

        return bev

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights based on inverse frequency."""
        return 1.0 / (SEMANTICKITTI_FREQUENCIES + 1e-6)

    def get_sample_weights(self) -> np.ndarray:
        """
        Compute per-sample weights for rare class sampling.

        Samples containing rare classes get higher weights.
        """
        weights = np.ones(len(self.samples))

        # For efficiency, we use pre-computed approximations
        # In practice, you'd compute this from actual data statistics
        rare_classes = [2, 3, 6, 7, 8, 18, 19]  # bicycle, motorcycle, person, etc.

        # This is a simplified version - in practice, you'd check each sample
        # for rare class presence
        return weights


class BEVTrainer:
    """
    Trainer for BEV diffusion model.

    Supports:
        - Single experiment training
        - Checkpoint saving/loading
        - Logging and metrics
        - Validation
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device = None,
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup logging
        self._setup_logging()

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config.save(str(self.output_dir / 'config.json'))

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Build optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Build scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )

        # Build datasets
        self.train_dataset, self.val_dataset = self._build_datasets()
        self.train_loader = self._build_dataloader(self.train_dataset, shuffle=True)
        self.val_loader = self._build_dataloader(self.val_dataset, shuffle=False)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        self.logger.info(f"Model parameters: {self.model.get_num_params()}")

    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.config.name)

    def _build_model(self) -> BEVDiffusionModel:
        """Build the BEV diffusion model."""
        # Compute class weights if needed
        class_weights = None
        if self.config.use_class_weights:
            class_weights = 1.0 / (SEMANTICKITTI_FREQUENCIES + 1e-6)
            class_weights = class_weights / class_weights.sum() * len(class_weights)

        return create_bev_diffusion_model(
            config=self.config.model_config,
            num_classes=self.config.num_classes,
            num_timesteps=self.config.num_timesteps,
            lidar_channels=self.config.lidar_channels,
            unet_base_channels=self.config.unet_base_channels,
            use_self_conditioning=self.config.use_self_conditioning,
            use_cross_attention=self.config.use_cross_attention,
            diffusion_type=self.config.diffusion_type,
            schedule_type=self.config.schedule_type,
            aux_loss_weight=self.config.aux_loss_weight,
            class_weights=class_weights,
        )

    def _build_datasets(self) -> Tuple[Dataset, Dataset]:
        """Build train and validation datasets."""
        train_dataset = SemanticKITTIBEVDataset(
            data_root=self.config.data_root,
            sequences=self.config.train_sequences,
            split='train',
        )

        val_dataset = SemanticKITTIBEVDataset(
            data_root=self.config.data_root,
            sequences=self.config.val_sequences,
            split='val',
        )

        return train_dataset, val_dataset

    def _build_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Build dataloader with optional rare class sampling."""
        sampler = None

        if shuffle and self.config.use_rare_class_sampling:
            weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(weights, len(weights))
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_ce_loss = 0
        total_aux_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            voxels = batch['voxels'].to(self.device)
            bev_labels = batch['bev_labels'].to(self.device)

            # Forward pass
            losses = self.model.compute_loss(voxels, bev_labels)

            # Backward pass
            self.optimizer.zero_grad()
            losses['loss'].backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_aux_loss += losses['aux_loss'].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })

            # Log periodically
            if self.global_step % self.config.log_every == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss={losses['loss'].item():.4f}, "
                    f"ce_loss={losses['ce_loss'].item():.4f}, "
                    f"aux_loss={losses['aux_loss'].item():.4f}"
                )

        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_pixels = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            voxels = batch['voxels'].to(self.device)
            bev_labels = batch['bev_labels'].to(self.device)

            # Compute loss
            losses = self.model.compute_loss(voxels, bev_labels)
            total_loss += losses['loss'].item()

            # Sample predictions for accuracy
            if num_batches < 10:  # Only sample a few batches for speed
                pred = self.model.sample(voxels, num_steps=50)
                total_correct += (pred == bev_labels).sum().item()
                total_pixels += bev_labels.numel()

            num_batches += 1

        accuracy = total_correct / total_pixels if total_pixels > 0 else 0

        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': accuracy,
        }

    def save_checkpoint(self, filename: str = None):
        """Save checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.epoch}.pt"

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Loaded checkpoint from {path}, epoch {self.epoch}")

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
            )

            # Validate
            if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                val_metrics = self.validate()
                self.logger.info(
                    f"Epoch {epoch}: val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_acc={val_metrics['val_accuracy']:.4f}"
                )

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best_model.pt')

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint()

            # Update scheduler
            self.scheduler.step()

        self.logger.info("Training complete!")


def run_experiment(config: ExperimentConfig, device: torch.device = None):
    """Run a single experiment."""
    trainer = BEVTrainer(config, device)
    trainer.train()


def create_experiment_configs() -> List[ExperimentConfig]:
    """
    Create multiple experiment configurations for parallel training.

    Returns list of configs with different hyperparameters.
    """
    configs = []

    # Experiment 1: Baseline (absorbing D3PM, cosine schedule)
    configs.append(ExperimentConfig(
        name="exp1_absorbing_cosine_base",
        description="Baseline: absorbing D3PM with cosine schedule",
        model_config="base",
        diffusion_type="absorbing",
        schedule_type="cosine",
        use_self_conditioning=True,
    ))

    # Experiment 2: Uniform D3PM
    configs.append(ExperimentConfig(
        name="exp2_uniform_cosine_base",
        description="Uniform D3PM with cosine schedule",
        model_config="base",
        diffusion_type="uniform",
        schedule_type="cosine",
        use_self_conditioning=True,
    ))

    # Experiment 3: Without self-conditioning
    configs.append(ExperimentConfig(
        name="exp3_absorbing_no_selfcond",
        description="Absorbing D3PM without self-conditioning",
        model_config="base",
        diffusion_type="absorbing",
        use_self_conditioning=False,
    ))

    # Experiment 4: Lightweight model (faster iteration)
    configs.append(ExperimentConfig(
        name="exp4_lightweight",
        description="Lightweight model for fast iteration",
        model_config="lightweight",
        diffusion_type="absorbing",
        num_timesteps=500,
        batch_size=8,
    ))

    # Experiment 5: Larger model
    configs.append(ExperimentConfig(
        name="exp5_large_model",
        description="Larger model with more capacity",
        model_config="large",
        diffusion_type="absorbing",
        lidar_channels=256,
        unet_base_channels=256,
        batch_size=2,
    ))

    # Experiment 6: Linear schedule
    configs.append(ExperimentConfig(
        name="exp6_linear_schedule",
        description="Linear noise schedule instead of cosine",
        model_config="base",
        diffusion_type="absorbing",
        schedule_type="linear",
    ))

    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEV Diffusion Training")
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--experiment', type=int, default=None, help='Experiment index (0-5)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.experiment is not None:
        configs = create_experiment_configs()
        config = configs[args.experiment]
    else:
        # Default config
        config = ExperimentConfig()

    run_experiment(config, device)
