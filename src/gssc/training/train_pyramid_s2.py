#!/usr/bin/env python3
"""
S2 (64x64x8) Pyramid Diffusion Training for SemanticKITTI.

S2 is trained conditioned on upsampled S1 predictions.
Supports multi-GPU training via DistributedDataParallel.

Usage:
    # Single GPU
    python train_s2.py --gpu 0

    # Multi-GPU (use torchrun)
    torchrun --nproc_per_node=4 train_s2.py

    # Resume from checkpoint
    python train_s2.py --resume outputs/checkpoints/s2/latest.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'data_augmentation'))

from pyramid_diffusion import PyramidDiscreteDiffusion
from pyramid_unet import Denoise


class S2Dataset(Dataset):
    """
    Dataset for S2 training with S1 conditioning.

    Loads:
    - S2 ground truth labels (64x64x8)
    - S1 labels for conditioning (32x32x4, upsampled to 64x64x8)
    """

    def __init__(
        self,
        quantized_root: str,
        sequences: list[int],
        conditioning_mode: str = 'ground_truth',
        augment: bool = True,
    ):
        self.quantized_root = Path(quantized_root)
        self.conditioning_mode = conditioning_mode
        self.augment = augment

        self.s1_path = self.quantized_root / 's1' / 'sequences'
        self.s2_path = self.quantized_root / 's2' / 'sequences'

        self.frames = []
        for seq_id in sequences:
            seq_s1 = self.s1_path / f'{seq_id:02d}'
            seq_s2 = self.s2_path / f'{seq_id:02d}'

            if not seq_s1.exists() or not seq_s2.exists():
                continue

            s1_frames = set(f.stem for f in seq_s1.glob('*.npy'))
            s2_frames = set(f.stem for f in seq_s2.glob('*.npy'))

            common_frames = sorted(s1_frames & s2_frames)
            for frame_id in common_frames:
                self.frames.append((seq_id, frame_id))

        print(f"[S2Dataset] Loaded {len(self.frames)} frames from {len(sequences)} sequences")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict:
        seq_id, frame_id = self.frames[idx]

        s2_path = self.s2_path / f'{seq_id:02d}' / f'{frame_id}.npy'
        s2_labels = np.load(s2_path).astype(np.int64)

        s1_path = self.s1_path / f'{seq_id:02d}' / f'{frame_id}.npy'
        s1_labels = np.load(s1_path).astype(np.int64)

        if self.augment and np.random.random() > 0.5:
            s2_labels = np.flip(s2_labels, axis=0).copy()
            s1_labels = np.flip(s1_labels, axis=0).copy()

        if self.augment and np.random.random() > 0.5:
            s2_labels = np.flip(s2_labels, axis=1).copy()
            s1_labels = np.flip(s1_labels, axis=1).copy()

        if self.augment:
            k = np.random.randint(4)
            if k > 0:
                s2_labels = np.rot90(s2_labels, k, axes=(0, 1)).copy()
                s1_labels = np.rot90(s1_labels, k, axes=(0, 1)).copy()

        return {
            's2_labels': torch.from_numpy(s2_labels).long(),
            's1_labels': torch.from_numpy(s1_labels).long(),
            'sequence': seq_id,
            'frame_id': frame_id,
        }


def upsample_s1_to_s2(s1_labels: torch.Tensor, num_classes: int = 20) -> torch.Tensor:
    """
    Upsample S1 (32x32x4) to S2 (64x64x8) using trilinear interpolation.
    Matches original pyramid-discrete-diffusion exactly.
    """
    one_hot = F.one_hot(s1_labels, num_classes=num_classes).float()
    one_hot = one_hot.permute(0, 4, 1, 2, 3)
    interpolated = F.interpolate(one_hot, size=(64, 64, 8), mode='trilinear', align_corners=False)
    upsampled_labels = interpolated.argmax(dim=1)
    return upsampled_labels.unsqueeze(1).byte()


def setup_distributed():
    """Setup distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process."""
    return rank == 0


class S2Trainer:
    """Trainer for S2 pyramid diffusion with multi-GPU support."""

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        learning_rate: float = 0.004,
        device: torch.device = None,
        output_dir: str = 'outputs/checkpoints/s2',
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        distributed: bool = False,
        scale_lr: bool = True,
        warmup_epochs: int = 5,
        num_epochs: int = 10000,
    ):
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.distributed = distributed
        self.scale_lr = scale_lr
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.device = device or torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)

        if is_main_process(rank):
            self.output_dir.mkdir(parents=True, exist_ok=True)

        class Args:
            next_stage = 's_2'
            prev_stage = 's_1'

        self.denoiser = Denoise(
            args=Args(),
            num_class=num_classes,
            init_size=32,
            discrete=True
        ).to(self.device)

        self.diffusion = PyramidDiscreteDiffusion(
            args=None,
            denoise_model=self.denoiser,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            multi_criterion=None,
            auxiliary_loss_weight=0.0005,
            adaptive_auxiliary_loss=True,
            recon_loss=False,
        ).to(self.device)

        if distributed:
            self.diffusion = DDP(self.diffusion, device_ids=[local_rank], output_device=local_rank)

        # Scale learning rate based on global batch size
        # Linear scaling rule: lr_scaled = lr_base * world_size
        self.base_lr = learning_rate
        if scale_lr and world_size > 1:
            self.scaled_lr = learning_rate * world_size
        else:
            self.scaled_lr = learning_rate

        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=self.scaled_lr,  # Start with scaled LR (warmup will handle gradual increase)
            betas=(0.9, 0.999),
        )

        # Create scheduler with warmup
        self.scheduler = self._create_scheduler_with_warmup()

        if is_main_process(rank):
            total_params = sum(p.numel() for p in self.diffusion.parameters())
            print(f"[S2Trainer] Total parameters: {total_params:,}")
            print(f"[S2Trainer] Device: {self.device}")
            print(f"[S2Trainer] Distributed: {distributed}, World size: {world_size}")
            print(f"[S2Trainer] Base LR: {self.base_lr}, Scaled LR: {self.scaled_lr}")
            if scale_lr and world_size > 1:
                print(f"[S2Trainer] LR scaling enabled: {self.base_lr} × {world_size} = {self.scaled_lr}")
                print(f"[S2Trainer] Warmup epochs: {warmup_epochs}")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 1

    def _create_scheduler_with_warmup(self):
        """Create learning rate scheduler with linear warmup."""
        # If using LR scaling with warmup, start from base_lr and warm up to scaled_lr
        if self.scale_lr and self.world_size > 1 and self.warmup_epochs > 0:
            # Warmup: linearly increase from base_lr to scaled_lr
            def warmup_lambda(epoch):
                if epoch < self.warmup_epochs:
                    # Linear warmup: start at 1/world_size, end at 1.0
                    return (1.0 / self.world_size) + (1.0 - 1.0 / self.world_size) * (epoch / self.warmup_epochs)
                return 1.0

            warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)

            # After warmup, use cosine annealing
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - self.warmup_epochs,
                eta_min=1e-6,
            )

            # Combine warmup + cosine
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs]
            )
        else:
            # No warmup needed, just cosine annealing
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6,
            )

    def train_epoch(self, dataloader: DataLoader, epoch: int, log_every: int = 100) -> float:
        """Train one epoch."""
        self.diffusion.train()
        total_loss = 0.0

        if self.distributed:
            dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False, disable=not is_main_process(self.rank))
        for batch in pbar:
            s2_labels = batch['s2_labels'].to(self.device)
            s1_labels = batch['s1_labels'].to(self.device)

            cond = upsample_s1_to_s2(s1_labels, self.num_classes).to(self.device)

            if self.distributed:
                loss = self.diffusion.module(s2_labels, cond=cond)
            else:
                loss = self.diffusion(s2_labels, cond=cond)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if is_main_process(self.rank) and self.global_step % log_every == 0:
                print(f'Step {self.global_step}: loss={loss.item():.4f}')

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate on held-out data."""
        self.diffusion.eval()
        total_loss = 0.0

        for batch in dataloader:
            s2_labels = batch['s2_labels'].to(self.device)
            s1_labels = batch['s1_labels'].to(self.device)

            cond = upsample_s1_to_s2(s1_labels, self.num_classes).to(self.device)

            if self.distributed:
                loss = self.diffusion.module(s2_labels, cond=cond)
            else:
                loss = self.diffusion(s2_labels, cond=cond)

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False):
        """Save model checkpoint with milestones."""
        if not is_main_process(self.rank):
            return

        model_state = self.diffusion.module.state_dict() if self.distributed else self.diffusion.state_dict()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'base_lr': self.base_lr,
            'scaled_lr': self.scaled_lr,
            'world_size': self.world_size,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / 'latest.pt')

        # Save milestone checkpoints
        milestones = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        if epoch in milestones or epoch % 500 == 0:
            torch.save(checkpoint, self.output_dir / f's2_epoch_{epoch:04d}.pt')
            print(f'  Milestone checkpoint saved: epoch {epoch}')

        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            print(f'  New best model saved! val_loss={val_loss:.4f}')

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.distributed:
            self.diffusion.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.diffusion.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_epoch = checkpoint['epoch'] + 1

        if is_main_process(self.rank):
            print(f"Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
            if 'world_size' in checkpoint and checkpoint['world_size'] != self.world_size:
                print(f"  WARNING: World size changed from {checkpoint['world_size']} to {self.world_size}")
                print("  LR scaling may need adjustment")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 10000,
        log_every: int = 100,
    ):
        """Full training loop."""
        if is_main_process(self.rank):
            print(f"[S2Trainer] Starting training from epoch {self.start_epoch} to {num_epochs}")
            print(f"[S2Trainer] Train samples: {len(train_dataloader.dataset)}")
            print(f"[S2Trainer] Val samples: {len(val_dataloader.dataset)}")

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_loss = self.train_epoch(train_dataloader, epoch, log_every)
            val_loss = self.validate(val_dataloader)

            # Step the learning rate scheduler
            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Get current learning rate for logging
            current_lr = self.optimizer.param_groups[0]['lr']

            if is_main_process(self.rank):
                print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.6f}{" (best)" if is_best else ""}')

            self.save_checkpoint(epoch, train_loss, val_loss, is_best)

        if is_main_process(self.rank):
            print(f"[S2Trainer] Training complete! Best val_loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train S2 Pyramid Diffusion')
    parser.add_argument('--data-root', type=str,
                        default='datasets/SemanticKITTI_quantized',
                        help='Quantized data root')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/checkpoints/s2',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size per GPU (original: 16)')
    parser.add_argument('--epochs', type=int, default=10000, help='Max epochs')
    parser.add_argument('--lr', type=float, default=0.004, help='Learning rate (original: 0.004)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device (single GPU mode)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--no-scale-lr', action='store_true',
                        help='Disable learning rate scaling with world size')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs when using LR scaling (default: 5)')

    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank, distributed = setup_distributed()

    if not distributed:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        local_rank = args.gpu
    else:
        device = torch.device(f'cuda:{local_rank}')

    # Create datasets
    train_sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
    val_sequences = [8]

    train_dataset = S2Dataset(
        quantized_root=args.data_root,
        sequences=train_sequences,
        augment=True,
    )

    val_dataset = S2Dataset(
        quantized_root=args.data_root,
        sequences=val_sequences,
        augment=False,
    )

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create trainer
    trainer = S2Trainer(
        num_classes=20,
        num_timesteps=100,
        learning_rate=args.lr,
        device=device,
        output_dir=args.output_dir,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=distributed,
        scale_lr=not args.no_scale_lr,
        warmup_epochs=args.warmup_epochs,
        num_epochs=args.epochs,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.epochs,
            log_every=100,
        )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
