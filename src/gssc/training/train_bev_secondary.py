#!/usr/bin/env python3
"""BEV S2D2 Refinement Training -- refine SCPNet/TALoS BEV predictions toward GT BEV.

Cold diffusion forward: bev_t = alpha_t * GT_BEV + (1-alpha_t) * SCPNet_BEV
Training: KL posterior loss (provides regularization)
Sampling: S2D2 correction sampling (bypasses posterior bottleneck)

Usage:
  # BEV-A: Refine SCPNet BEV
  python bev_diffusion/train_bev_cold_diffusion.py \
      --output_dir outputs/bev_cold/BEV_A_scpnet \
      --pred_bev_dir datasets/scpnet_predictions \
      --batch_size 8 --num_iterations 100000

  # BEV-B: Refine TALoS BEV
  python bev_diffusion/train_bev_cold_diffusion.py \
      --output_dir outputs/bev_cold/BEV_B_talos \
      --pred_bev_dir datasets/talos_predictions \
      --batch_size 8 --num_iterations 100000
"""
import os
import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gssc.models.bev_multinomial_diffusion_2d import MultinomialDiffusion2D
from gssc.models.bev_unet_v2 import create_modular_bev_unet
from gssc.models.bev_lidar_encoder import SparseLiDAREncoder

try:
    import spconv.pytorch as spconv
except ImportError:
    import spconv


# ============================================================================
# Dataset
# ============================================================================

class BEVColdDiffusionDataset(Dataset):
    """Dataset for BEV S2D2 refinement training.

    Loads GT BEV, predicted BEV (SCPNet or TALoS), and LiDAR voxels.
    """

    def __init__(
        self,
        data_root: str,
        sequences: List[str],
        pred_bev_dir: str,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.pred_bev_dir = Path(pred_bev_dir)
        self.augment = augment
        self.voxels_root = self.data_root / 'SemanticKITTI_3D' / '256'
        self.ssc_root = self.data_root / 'dataset_SemanticKITTI_SSC'

        # Build sample list
        self.samples = []
        for seq in sequences:
            voxels_dir = self.voxels_root / seq
            if not voxels_dir.exists():
                continue

            bev_files = sorted(voxels_dir.glob('*_bev.npy'))
            for bev_file in bev_files:
                frame_id = bev_file.stem.replace('_bev', '')
                gt_bev_file = bev_file
                voxels_file = voxels_dir / f'{frame_id}_voxels.npy'
                pred_bev_file = self.pred_bev_dir / seq / f'{frame_id}_bev_top.npy'
                ssc_voxels_dir = self.ssc_root / 'sequences' / seq / 'voxels'
                invalid_file = ssc_voxels_dir / f'{frame_id}.invalid'

                if not voxels_file.exists() or not pred_bev_file.exists():
                    continue

                self.samples.append({
                    'gt_bev': str(gt_bev_file),
                    'voxels': str(voxels_file),
                    'pred_bev': str(pred_bev_file),
                    'invalid_file': str(invalid_file) if invalid_file.exists() else None,
                    'seq': seq,
                    'frame': frame_id,
                })

        print(f"BEVColdDiffusionDataset: {len(self.samples)} samples from {len(sequences)} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        gt_bev = np.load(sample['gt_bev']).astype(np.int64)
        voxels = np.load(sample['voxels']).astype(np.float32)
        voxels = (voxels > 0).astype(np.float32)
        pred_bev = np.load(sample['pred_bev']).astype(np.int64)

        # Augmentation: random flip + rotation
        if self.augment:
            if np.random.rand() > 0.5:
                gt_bev = np.flip(gt_bev, axis=0).copy()
                voxels = np.flip(voxels, axis=0).copy()
                pred_bev = np.flip(pred_bev, axis=0).copy()
            if np.random.rand() > 0.5:
                gt_bev = np.flip(gt_bev, axis=1).copy()
                voxels = np.flip(voxels, axis=1).copy()
                pred_bev = np.flip(pred_bev, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                gt_bev = np.rot90(gt_bev, k).copy()
                voxels = np.rot90(voxels, k, axes=(0, 1)).copy()
                pred_bev = np.rot90(pred_bev, k).copy()

        result = {
            'gt_bev': torch.from_numpy(gt_bev).long(),
            'lidar_voxels': torch.from_numpy(voxels).float().unsqueeze(0),  # [1, 256, 256, 32]
            'pred_bev': torch.from_numpy(pred_bev).long(),
            'frame': sample['frame'],
        }

        if sample['invalid_file'] is not None:
            inv = np.unpackbits(np.fromfile(sample['invalid_file'], dtype=np.uint8))
            inv = inv.reshape(256, 256, 32)
            bev_invalid = inv.all(axis=2)
            result['invalid_mask'] = torch.from_numpy(bev_invalid).bool()
        else:
            result['invalid_mask'] = torch.zeros(256, 256, dtype=torch.bool)

        return result


# ============================================================================
# Metrics
# ============================================================================

class BEVMetrics:
    """BEV semantic segmentation metrics (mIoU over 19 classes)."""

    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target, invalid_mask=None):
        pred = pred.flatten()
        target = target.flatten()
        if invalid_mask is not None:
            valid = ~invalid_mask.flatten()
            pred = pred[valid]
            target = target[valid]
        np.add.at(self.conf_matrix, (pred, target), 1)

    def get_iou(self):
        tp = np.diag(self.conf_matrix)
        fp = self.conf_matrix.sum(axis=1) - tp
        fn = self.conf_matrix.sum(axis=0) - tp
        iou_per_class = {}
        for c in range(1, self.num_classes):
            denom = tp[c] + fp[c] + fn[c]
            iou_per_class[c] = tp[c] / max(denom, 1)
        valid_ious = list(iou_per_class.values())
        miou = np.mean(valid_ious) if valid_ious else 0.0
        return {'mIoU': miou, 'class_iou': iou_per_class}


# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        self.backup = {n: p.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self.backup[n])


# ============================================================================
# Trainer
# ============================================================================

class BEVColdDiffusionTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config["gpu"]}')
        self.global_step = 0
        self.best_miou = 0.0

        os.makedirs(config['output_dir'], exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config['output_dir'], 'training.log')),
                logging.StreamHandler(),
            ]
        )
        self.logger = logging.getLogger()
        self.logger.info(f"Config: {config}")

        # Create model
        self.encoder = SparseLiDAREncoder(in_channels=1, out_channels=64).to(self.device)
        self.unet = create_modular_bev_unet(
            num_classes=20, cond_channels=64,
            model_size='base', input_resolution=256, conditioning_type='sum',
        ).to(self.device)

        total_params = sum(p.numel() for p in self.encoder.parameters()) + \
                       sum(p.numel() for p in self.unet.parameters())
        self.logger.info(f"Model: encoder + UNet = {total_params:,} params")

        # Diffusion
        self.diffusion = MultinomialDiffusion2D(
            num_classes=20, num_timesteps=config['num_timesteps'],
            beta_max=0.1, focal_gamma=2.0, class_0_weight=0.02,
            occupied_weight=10.0, lovasz_weight=0.3, obs_weight_factor=2.0,
        ).to(self.device)

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.unet.parameters())
        self.optimizer = AdamW(params, lr=config['lr'], weight_decay=0.01)

        # EMA
        self.ema_encoder = EMA(self.encoder)
        self.ema_unet = EMA(self.unet)

        # Data
        train_seqs = config.get('train_sequences',
                                ['00','01','02','03','04','05','06','07','09','10'])
        self.train_dataset = BEVColdDiffusionDataset(
            config['data_root'], train_seqs, config['pred_bev_dir'], augment=True)
        self.val_dataset = BEVColdDiffusionDataset(
            config['data_root'], ['08'], config['pred_bev_dir'], augment=False)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config['batch_size'],
            shuffle=True, num_workers=4, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=2)

    def _encode_lidar(self, voxels):
        B = voxels.shape[0]
        coords_list, feats_list = [], []
        for b in range(B):
            nz = voxels[b, 0].nonzero()
            batch_col = torch.full((nz.shape[0], 1), b, device=voxels.device, dtype=torch.int32)
            coords_list.append(torch.cat([batch_col, nz.int()], dim=1))
            feats_list.append(voxels[b, :, nz[:, 0], nz[:, 1], nz[:, 2]].T)
        coords = torch.cat(coords_list, dim=0)
        feats = torch.cat(feats_list, dim=0)
        return self.encoder(feats, coords, B)

    def _model_fn(self, x_t, t, lidar_features):
        x_t_classes = x_t.argmax(dim=1)
        out = self.unet(x_t_classes, t, lidar_features)
        # UNet may return (logits, aux_logits) tuple
        return out[0] if isinstance(out, tuple) else out

    def train_step(self, batch):
        self.encoder.train()
        self.unet.train()

        gt_bev = batch['gt_bev'].to(self.device)
        voxels = batch['lidar_voxels'].to(self.device)
        pred_bev = batch['pred_bev'].to(self.device)

        lidar_features = self._encode_lidar(voxels)
        pred_bev_oh = F.one_hot(pred_bev.long(), 20).float().permute(0, 3, 1, 2)

        B = gt_bev.shape[0]
        t = torch.randint(0, self.config['num_timesteps'], (B,), device=self.device)

        losses = self.diffusion.training_losses(
            self._model_fn, gt_bev, t, lidar_features, x_scpnet=pred_bev_oh)

        loss = losses['loss']
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.unet.parameters()), 1.0)
        self.optimizer.step()

        self.ema_encoder.update(self.encoder)
        self.ema_unet.update(self.unet)

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    @torch.no_grad()
    def run_algo2_on_samples(self):
        """Run S2D2 correction sampling on fixed 100 val samples with official BEV metrics."""
        self.encoder.eval()
        self.unet.eval()

        rng = np.random.RandomState(42)
        total = len(self.val_dataset)
        indices = rng.choice(total, min(100, total), replace=False)

        metrics = BEVMetrics(num_classes=20)
        scpnet_metrics = BEVMetrics(num_classes=20)

        for idx in indices:
            sample = self.val_dataset[idx]
            gt_bev = sample['gt_bev'].numpy()
            pred_bev = sample['pred_bev'].unsqueeze(0).to(self.device)
            voxels = sample['lidar_voxels'].unsqueeze(0).to(self.device)

            lidar_features = self._encode_lidar(voxels)
            refined = self.diffusion.sample_algo2(
                self._model_fn, lidar_features, pred_bev,
                n_steps=self.config.get(
                    'correction_eval_steps',
                    self.config.get('algo2_eval_steps', 100),
                ))

            # BEV eval: no invalid mask (standard 2D seg), mIoU over classes 1-19
            metrics.update(refined.cpu().numpy()[0], gt_bev)
            scpnet_metrics.update(sample['pred_bev'].numpy(), gt_bev)

        return metrics.get_iou(), scpnet_metrics.get_iou()

    def save_checkpoint(self, filename):
        path = os.path.join(self.config['output_dir'], filename)
        torch.save({
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_encoder': self.ema_encoder.shadow,
            'ema_unet': self.ema_unet.shadow,
            'best_miou': self.best_miou,
        }, path)

    def train(self):
        self.logger.info("Starting BEV S2D2 refinement training...")
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.config['num_iterations'], initial=self.global_step, desc="Training")

        while self.global_step < self.config['num_iterations']:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)
            self.global_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")

            if self.global_step % 100 == 0:
                self.logger.info(f"Step {self.global_step}: loss={metrics['loss']:.4f}")

            if self.global_step % self.config.get('miou_interval', 5000) == 0:
                self.logger.info(f"Running S2D2 BEV eval at step {self.global_step}...")

                # Training weights
                result, scp_result = self.run_algo2_on_samples()
                delta = (result['mIoU'] - scp_result['mIoU']) * 100
                self.logger.info(
                    f"Step {self.global_step} [mIoU-train]: "
                    f"mIoU={result['mIoU']*100:.2f}% "
                    f"(SCPNet={scp_result['mIoU']*100:.2f}%, Δ={delta:+.2f}%)")

                if result['mIoU'] > self.best_miou:
                    self.best_miou = result['mIoU']
                    self.save_checkpoint('best_miou.pt')
                    self.logger.info(f"New best mIoU: {self.best_miou*100:.2f}%")

                # EMA weights
                self.ema_encoder.apply_shadow(self.encoder)
                self.ema_unet.apply_shadow(self.unet)
                ema_result, _ = self.run_algo2_on_samples()
                ema_delta = (ema_result['mIoU'] - scp_result['mIoU']) * 100
                self.logger.info(
                    f"Step {self.global_step} [mIoU-EMA]: "
                    f"mIoU={ema_result['mIoU']*100:.2f}% "
                    f"(SCPNet={scp_result['mIoU']*100:.2f}%, Δ={ema_delta:+.2f}%)")
                self.ema_encoder.restore(self.encoder)
                self.ema_unet.restore(self.unet)

            if self.global_step % self.config.get('save_interval', 5000) == 0:
                self.save_checkpoint(f'step_{self.global_step}.pt')

        pbar.close()
        self.save_checkpoint('final.pt')
        self.logger.info("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BEV S2D2 Refinement Training')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='datasets')
    parser.add_argument('--pred_bev_dir', type=str, required=True,
                        help='Dir with predicted BEV (e.g., datasets/scpnet_predictions)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--algo2_eval_steps', type=int, default=100)
    parser.add_argument('--miou_interval', type=int, default=5000)
    parser.add_argument('--save_interval', type=int, default=5000)
    # The following four flags are forwarded by the bev_secondary.yaml recipe
    # via scripts/train.py. They are accepted (and validated) here so the
    # documented `python scripts/train.py train/bev_secondary` command parses
    # cleanly. They are currently informational at the trainer level: the model
    # is built with the hardcoded values matching these defaults (task='bev',
    # 20 classes, encoder out=64, UNet model_size='base'), so passing them does
    # not change the reproduced 36.09% BEV mIoU run.
    parser.add_argument('--task', type=str, default='bev',
                        help='Secondary-task discriminant (bev). Used by '
                             'scripts/train.py for trainer dispatch.')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='Number of semantic classes (fixed at 20 for '
                             'SemanticKITTI; model is built with 20).')
    parser.add_argument('--base_channels', type=int, default=128,
                        help='Documented base channel width of the BEV UNet '
                             '(informational; UNet uses model_size=base).')
    parser.add_argument('--lidar_channels', type=int, default=128,
                        help='Documented LiDAR-encoder channel width '
                             '(informational; encoder out_channels=64).')
    parser.add_argument('--train_sequences', type=str, default=None,
                        help='Comma-separated train sequences (default: 00-10 real only)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    # scripts/train.py always injects --seed (default 42). The original BEV-A
    # 36.09% reproduction did no RNG seeding, so this flag is accepted but not
    # applied to preserve exact reproduction behavior; it is registered here so
    # the documented `python scripts/train.py train/bev_secondary` parses cleanly
    # instead of crashing with SystemExit 2 on an unrecognized argument.
    parser.add_argument('--seed', type=int, default=42,
                        help='Accepted for CLI compatibility with scripts/train.py; '
                             'not applied (BEV-A reproduction does no RNG seeding).')
    args = parser.parse_args()

    config = vars(args)
    if args.train_sequences:
        config['train_sequences'] = args.train_sequences.split(',')
    else:
        config['train_sequences'] = ['00','01','02','03','04','05','06','07','09','10']
    trainer = BEVColdDiffusionTrainer(config)

    if args.resume:
        import torch as _torch
        ckpt = _torch.load(args.resume, map_location='cpu', weights_only=False)
        trainer.encoder.load_state_dict(ckpt['encoder_state_dict'])
        trainer.unet.load_state_dict(ckpt['unet_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.ema_encoder.shadow = {k: v.to(trainer.device) for k, v in ckpt['ema_encoder'].items()}
        trainer.ema_unet.shadow = {k: v.to(trainer.device) for k, v in ckpt['ema_unet'].items()}
        trainer.global_step = ckpt['global_step']
        trainer.best_miou = ckpt.get('best_miou', 0.0)
        trainer.logger.info(f"Resumed from step {trainer.global_step}")

    trainer.train()
