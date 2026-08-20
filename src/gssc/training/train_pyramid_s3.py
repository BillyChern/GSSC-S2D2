#!/usr/bin/env python3
"""
S3 (144x144x32 sub-scenes → 256x256x32 full) Pyramid Diffusion Training for SemanticKITTI.

S3 is trained with DUAL conditioning:
1. Upsampled S2 sub-patches (36×36×8 → 144×144×32)
2. Masked full-resolution sub-scenes (for scene completion / inpainting)

This follows the pyramid-discrete-diffusion paper but with FULL SCENE COVERAGE.
The original paper uses 128×128 with ~93.8% coverage. We use 144×144 with 32-voxel
overlap for 100% coverage: 144 + 144 - 32 = 256.

IMPORTANT: Z-axis is NEVER divided (confirmed from original spilt_scenes.py line 22).

Key dimensions for SemanticKITTI:
- Full resolution: 256×256×32
- Sub-scene target: 144×144×32 (4 sub-scenes per frame with 32-voxel overlap)
- S2 quantized: 64×64×8
- S2 sub-patch: 36×36×8 (upsampled to 144×144×32 for conditioning)

Subdivision scheme (training with overlap for inpainting):
- Scene 256×256×32 → 4 overlapping sub-scenes of 144×144×32
- Step = 112 (overlap = 32)
- Positions: (0,0), (112,0), (0,112), (112,112) in XY

Training mask types (25% probability each, matching inference):
- Type 0: TOP edge (keep Y[-32:]) - for S3-1 (sees S3-0's bottom)
- Type 1: LEFT edge (keep X[:32]) - for S3-2 (sees S3-0's right)
- Type 2: L-SHAPE (keep X[:32] OR Y[-32:]) - for S3-3 (sees S3-1 bottom, S3-2 right)
- Type 3: NOTHING (fully masked) - for S3-0 (no prior context)

Fusion scheme (inference: inpainting with voting):
- Autoregressive generation: S3-0 → S3-1 → S3-2 → S3-3
- Each sub-scene uses previously generated neighbors as context
- Voting mechanism resolves conflicts in overlap regions

Usage:
    # Single GPU
    python train_s3.py --gpu 0

    # Multi-GPU (use torchrun)
    torchrun --nproc_per_node=4 train_s3.py

    # Resume from checkpoint
    python train_s3.py --resume outputs/checkpoints/s3/latest.pt
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

from gssc.models.pyramid_diffusion import PyramidDiscreteDiffusion
from gssc.models.pyramid_unet import Denoise

# Fixed frames for mIoU evaluation (10 random frames from sequence 08)
EVAL_FRAMES = ["000102", "000419", "000456", "000571", "000914",
               "001003", "001126", "002619", "003016", "003037"]


def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 20) -> dict:
    """Compute IoU metrics."""
    ious = {}
    for c in range(1, num_classes):  # Skip class 0 (unlabeled)
        pred_c = pred == c
        gt_c = gt == c
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union > 0:
            ious[c] = intersection / union
    valid_ious = [v for v in ious.values()]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    return {'miou': miou, 'class_ious': ious}


# SemanticKITTI learning map (original label → training label 0-19)
LEARNING_MAP = {
    0: 0,      # "unlabeled"
    1: 0,      # "outlier"
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus"
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails"
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure"
    60: 9,     # "lane-marking"
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object"
    252: 1,    # "moving-car"
    253: 7,    # "moving-bicyclist"
    254: 6,    # "moving-person"
    255: 8,    # "moving-motorcyclist"
    256: 5,    # "moving-on-rails"
    257: 5,    # "moving-bus"
    258: 4,    # "moving-truck"
    259: 5,    # "moving-other-vehicle"
}


def apply_learning_map(labels: np.ndarray) -> np.ndarray:
    """Convert original SemanticKITTI labels to training labels (0-19)."""
    mapped = np.zeros_like(labels, dtype=np.uint8)
    for orig_label, train_label in LEARNING_MAP.items():
        mapped[labels == orig_label] = train_label
    return mapped


class S3Dataset(Dataset):
    """
    Dataset for S3 training with OVERLAPPING subdivision and dual conditioning.

    For each frame:
    - Loads full resolution labels (256×256×32) and applies learning_map
    - Loads S2 quantized data (64×64×8)
    - Extracts 4 OVERLAPPING sub-scenes (144×144×32 with step=112)
    - Extracts corresponding S2 patches (36×36×8 with step=28)
    - Creates masked versions for dual conditioning (inpainting training)

    IMPORTANT: Z-axis is NEVER divided (from original spilt_scenes.py).

    Key insight: Training with overlapping regions that match inference!
    - Sub-scene size: 144×144×32 (divisible by 16 for UNet)
    - Step: 112 (overlap = 32)
    - Full coverage: 144 + 144 - 32 = 256

    S2 patch extraction:
    - S2 resolution: 64×64×8 (scale: 4× in XY, 4× in Z)
    - S2 patch size: 36×36×8 (144/4 = 36)
    - S2 step: 28 (112/4 = 28)

    Returns sub-scenes of size 144×144×32 for SemanticKITTI.
    """

    # Sub-scene target size for SemanticKITTI (Z = full 32, not divided)
    TARGET_SIZE = (144, 144, 32)

    # Overlap and step parameters for full 256×256 coverage
    OVERLAP = 32
    STEP = 112  # 144 - 32 = 112

    # S2 patch parameters (scale factor: 4× in XY, 4× in Z)
    S2_PATCH_SIZE = (36, 36, 8)  # 144/4 = 36
    S2_STEP = 28  # 112/4 = 28

    # Number of sub-scenes per frame (2×2 in XY, no Z division)
    NUM_SUB_SCENES = 4

    # Sub-scene positions in XY (full resolution)
    SUB_SCENE_POSITIONS = [(0, 0), (112, 0), (0, 112), (112, 112)]

    # S2 patch positions (S2 resolution)
    S2_PATCH_POSITIONS = [(0, 0), (28, 0), (0, 28), (28, 28)]

    # Mask parameters: 32/144 ≈ 22.22% overlap region
    MASK_RATIO = 32 / 144  # 0.2222...
    MASK_PROBS = [0.25, 0.25, 0.25, 0.25]  # Equal prob for each type

    def __init__(
        self,
        ssc_root: str,
        quantized_root: str,
        sequences: list[int],
        augment: bool = True,
        use_prequantized: bool = True,
    ):
        self.ssc_root = Path(ssc_root)
        self.quantized_root = Path(quantized_root)
        self.augment = augment
        self.use_prequantized = use_prequantized

        self.s2_path = self.quantized_root / 's2' / 'sequences'
        self.s3_cond_path = self.quantized_root / 's3_cond' / 'sequences'

        # Check if pre-quantized data exists
        if use_prequantized and not self.s3_cond_path.exists():
            print(f"[S3Dataset] Warning: Pre-quantized data not found at {self.s3_cond_path}")
            print("[S3Dataset] Falling back to online upsampling (slower)")
            self.use_prequantized = False

        # Find all valid frames
        self.frames = []
        for seq_id in sequences:
            voxels_dir = self.ssc_root / 'sequences' / f'{seq_id:02d}' / 'voxels'
            s2_dir = self.s2_path / f'{seq_id:02d}'

            if not voxels_dir.exists() or not s2_dir.exists():
                print(f"  Warning: Missing sequence {seq_id}")
                continue

            # Get frame IDs from .label files
            label_files = sorted(voxels_dir.glob('*.label'))
            for label_file in label_files:
                frame_id = label_file.stem
                s2_file = s2_dir / f'{frame_id}.npy'
                if s2_file.exists():
                    # If using pre-quantized, check that all sub-scene files exist
                    if self.use_prequantized:
                        s3_cond_dir = self.s3_cond_path / f'{seq_id:02d}'
                        all_exist = all(
                            (s3_cond_dir / f'{frame_id}_sub{i}.npy').exists()
                            for i in range(self.NUM_SUB_SCENES)
                        )
                        if all_exist:
                            self.frames.append((seq_id, frame_id))
                    else:
                        self.frames.append((seq_id, frame_id))

        print(f"[S3Dataset] Loaded {len(self.frames)} frames from {len(sequences)} sequences")
        print(f"[S3Dataset] Sub-scene size: {self.TARGET_SIZE} with step={self.STEP} (overlap={self.OVERLAP})")
        if self.use_prequantized:
            print("[S3Dataset] Using PRE-QUANTIZED S2 conditioning (fast)")
        else:
            print("[S3Dataset] Using ONLINE upsampling for S2 conditioning (slow)")
            print(f"[S3Dataset] S2 patch size: {self.S2_PATCH_SIZE} with step={self.S2_STEP}")
        print(f"[S3Dataset] {self.NUM_SUB_SCENES} overlapping sub-scenes per frame (2×2 in XY)")
        print(f"[S3Dataset] Mask ratio: {self.MASK_RATIO:.4f} (32/144 for 32-voxel overlap)")
        print(f"[S3Dataset] Total samples: {len(self.frames) * self.NUM_SUB_SCENES}")

    def __len__(self) -> int:
        return len(self.frames) * self.NUM_SUB_SCENES

    def _load_full_labels(self, seq_id: int, frame_id: str) -> np.ndarray:
        """Load full resolution labels (256×256×32) with learning map applied."""
        label_path = self.ssc_root / 'sequences' / f'{seq_id:02d}' / 'voxels' / f'{frame_id}.label'
        labels = np.fromfile(label_path, dtype=np.uint16).reshape(256, 256, 32)
        return apply_learning_map(labels)

    def _load_s2(self, seq_id: int, frame_id: str) -> np.ndarray:
        """Load S2 quantized data (64×64×8)."""
        s2_path = self.s2_path / f'{seq_id:02d}' / f'{frame_id}.npy'
        return np.load(s2_path).astype(np.int64)

    def _load_prequantized_s2_cond(self, seq_id: int, frame_id: str, sub_idx: int) -> np.ndarray:
        """Load pre-quantized S2 conditioning (already upsampled to 144×144×32)."""
        cond_path = self.s3_cond_path / f'{seq_id:02d}' / f'{frame_id}_sub{sub_idx}.npy'
        return np.load(cond_path).astype(np.int64)

    def _extract_sub_scenes(self, scene: np.ndarray) -> list[np.ndarray]:
        """
        Extract 4 OVERLAPPING sub-scenes from full resolution scene.

        256×256×32 → 4 sub-scenes of 144×144×32
        Positions: (0,0), (112,0), (0,112), (112,112) in XY
        Step = 112, Overlap = 32, 144 + 144 - 32 = 256 (full coverage)
        """
        sub_scenes = []
        sx, sy, sz = self.TARGET_SIZE

        for x_start, y_start in self.SUB_SCENE_POSITIONS:
            sub_scene = scene[
                x_start:x_start + sx,
                y_start:y_start + sy,
                :  # Full Z dimension
            ].copy()
            sub_scenes.append(sub_scene)

        return sub_scenes

    def _extract_s2_patches(self, s2: np.ndarray) -> list[np.ndarray]:
        """
        Extract 4 OVERLAPPING S2 patches corresponding to full resolution sub-scenes.

        S2 resolution: 64×64×8 (scale factor: 4× in XY, 4× in Z)
        64×64×8 → 4 patches of 36×36×8
        Positions: (0,0), (28,0), (0,28), (28,28) in S2 coordinates
        Step = 28, Overlap = 8, 36 + 36 - 8 = 64 (full S2 coverage)
        """
        patches = []
        px, py, pz = self.S2_PATCH_SIZE

        for x_start, y_start in self.S2_PATCH_POSITIONS:
            patch = s2[
                x_start:x_start + px,
                y_start:y_start + py,
                :  # Full Z dimension (8)
            ].copy()
            patches.append(patch)

        return patches

    def _mask_scene(self, scene: np.ndarray) -> np.ndarray:
        """
        Create masked version of scene for dual conditioning (inpainting training).

        Mask types (25% probability each, matching inference positions):
        - Type 0: TOP edge only (keep Y[-32:]) - for S3-1 (sees S3-0's bottom)
        - Type 1: LEFT edge only (keep X[:32]) - for S3-2 (sees S3-0's right)
        - Type 2: L-SHAPE (keep X[:32] OR Y[-32:]) - for S3-3 (sees S3-1 bottom, S3-2 right)
        - Type 3: NOTHING (fully masked) - for S3-0 (no prior context)

        This matches the original mask_scene.py from pyramid-discrete-diffusion:
        - Type 0: mask[:, :-y_limit, :] = 0 (keep bottom/top edge in Y)
        - Type 1: mask[x_limit:, :, :] = 0 (keep left edge in X)
        - Type 2: mask[x_limit:, :-y_limit, :] = 0 (L-shape: union of X[:32] and Y[-32:])
        - Type 3: mask[:, :, :] = 0 (fully masked)

        The overlap region is 32 voxels = 32/144 ≈ 22.22% of each edge.
        """
        masked = scene.copy()
        x, y, z = scene.shape
        mask_type = np.random.choice(4, p=self.MASK_PROBS)

        # Calculate limits: 32 voxels for 144×144 sub-scenes
        x_limit = int(x * self.MASK_RATIO)  # 32 voxels
        y_limit = int(y * self.MASK_RATIO)  # 32 voxels

        if mask_type == 0:  # TOP edge: keep Y[-32:] (sees previous sub-scene's bottom)
            # Zero out everything except the top 32 rows in Y
            masked[:, :-y_limit, :] = 0
        elif mask_type == 1:  # LEFT edge: keep X[:32] (sees previous sub-scene's right)
            # Zero out everything except the left 32 columns in X
            masked[x_limit:, :, :] = 0
        elif mask_type == 2:  # L-SHAPE: keep X[:32] OR Y[-32:] (union)
            # Zero out the intersection (where both x >= 32 AND y < y_max-32)
            # This keeps the L-shape: left edge + top edge
            masked[x_limit:, :-y_limit, :] = 0
        else:  # mask_type == 3: NOTHING (fully masked, no prior context)
            masked[:, :, :] = 0

        return masked

    def __getitem__(self, idx: int) -> dict:
        # Determine frame and sub-scene index
        frame_idx = idx // self.NUM_SUB_SCENES
        sub_idx = idx % self.NUM_SUB_SCENES
        seq_id, frame_id = self.frames[frame_idx]

        # Load full resolution labels
        full_labels = self._load_full_labels(seq_id, frame_id)

        if self.use_prequantized:
            # Load pre-quantized S2 conditioning (already upsampled to 144×144×32)
            # Note: We load ALL 4 sub-scenes' conditioning for augmentation consistency
            s2_cond_all = [
                self._load_prequantized_s2_cond(seq_id, frame_id, i)
                for i in range(self.NUM_SUB_SCENES)
            ]

            # Apply augmentation (must transform both full_labels and s2_cond consistently)
            if self.augment:
                # Random flip X
                if np.random.random() > 0.5:
                    full_labels = np.flip(full_labels, axis=0).copy()
                    s2_cond_all = [np.flip(c, axis=0).copy() for c in s2_cond_all]
                    # Swap sub-scenes: (0,1,2,3) -> (1,0,3,2) for X-flip
                    s2_cond_all = [s2_cond_all[1], s2_cond_all[0], s2_cond_all[3], s2_cond_all[2]]
                # Random flip Y
                if np.random.random() > 0.5:
                    full_labels = np.flip(full_labels, axis=1).copy()
                    s2_cond_all = [np.flip(c, axis=1).copy() for c in s2_cond_all]
                    # Swap sub-scenes: (0,1,2,3) -> (2,3,0,1) for Y-flip
                    s2_cond_all = [s2_cond_all[2], s2_cond_all[3], s2_cond_all[0], s2_cond_all[1]]
                # Random rotation in XY plane
                k = np.random.randint(4)
                if k > 0:
                    full_labels = np.rot90(full_labels, k, axes=(0, 1)).copy()
                    s2_cond_all = [np.rot90(c, k, axes=(0, 1)).copy() for c in s2_cond_all]
                    # Rotate sub-scene indices
                    for _ in range(k):
                        # 90° rotation: (0,1,2,3) -> (1,3,0,2)
                        s2_cond_all = [s2_cond_all[1], s2_cond_all[3], s2_cond_all[0], s2_cond_all[2]]

            # Extract sub-scenes and get S2 conditioning
            full_sub_scenes = self._extract_sub_scenes(full_labels)
            s3_target = full_sub_scenes[sub_idx]
            s2_upsampled = s2_cond_all[sub_idx]  # Already 144×144×32

            # Create masked version for dual conditioning
            s3_masked = self._mask_scene(s3_target)

            return {
                's3_target': torch.from_numpy(s3_target).long(),
                's2_upsampled': torch.from_numpy(s2_upsampled).long(),  # Pre-upsampled!
                's3_masked': torch.from_numpy(s3_masked).long(),
                'sequence': seq_id,
                'frame_id': frame_id,
                'sub_idx': sub_idx,
                'prequantized': True,
            }
        else:
            # Online upsampling path (slower, for backwards compatibility)
            s2_labels = self._load_s2(seq_id, frame_id)

            # Apply augmentation (before extraction to maintain consistency)
            if self.augment:
                # Random flip X
                if np.random.random() > 0.5:
                    full_labels = np.flip(full_labels, axis=0).copy()
                    s2_labels = np.flip(s2_labels, axis=0).copy()
                # Random flip Y
                if np.random.random() > 0.5:
                    full_labels = np.flip(full_labels, axis=1).copy()
                    s2_labels = np.flip(s2_labels, axis=1).copy()
                # Random rotation in XY plane
                k = np.random.randint(4)
                if k > 0:
                    full_labels = np.rot90(full_labels, k, axes=(0, 1)).copy()
                    s2_labels = np.rot90(s2_labels, k, axes=(0, 1)).copy()

            # Extract overlapping sub-scenes and S2 patches
            full_sub_scenes = self._extract_sub_scenes(full_labels)
            s2_patches = self._extract_s2_patches(s2_labels)

            # Get specific sub-scene (144×144×32) and S2 patch (36×36×8)
            s3_target = full_sub_scenes[sub_idx]
            s2_patch = s2_patches[sub_idx]

            # Create masked version for dual conditioning (inpainting training)
            s3_masked = self._mask_scene(s3_target)

            return {
                's3_target': torch.from_numpy(s3_target).long(),
                's2_patch': torch.from_numpy(s2_patch).long(),  # Needs upsampling in trainer
                's3_masked': torch.from_numpy(s3_masked).long(),
                'sequence': seq_id,
                'frame_id': frame_id,
                'sub_idx': sub_idx,
                'prequantized': False,
            }


def upsample_to_target(labels: torch.Tensor, target_size: tuple[int, int, int], num_classes: int = 20) -> torch.Tensor:
    """
    Upsample labels to target size using trilinear interpolation on one-hot encoding.

    Args:
        labels: [B, X, Y, Z] tensor of class labels
        target_size: (X, Y, Z) target dimensions
        num_classes: number of classes

    Returns:
        [B, 1, X, Y, Z] tensor of upsampled labels (byte)
    """
    # Convert to one-hot: [B, X, Y, Z] → [B, C, X, Y, Z]
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    one_hot = one_hot.permute(0, 4, 1, 2, 3)

    # Trilinear interpolation
    interpolated = F.interpolate(one_hot, size=target_size, mode='trilinear', align_corners=False)

    # Convert back to labels
    upsampled = interpolated.argmax(dim=1)
    return upsampled.unsqueeze(1).byte()


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


class S3Trainer:
    """Trainer for S3 pyramid diffusion with dual conditioning and multi-GPU support."""

    # Target sub-scene size for SemanticKITTI (Z NOT divided)
    # 144×144×32 with 32-voxel overlap for full 256×256 coverage
    TARGET_SIZE = (144, 144, 32)

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        learning_rate: float = 0.004,
        device: torch.device = None,
        output_dir: str = 'outputs/checkpoints/s3',
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        distributed: bool = False,
        scale_lr: bool = True,
        warmup_epochs: int = 5,
        num_epochs: int = 1000,
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

        # Create denoiser with S3 configuration (dual conditioning)
        class Args:
            next_stage = 's_3'  # Enables dual conditioning (context_size = init_size + 2)
            prev_stage = 's_2'

        # Use init_size=32 (default in original con_diffusion.py)
        # Note: config's init_size=8 is NOT passed to Denoise, it uses default=32
        self.denoiser = Denoise(
            args=Args(),
            num_class=num_classes,
            init_size=32,  # Default value used in original code
            discrete=True
        ).to(self.device)

        self.diffusion = PyramidDiscreteDiffusion(
            args=None,
            denoise_model=self.denoiser,
            num_classes=num_classes,
            num_timesteps=num_timesteps,
            multi_criterion=None,
            auxiliary_loss_weight=0.0005,  # Matches reference paper YAML config
            adaptive_auxiliary_loss=True,
            recon_loss=False,
        ).to(self.device)

        if distributed:
            self.diffusion = DDP(self.diffusion, device_ids=[local_rank], output_device=local_rank)

        # Scale learning rate based on global batch size
        self.base_lr = learning_rate
        if scale_lr and world_size > 1:
            self.scaled_lr = learning_rate * world_size
        else:
            self.scaled_lr = learning_rate

        self.optimizer = AdamW(
            self.diffusion.parameters(),
            lr=self.scaled_lr,
            betas=(0.9, 0.999),
        )

        self.scheduler = self._create_scheduler_with_warmup()

        if is_main_process(rank):
            total_params = sum(p.numel() for p in self.diffusion.parameters())
            print(f"[S3Trainer] Total parameters: {total_params:,}")
            print(f"[S3Trainer] Device: {self.device}")
            print(f"[S3Trainer] Target sub-scene size: {self.TARGET_SIZE} (full 256×256 coverage)")
            print("[S3Trainer] S2 patch: 36×36×8 → upsampled to 144×144×32")
            print("[S3Trainer] Dual conditioning: upsampled S2 + masked S3 (inpainting)")
            print(f"[S3Trainer] Distributed: {distributed}, World size: {world_size}")
            print(f"[S3Trainer] Base LR: {self.base_lr}, Scaled LR: {self.scaled_lr}")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.start_epoch = 1

        # Paths for mIoU evaluation
        self.ssc_root = None
        self.quantized_root = None

    def _create_scheduler_with_warmup(self):
        """Create learning rate scheduler with linear warmup."""
        if self.scale_lr and self.world_size > 1 and self.warmup_epochs > 0:
            def warmup_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return (1.0 / self.world_size) + (1.0 - 1.0 / self.world_size) * (epoch / self.warmup_epochs)
                return 1.0

            warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs - self.warmup_epochs,
                eta_min=1e-6,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs]
            )
        else:
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
            s3_target = batch['s3_target'].to(self.device)
            s3_masked = batch['s3_masked'].to(self.device)

            # Handle both pre-quantized and online upsampling paths
            if batch.get('prequantized', [False])[0]:
                # Pre-quantized path: s2_upsampled is already 144×144×32
                s2_upsampled = batch['s2_upsampled'].to(self.device)
                s2_upsampled = s2_upsampled.unsqueeze(1).byte()  # Add channel dim
            else:
                # Online upsampling path: s2_patch is 36×36×8
                s2_patch = batch['s2_patch'].to(self.device)
                s2_upsampled = upsample_to_target(s2_patch, self.TARGET_SIZE, self.num_classes).to(self.device)

            # Create dual conditioning: [upsampled_s2, masked_s3]
            # Both should be [B, 1, 144, 144, 32]
            s3_masked_cond = s3_masked.unsqueeze(1).byte()
            cond = torch.cat([s2_upsampled, s3_masked_cond], dim=1)

            # Forward pass
            if self.distributed:
                loss = self.diffusion.module(s3_target, cond=cond)
            else:
                loss = self.diffusion(s3_target, cond=cond)

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
            s3_target = batch['s3_target'].to(self.device)
            s3_masked = batch['s3_masked'].to(self.device)

            # Handle both pre-quantized and online upsampling paths
            if batch.get('prequantized', [False])[0]:
                s2_upsampled = batch['s2_upsampled'].to(self.device)
                s2_upsampled = s2_upsampled.unsqueeze(1).byte()
            else:
                s2_patch = batch['s2_patch'].to(self.device)
                s2_upsampled = upsample_to_target(s2_patch, self.TARGET_SIZE, self.num_classes).to(self.device)

            s3_masked_cond = s3_masked.unsqueeze(1).byte()
            cond = torch.cat([s2_upsampled, s3_masked_cond], dim=1)

            if self.distributed:
                loss = self.diffusion.module(s3_target, cond=cond)
            else:
                loss = self.diffusion(s3_target, cond=cond)

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate_miou(self, ssc_root: str, quantized_root: str) -> float:
        """
        Evaluate mIoU on fixed 10 frames from sequence 08.

        This generates samples using the diffusion model and compares with GT.
        Takes ~10-15 minutes per evaluation.
        """
        self.diffusion.eval()

        # Sub-scene parameters (matching S3Sampler)
        SUB_SCENE_SIZE = (144, 144, 32)
        SUB_SCENE_POSITIONS = [(0, 0), (112, 0), (0, 112), (112, 112)]
        S2_PATCH_POSITIONS = [(0, 0), (28, 0), (0, 28), (28, 28)]
        OVERLAP = 32

        ssc_path = Path(ssc_root)
        quantized_path = Path(quantized_root)

        all_ious = []

        for frame_id in EVAL_FRAMES:
            # Load S2 data (64x64x8)
            s2_file = quantized_path / 's2' / 'sequences' / '08' / f'{frame_id}.npy'
            if not s2_file.exists():
                continue
            s2_full = np.load(s2_file).astype(np.int64)

            # Load GT for comparison
            label_path = ssc_path / 'sequences' / '08' / 'voxels' / f'{frame_id}.label'
            if not label_path.exists():
                continue
            gt_labels = np.fromfile(label_path, dtype=np.uint16).reshape(256, 256, 32)
            gt_mapped = apply_learning_map(gt_labels)

            # Generate 4 sub-scenes autoregressively
            generated_sub_scenes = {}

            for sub_idx in range(4):
                # Extract S2 patch (36x36x8)
                x_s, y_s = S2_PATCH_POSITIONS[sub_idx]
                s2_patch = s2_full[x_s:x_s+36, y_s:y_s+36, :].copy()

                # Upsample S2 patch to 144x144x32
                s2_t = torch.from_numpy(s2_patch).long().unsqueeze(0).to(self.device)
                one_hot = F.one_hot(s2_t, num_classes=self.num_classes).float()
                one_hot = one_hot.permute(0, 4, 1, 2, 3)
                interpolated = F.interpolate(one_hot, size=SUB_SCENE_SIZE, mode='trilinear', align_corners=False)
                s2_cond = interpolated.argmax(dim=1).unsqueeze(1).byte()  # [1, 1, 144, 144, 32]

                # Create inpainting mask
                inpaint_np = np.zeros(SUB_SCENE_SIZE, dtype=np.int64)
                if sub_idx == 1:  # RIGHT edge from S3-0
                    s3_0 = generated_sub_scenes[0]
                    inpaint_np[:OVERLAP, :, :] = s3_0[-OVERLAP:, :, :]
                elif sub_idx == 2:  # BOTTOM edge from S3-0
                    s3_0 = generated_sub_scenes[0]
                    inpaint_np[:, -OVERLAP:, :] = s3_0[:, -OVERLAP:, :]
                elif sub_idx == 3:  # L-shape from S3-2 (X) + S3-1 (Y)
                    s3_1 = generated_sub_scenes[1]
                    s3_2 = generated_sub_scenes[2]
                    inpaint_np[:OVERLAP, :, :] = s3_2[-OVERLAP:, :, :]
                    inpaint_np[:, -OVERLAP:, :] = s3_1[:, -OVERLAP:, :]

                inpaint_mask = torch.from_numpy(inpaint_np).long().unsqueeze(0).unsqueeze(0).byte().to(self.device)

                # Dual conditioning
                cond = torch.cat([s2_cond, inpaint_mask], dim=1)

                # Generate sub-scene
                shape = (1, *SUB_SCENE_SIZE)
                if self.distributed:
                    generated = self.diffusion.module.p_sample_loop(shape, cond=cond, device=self.device)
                else:
                    generated = self.diffusion.p_sample_loop(shape, cond=cond, device=self.device)

                generated_sub_scenes[sub_idx] = generated[0].cpu().numpy()

            # Fuse sub-scenes (first seen wins)
            full_scene = np.zeros((256, 256, 32), dtype=np.int64)
            added = set()
            for sub_idx in range(4):
                sub_scene = generated_sub_scenes[sub_idx]
                x_start, y_start = SUB_SCENE_POSITIONS[sub_idx]
                for i in range(144):
                    for j in range(144):
                        for k in range(32):
                            x, y, z = x_start + i, y_start + j, k
                            if (x, y, z) not in added:
                                full_scene[x, y, z] = sub_scene[i, j, k]
                                added.add((x, y, z))

            # Compute mIoU
            metrics = compute_iou(full_scene, gt_mapped)
            all_ious.append(metrics['miou'])

        avg_miou = np.mean(all_ious) if all_ious else 0.0
        return avg_miou

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False, miou: float = None, is_best_miou: bool = False):
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
            'miou': miou,
            'best_miou': self.best_miou,
            'base_lr': self.base_lr,
            'scaled_lr': self.scaled_lr,
            'world_size': self.world_size,
        }

        # Save latest
        torch.save(checkpoint, self.output_dir / 'latest.pt')

        # Save milestone checkpoints
        milestones = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        if epoch in milestones or epoch % 500 == 0:
            torch.save(checkpoint, self.output_dir / f's3_epoch_{epoch:04d}.pt')
            print(f'  Milestone checkpoint saved: epoch {epoch}')

        # Save best by val_loss
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            torch.save(checkpoint, self.output_dir / f'best_val_{epoch:04d}.pt')
            print(f'  New best model saved! val_loss={val_loss:.4f} (best_val_{epoch:04d}.pt)')

        # Save best by mIoU
        if is_best_miou and miou is not None:
            torch.save(checkpoint, self.output_dir / 'best_miou.pt')
            torch.save(checkpoint, self.output_dir / f'best_miou_{epoch:04d}.pt')
            print(f'  New best mIoU model saved! mIoU={miou*100:.2f}% (best_miou_{epoch:04d}.pt)')

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if self.distributed:
            self.diffusion.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.diffusion.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_miou = checkpoint.get('best_miou', 0.0)
        self.start_epoch = checkpoint['epoch'] + 1

        if is_main_process(self.rank):
            print(f"Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
            print(f"  best_val_loss: {self.best_val_loss:.4f}, best_miou: {self.best_miou*100:.2f}%")
            if 'world_size' in checkpoint and checkpoint['world_size'] != self.world_size:
                print(f"  WARNING: World size changed from {checkpoint['world_size']} to {self.world_size}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 1000,
        log_every: int = 100,
        ssc_root: str = None,
        quantized_root: str = None,
    ):
        """Full training loop with mIoU evaluation."""
        if is_main_process(self.rank):
            print(f"[S3Trainer] Starting training from epoch {self.start_epoch} to {num_epochs}")
            print(f"[S3Trainer] Train samples: {len(train_dataloader.dataset)}")
            print(f"[S3Trainer] Val samples: {len(val_dataloader.dataset)}")
            if ssc_root and quantized_root:
                print(f"[S3Trainer] mIoU evaluation enabled with {len(EVAL_FRAMES)} frames")

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_loss = self.train_epoch(train_dataloader, epoch, log_every)
            val_loss = self.validate(val_dataloader)

            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            current_lr = self.optimizer.param_groups[0]['lr']

            # Evaluate mIoU if paths are provided
            miou = None
            is_best_miou = False
            if ssc_root and quantized_root and is_main_process(self.rank):
                print(f"  Evaluating mIoU on {len(EVAL_FRAMES)} frames...")
                miou = self.evaluate_miou(ssc_root, quantized_root)
                is_best_miou = miou > self.best_miou
                if is_best_miou:
                    self.best_miou = miou

            if is_main_process(self.rank):
                miou_str = f", mIoU={miou*100:.2f}%" if miou is not None else ""
                best_str = " (best)" if is_best else ""
                best_miou_str = " (best_miou)" if is_best_miou else ""
                print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}{miou_str}, lr={current_lr:.6f}{best_str}{best_miou_str}')

            self.save_checkpoint(epoch, train_loss, val_loss, is_best, miou=miou, is_best_miou=is_best_miou)

        if is_main_process(self.rank):
            print(f"[S3Trainer] Training complete! Best val_loss: {self.best_val_loss:.4f}, Best mIoU: {self.best_miou*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train S3 Pyramid Diffusion')
    parser.add_argument('--ssc-root', type=str,
                        default='datasets/dataset_SemanticKITTI_SSC',
                        help='SemanticKITTI SSC data root (full resolution)')
    parser.add_argument('--quantized-root', type=str,
                        default='datasets/SemanticKITTI_quantized',
                        help='Quantized data root (contains s2/)')
    parser.add_argument('--output-dir', type=str,
                        default='outputs/checkpoints/s3',
                        help='Output directory')
    # IMPORTANT: The original S3 training used batch_size=9, lr=0.006
    # These settings produced ~80GB GPU memory usage and 8502 iterations/epoch
    # To resume training with same settings: --batch-size 9 --lr 0.006
    parser.add_argument('--batch-size', type=int, default=9, help='Batch size per GPU (used: 9, ~80GB VRAM)')
    parser.add_argument('--epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--lr', type=float, default=0.006, help='Learning rate (used: 0.006)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device (single GPU mode)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--no-scale-lr', action='store_true',
                        help='Disable learning rate scaling with world size')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs when using LR scaling')

    # Every boolean flag also gets a hidden ``--no_<dest>`` twin, so a YAML config can
    # switch OFF a flag that defaults to on (configs/ -> gssc.utils.config_loader).
    from gssc.utils.config_loader import add_boolean_negations
    add_boolean_negations(parser)

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

    train_dataset = S3Dataset(
        ssc_root=args.ssc_root,
        quantized_root=args.quantized_root,
        sequences=train_sequences,
        augment=True,
    )

    val_dataset = S3Dataset(
        ssc_root=args.ssc_root,
        quantized_root=args.quantized_root,
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
    trainer = S3Trainer(
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
            ssc_root=args.ssc_root,
            quantized_root=args.quantized_root,
        )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
