"""
Training script for 3D Scene Completion with Multinomial Diffusion.

Phase 3 of the pipeline:
- Input: Sparse LiDAR voxels + BEV semantic map (GT during training)
- Output: Complete 3D semantic scene

Training configuration (adapted for SemanticKITTI):
- Resolution: 256×256×32 (SemanticKITTI native resolution)
- Batch size: 4 (reduced from 8 due to larger resolution)
- Learning rate: 1e-4
- Optimizer iterations: 100,000
- Diffusion timesteps: 100
- EMA decay: 0.9999

Note: Original paper used CarlaSC at 128×128×8. We scale up for SemanticKITTI.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# S1: Classifier-Free Guidance
try:
    from cfg_diffusion import CFGWrapper, SDEditSampler
except ImportError:
    pass

# S2: 2D→3D Lifting
try:
    from lifting import BEVTo3DLifter, LiftingModule
except ImportError:
    pass

# S3: DSKD for Scene Completion (pairwise feature similarity, per SCPNet)
# Reference: SCPNet CVPR 2023 - matches pairwise feature relationships, not KL divergence
try:
    from dskd import DSKDLoss3D
except ImportError:
    try:
        from gssc.models.dskd import DSKDLoss3D
    except ImportError:
        DSKDLoss3D = None  # Will be None if DSKD not available

# S3: Enhanced DSKD Dataset (separate Teacher/Student modes)
try:
    from s3_dskd_dataset import S3DSKDDataset, create_s3_dataloader
except ImportError:
    try:
        from gssc.data.semantickitti import S3DSKDDataset, create_s3_dataloader
    except ImportError:
        S3DSKDDataset = None
        create_s3_dataloader = None

# S4: MIMO (Multi-Input Multi-Output)
try:
    from mimo import BEVAugmenter, MIMOEnsembler, MIMOSceneCompletion
    from mimo_dataset import MIMODatasetWrapper, create_mimo_dataloader, mimo_collate_fn
    from mimo_scene_unet import MIMOSceneCompletionUNet, MIMOSceneCompletionUNetLite
except ImportError:
    try:
        from gssc.models.mimo import BEVAugmenter, MIMOEnsembler, MIMOSceneCompletion
        from gssc.models.mimo_dataset import (
            MIMODatasetWrapper,
            create_mimo_dataloader,
            mimo_collate_fn,
        )
        from gssc.models.mimo_scene_unet import MIMOSceneCompletionUNet, MIMOSceneCompletionUNetLite
    except ImportError:
        MIMOSceneCompletion = None
        BEVAugmenter = None
        MIMOEnsembler = None
        MIMODatasetWrapper = None
        create_mimo_dataloader = None
        mimo_collate_fn = None
        MIMOSceneCompletionUNet = None
        MIMOSceneCompletionUNetLite = None

# S5: Dense 3D CNN at coarse resolution
try:
    from dense_3d_cnn import Dense3DBottleneck, Dense3DEnhancer, DenseHallucinationModule
except ImportError:
    try:
        from gssc.models.dense_3d_cnn import (
            Dense3DBottleneck,
            Dense3DEnhancer,
            DenseHallucinationModule,
        )
    except ImportError:
        Dense3DBottleneck = None
        DenseHallucinationModule = None
        Dense3DEnhancer = None


class SSCMetrics:
    """
    SemanticKITTI Scene Completion evaluation metrics.

    Computes:
    - mIoU SSC: Mean IoU over semantic classes (1-19), excluding empty (0)
    - IoU Completion: Binary completion IoU (occupied vs empty)
    - Per-class IoU: IoU for each semantic class

    Following the official SemanticKITTI SSC benchmark evaluation.
    """

    def __init__(self, num_classes: int = 20, ignore_index: int = 255):
        """
        Args:
            num_classes: Number of classes (20 for SemanticKITTI: 0=empty + 19 semantic)
            ignore_index: Label to ignore in evaluation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Reset confusion matrix."""
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray, invalid_mask: np.ndarray = None):
        """
        Update confusion matrix with a batch.

        Args:
            pred: Predicted labels [N] or [H, W, D]
            target: Ground truth labels [N] or [H, W, D]
            invalid_mask: Optional mask for invalid voxels (1=invalid, 0=valid)
        """
        pred = pred.flatten()
        target = target.flatten()

        # Create valid mask
        valid = (target != self.ignore_index)
        if invalid_mask is not None:
            valid = valid & (invalid_mask.flatten() == 0)

        pred = pred[valid]
        target = target[valid]

        # Update confusion matrix
        idxs = (pred, target)
        np.add.at(self.conf_matrix, idxs, 1)

    def get_stats(self):
        """Get TP, FP, FN for each class."""
        conf = self.conf_matrix.copy()
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def get_iou(self):
        """
        Compute IoU metrics following SemanticKITTI SSC benchmark.

        Returns:
            dict with:
            - mIoU: Mean IoU over semantic classes (1-19)
            - iou_completion: Binary completion IoU (occupied vs empty)
            - class_iou: Per-class IoU dict
        """
        tp, fp, fn = self.get_stats()

        # Per-class IoU
        iou = tp / (tp + fp + fn + 1e-15)

        # mIoU SSC: Mean over semantic classes (1-19), excluding empty (0)
        semantic_classes = list(range(1, self.num_classes))
        miou = iou[semantic_classes].mean()

        # IoU Completion: Binary (occupied vs empty)
        # TP = correctly predicted occupied, FP = predicted occupied but empty
        # FN = predicted empty but occupied
        conf = self.conf_matrix
        tp_occupied = conf[1:, 1:].sum()  # Predicted occupied, GT occupied
        fp_occupied = conf[1:, 0].sum()   # Predicted occupied, GT empty
        fn_occupied = conf[0, 1:].sum()   # Predicted empty, GT occupied
        iou_completion = tp_occupied / (tp_occupied + fp_occupied + fn_occupied + 1e-15)

        # Precision and Recall
        precision = tp_occupied / (tp_occupied + fp_occupied + 1e-15)
        recall = tp_occupied / (tp_occupied + fn_occupied + 1e-15)

        # Per-class IoU dict (only semantic classes)
        class_iou = {i: iou[i] for i in semantic_classes}

        return {
            'mIoU': miou,
            'iou_completion': iou_completion,
            'precision': precision,
            'recall': recall,
            'class_iou': class_iou,
        }


# exp_1: Dense Conv3d for LiDAR
# Diffusion
from gssc.diffusion.multinomial import (
    MultinomialDiffusion3D,
    MultinomialDiffusion3DEMA,
    MultinomialDiffusion3DV2,
)

# exp_2: Sparse spconv for LiDAR
from gssc.models.s2d2_unet import SceneCompletionUNetSparse, SceneCompletionUNetSparseLite
from gssc.models.scene_unet import SceneCompletionUNet, SceneCompletionUNetLite, count_parameters

# V2: FiLM conditioning + multi-scale auxiliary BEV (no cascade)
from gssc.models.scene_unet_v2 import SceneCompletionUNetV2, V2ModelWrapper
from gssc.models.scene_unet_v3 import SceneCompletionUNetV3, V3ModelWrapper
from gssc.utils.compat import resolve_bev_from_base


class SemanticKITTI3DDataset(Dataset):
    """
    Dataset for 3D Scene Completion training on SemanticKITTI.

    Loads:
    - Complete 3D semantic voxels (ground truth): 256×256×32
    - Sparse LiDAR binary voxels (input): 256×256×32
    - BEV semantic map (conditioning, derived from GT during training): 256×256

    Native resolution: 256×256×32 (SemanticKITTI standard)
    """

    def __init__(
        self,
        data_root: str,
        sequences: list,
        augment: bool = False,
        use_rectified_labels: bool = False,
        waffleiron_root: str = None,  # B6: Path to WaffleIron BEV features
        lsk3d_3d_root: str = None,  # S18: Path to LSK3DNet sparse 3D predictions for SDEdit init
        densify_nn: bool = False,  # S25-S27: Precompute NN indices for sparse conditioning
    ):
        """
        Args:
            data_root: Path to SemanticKITTI SSC dataset
            sequences: List of sequence IDs to load
            augment: Whether to apply data augmentation
            use_rectified_labels: Use rectified labels (SCPNet protocol) (removes ghost trails
                                  from dynamic objects). Improves +10-25% on dynamic classes.
                                  Requires running tools/rectify_labels.py first.
            waffleiron_root: Path to precomputed WaffleIron BEV features (B6 experiment).
                             Expected format: {waffleiron_root}/{seq}/{frame_id}_waffleiron.npy
                             Each file should be (64, 256, 256) float16.
            lsk3d_3d_root: Path to LSK3DNet sparse 3D predictions for SDEdit initialization.
                           Expected format: {lsk3d_3d_root}/{seq}/{frame_id}_lsk3d_3d.npz
                           Each file has 'coords' (N,3) uint8 and 'probs' (N,20) float16.
            densify_nn: If True, precompute nearest-neighbor indices for NN densification.
        """
        self.data_root = Path(data_root)
        self.augment = augment
        self.use_rectified_labels = use_rectified_labels
        self.waffleiron_root = Path(waffleiron_root) if waffleiron_root else None
        self.lsk3d_3d_root = Path(lsk3d_3d_root) if lsk3d_3d_root else None
        self.densify_nn = densify_nn
        self.samples = []

        rectified_count = 0
        original_count = 0

        # Collect samples
        for seq in sequences:
            seq_path = self.data_root / "sequences" / seq / "voxels"
            if not seq_path.exists():
                logging.warning(f"Sequence {seq} not found at {seq_path}")
                continue

            # Find all .bin files (sparse LiDAR)
            bin_files = sorted(seq_path.glob("*.bin"))
            for bin_file in bin_files:
                frame_id = bin_file.stem

                # Try rectified labels first if enabled
                if use_rectified_labels:
                    rectified_file = seq_path / f"{frame_id}_rectified.label"
                    if rectified_file.exists():
                        self.samples.append({
                            'seq': seq,
                            'frame': frame_id,
                            'bin': bin_file,
                            'label': rectified_file,
                        })
                        rectified_count += 1
                        continue

                # Fall back to original labels
                label_file = seq_path / f"{frame_id}.label"
                if label_file.exists():
                    self.samples.append({
                        'seq': seq,
                        'frame': frame_id,
                        'bin': bin_file,
                        'label': label_file,
                    })
                    original_count += 1

        if use_rectified_labels:
            logging.info(f"Loaded {len(self.samples)} samples: {rectified_count} rectified, {original_count} original")
        else:
            logging.info(f"Loaded {len(self.samples)} samples from sequences {sequences}")

        # B6: Log WaffleIron feature status
        if self.waffleiron_root:
            logging.info(f"WaffleIron features enabled: {self.waffleiron_root}")

        # S18: Log LSK3DNet 3D feature status
        if self.lsk3d_3d_root:
            logging.info(f"LSK3DNet 3D features enabled for SDEdit init: {self.lsk3d_3d_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load sparse LiDAR voxels: 256×256×32
        lidar = self._load_bin(sample['bin'])  # [256, 256, 32] binary

        # Load complete semantic voxels: 256×256×32
        gt_scene = self._load_label(sample['label'])  # [256, 256, 32] uint8

        # Create BEV from GT (max semantic class along Z): 256×256
        # For training, we use GT BEV as conditioning
        bev = self._create_bev(gt_scene)  # [256, 256] uint8

        # Load LSK3DNet 3D sparse predictions BEFORE augmentation
        # so they receive the same spatial transforms as gt_scene
        lsk3d_dense = None
        if self.lsk3d_3d_root:
            lsk3d_path = self.lsk3d_3d_root / sample['seq'] / f"{sample['frame']}_lsk3d_3d.npz"
            if lsk3d_path.exists():
                lsk3d_data = np.load(str(lsk3d_path))
                coords = lsk3d_data['coords']  # [N, 3] uint8 (x, y, z voxel coords)
                probs = lsk3d_data['probs'].astype(np.float32)  # [N, 20] probabilities

                # Convert sparse to dense [20, 256, 256, 32]
                # Zeros for unobserved voxels (preserves sparsity for spconv)
                lsk3d_dense = np.zeros((20, 256, 256, 32), dtype=np.float32)

                # Fill in LSK3DNet predictions at observed voxels
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                for c in range(20):
                    lsk3d_dense[c, x, y, z] = probs[:, c]

        # Data augmentation (applied to all spatial data consistently)
        if self.augment:
            lidar, gt_scene, bev, lsk3d_dense = self._augment(lidar, gt_scene, bev, lsk3d_dense)

        # Convert to tensors
        lidar = torch.from_numpy(lidar).float().unsqueeze(0)  # [1, 256, 256, 32]
        gt_scene = torch.from_numpy(gt_scene).long()  # [256, 256, 32]
        bev = torch.from_numpy(bev).long()  # [256, 256]

        result = {
            'lidar': lidar,
            'gt_scene': gt_scene,
            'bev': bev,
            'seq': sample['seq'],
            'frame': sample['frame'],
        }

        # B6: Load WaffleIron BEV features if available
        if self.waffleiron_root:
            waffleiron_path = self.waffleiron_root / sample['seq'] / f"{sample['frame']}_waffleiron.npy"
            if waffleiron_path.exists():
                waffleiron_feat = np.load(str(waffleiron_path)).astype(np.float32)  # [64, 256, 256]
                result['waffleiron'] = torch.from_numpy(waffleiron_feat)
            else:
                # Fallback: zeros if file missing
                result['waffleiron'] = torch.zeros(64, 256, 256, dtype=torch.float32)
        # Note: Don't add waffleiron key if not using (None causes collate error)

        # Add LSK3DNet 3D probs (already augmented)
        if lsk3d_dense is not None:
            result['lsk3d_3d_probs'] = torch.from_numpy(lsk3d_dense)
        elif self.lsk3d_3d_root:
            # Fallback: zeros if file missing
            result['lsk3d_3d_probs'] = torch.zeros(20, 256, 256, 32, dtype=torch.float32)

        # S25-S27: Precompute NN indices for level-0 sparse conditioning densification
        # SubMConv3d preserves sparsity exactly, so level-0 occupancy == input occupancy.
        # EDT at 256×256×32 takes ~10ms on CPU — precompute here to avoid GPU overhead.
        if self.densify_nn:
            from scipy.ndimage import distance_transform_edt
            if lsk3d_dense is not None:
                occ_mask = np.abs(lsk3d_dense).sum(axis=0) > 1e-6  # [256, 256, 32]
            else:
                occ_mask = lidar.numpy() if isinstance(lidar, torch.Tensor) else lidar
                occ_mask = occ_mask.squeeze() > 0  # [256, 256, 32]
            _, nn_idx = distance_transform_edt(~occ_mask, return_indices=True)
            result['nn_indices'] = torch.from_numpy(nn_idx.astype(np.int16))  # [3, 256, 256, 32]

        return result

    def _load_bin(self, path: Path) -> np.ndarray:
        """Load sparse LiDAR voxels from .bin file."""
        # SemanticKITTI stores as packed binary
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            # Unpack bits to voxels
            voxels = np.unpackbits(data).reshape(256, 256, 32)
            return voxels.astype(np.float32)
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
            return np.zeros((256, 256, 32), dtype=np.float32)

    def _load_label(self, path: Path) -> np.ndarray:
        """Load semantic label voxels from .label file."""
        try:
            data = np.fromfile(str(path), dtype=np.uint16)
            # Reshape to voxel grid
            voxels = data.reshape(256, 256, 32)
            # Apply learning map to convert to 0-19 classes
            voxels = self._apply_learning_map(voxels)
            return voxels.astype(np.uint8)
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
            return np.zeros((256, 256, 32), dtype=np.uint8)

    def _apply_learning_map(self, voxels: np.ndarray) -> np.ndarray:
        """Apply SemanticKITTI learning map."""
        # Standard SemanticKITTI learning map (original label -> training label)
        learning_map = {
            0: 0,    # unlabeled
            1: 0,    # outlier -> unlabeled
            10: 1,   # car
            11: 2,   # bicycle
            13: 5,   # bus
            15: 3,   # motorcycle
            16: 5,   # on-rails -> bus
            18: 4,   # truck
            20: 5,   # other-vehicle -> bus
            30: 6,   # person
            31: 7,   # bicyclist
            32: 8,   # motorcyclist
            40: 9,   # road
            44: 10,  # parking
            48: 11,  # sidewalk
            49: 12,  # other-ground
            50: 13,  # building
            51: 14,  # fence
            52: 0,   # other-structure -> unlabeled
            60: 9,   # lane-marking -> road
            70: 15,  # vegetation
            71: 16,  # trunk
            72: 17,  # terrain
            80: 18,  # pole
            81: 19,  # traffic-sign
            99: 0,   # other-object -> unlabeled
            252: 1,  # moving-car -> car
            253: 7,  # moving-bicyclist -> bicyclist
            254: 6,  # moving-person -> person
            255: 8,  # moving-motorcyclist -> motorcyclist
            256: 5,  # moving-on-rails -> bus
            257: 5,  # moving-bus -> bus
            258: 4,  # moving-truck -> truck
            259: 5,  # moving-other-vehicle -> bus
        }

        result = np.zeros_like(voxels, dtype=np.uint8)
        for orig_label, new_label in learning_map.items():
            result[voxels == orig_label] = new_label

        return result

    def _create_bev(self, voxels: np.ndarray) -> np.ndarray:
        """Create BEV from 3D semantic voxels.

        For each (x, y) position, take the most common non-empty class along Z.
        """
        H, W, D = voxels.shape
        bev = np.zeros((H, W), dtype=np.uint8)

        for x in range(H):
            for y in range(W):
                column = voxels[x, y, :]
                non_empty = column[column > 0]
                if len(non_empty) > 0:
                    # Most common non-empty class
                    bev[x, y] = np.bincount(non_empty).argmax()

        return bev

    def _augment(self, lidar, gt_scene, bev, lsk3d=None):
        """Apply fast data augmentation for 3D scene completion.

        Uses only fast numpy operations (no scipy.ndimage.rotate which is too slow).
        Augmentations: flip, rot90, translation, dropout

        Args:
            lidar: [H, W, D] binary voxels
            gt_scene: [H, W, D] semantic labels
            bev: [H, W] BEV semantic map
            lsk3d: optional [C, H, W, D] multichannel probs (spatial axes 1,2)
        """
        # lsk3d is [C, H, W, D] so spatial X,Y axes are (1, 2) not (0, 1)
        lx, ly = 1, 2

        # 1. Random horizontal flip (X and Y axes independently)
        if np.random.random() > 0.5:
            lidar = np.flip(lidar, axis=0).copy()
            gt_scene = np.flip(gt_scene, axis=0).copy()
            bev = np.flip(bev, axis=0).copy()
            if lsk3d is not None:
                lsk3d = np.flip(lsk3d, axis=lx).copy()
        if np.random.random() > 0.5:
            lidar = np.flip(lidar, axis=1).copy()
            gt_scene = np.flip(gt_scene, axis=1).copy()
            bev = np.flip(bev, axis=1).copy()
            if lsk3d is not None:
                lsk3d = np.flip(lsk3d, axis=ly).copy()

        # 2. Random 90-degree rotation (0, 90, 180, or 270 degrees)
        k = np.random.randint(4)
        if k > 0:
            lidar = np.rot90(lidar, k, axes=(0, 1)).copy()
            gt_scene = np.rot90(gt_scene, k, axes=(0, 1)).copy()
            bev = np.rot90(bev, k, axes=(0, 1)).copy()
            if lsk3d is not None:
                lsk3d = np.rot90(lsk3d, k, axes=(lx, ly)).copy()

        # 3. Random translation (±1-2 voxels)
        if np.random.random() > 0.5:
            shift_x = np.random.randint(-2, 3)  # -2 to 2 voxels
            shift_y = np.random.randint(-2, 3)
            if shift_x != 0 or shift_y != 0:
                lidar = np.roll(lidar, shift_x, axis=0)
                lidar = np.roll(lidar, shift_y, axis=1)
                gt_scene = np.roll(gt_scene, shift_x, axis=0)
                gt_scene = np.roll(gt_scene, shift_y, axis=1)
                bev = np.roll(bev, shift_x, axis=0)
                bev = np.roll(bev, shift_y, axis=1)
                if lsk3d is not None:
                    lsk3d = np.roll(lsk3d, shift_x, axis=lx)
                    lsk3d = np.roll(lsk3d, shift_y, axis=ly)

        # 4. Random dropout on LiDAR observations (10-20% dropout rate)
        if np.random.random() > 0.5:
            dropout_rate = np.random.uniform(0.1, 0.2)
            dropout_mask = np.random.random(lidar.shape) > dropout_rate
            lidar = lidar * dropout_mask

        return lidar.copy(), gt_scene.copy(), bev.copy(), lsk3d.copy() if lsk3d is not None else None


class SemanticKITTI3DQuantizedDataset(Dataset):
    """
    Fast quantized dataset for 3D Scene Completion training on SemanticKITTI.

    Loads pre-processed numpy files from SemanticKITTI_3D/256/:
    - *_voxels.npy: Sparse LiDAR voxels (uint8, 256×256×32)
    - *_bev.npy: BEV semantic map (uint8, 256×256)
    - *_gt_scene.npy: Complete 3D semantic scene (uint8, 256×256×32, learning-mapped)

    This is ~10x faster than loading raw .bin/.label files.
    """

    def __init__(
        self,
        data_root: str,
        sequences: list,
        augment: bool = False,
    ):
        self.data_root = Path(data_root)
        self.augment = augment
        self.samples = []

        # Collect samples from quantized directory
        for seq in sequences:
            seq_path = self.data_root / seq
            if not seq_path.exists():
                logging.warning(f"Sequence {seq} not found at {seq_path}")
                continue

            # Find all voxels files
            voxel_files = sorted(seq_path.glob("*_voxels.npy"))
            for voxel_file in voxel_files:
                frame_id = voxel_file.stem.replace('_voxels', '')
                bev_file = seq_path / f"{frame_id}_bev.npy"
                gt_file = seq_path / f"{frame_id}_gt_scene.npy"

                if bev_file.exists() and gt_file.exists():
                    self.samples.append({
                        'seq': seq,
                        'frame': frame_id,
                        'voxels': voxel_file,
                        'bev': bev_file,
                        'gt_scene': gt_file,
                    })

        logging.info(f"Loaded {len(self.samples)} quantized samples from sequences {sequences}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load pre-processed numpy arrays (very fast)
        lidar = np.load(sample['voxels']).astype(np.float32)  # [256, 256, 32]
        bev = np.load(sample['bev'])  # [256, 256] uint8
        gt_scene = np.load(sample['gt_scene'])  # [256, 256, 32] uint8

        # Data augmentation
        if self.augment:
            lidar, gt_scene, bev = self._augment(lidar, gt_scene, bev)

        # Convert to tensors
        lidar = torch.from_numpy(lidar).float().unsqueeze(0)  # [1, 256, 256, 32]
        gt_scene = torch.from_numpy(gt_scene.copy()).long()  # [256, 256, 32]
        bev = torch.from_numpy(bev.copy()).long()  # [256, 256]

        return {
            'lidar': lidar,
            'gt_scene': gt_scene,
            'bev': bev,
            'seq': sample['seq'],
            'frame': sample['frame'],
        }

    def _augment(self, lidar, gt_scene, bev):
        """Apply fast data augmentation for 3D scene completion.

        Uses only fast numpy operations (no scipy.ndimage.rotate which is too slow).
        Augmentations: flip, rot90, translation, dropout
        """
        # 1. Random horizontal flip (X and Y axes independently)
        if np.random.random() > 0.5:
            lidar = np.flip(lidar, axis=0).copy()
            gt_scene = np.flip(gt_scene, axis=0).copy()
            bev = np.flip(bev, axis=0).copy()
        if np.random.random() > 0.5:
            lidar = np.flip(lidar, axis=1).copy()
            gt_scene = np.flip(gt_scene, axis=1).copy()
            bev = np.flip(bev, axis=1).copy()

        # 2. Random 90-degree rotation (0, 90, 180, or 270 degrees)
        k = np.random.randint(4)
        if k > 0:
            lidar = np.rot90(lidar, k, axes=(0, 1)).copy()
            gt_scene = np.rot90(gt_scene, k, axes=(0, 1)).copy()
            bev = np.rot90(bev, k, axes=(0, 1)).copy()

        # 3. Random translation (±1-2 voxels)
        if np.random.random() > 0.5:
            shift_x = np.random.randint(-2, 3)  # -2 to 2 voxels
            shift_y = np.random.randint(-2, 3)
            if shift_x != 0 or shift_y != 0:
                lidar = np.roll(lidar, shift_x, axis=0)
                lidar = np.roll(lidar, shift_y, axis=1)
                gt_scene = np.roll(gt_scene, shift_x, axis=0)
                gt_scene = np.roll(gt_scene, shift_y, axis=1)
                bev = np.roll(bev, shift_x, axis=0)
                bev = np.roll(bev, shift_y, axis=1)

        # 4. Random dropout on LiDAR observations (10-20% dropout rate)
        if np.random.random() > 0.5:
            dropout_rate = np.random.uniform(0.1, 0.2)
            dropout_mask = np.random.random(lidar.shape) > dropout_rate
            lidar = lidar * dropout_mask

        return lidar.copy(), gt_scene.copy(), bev.copy()


class SceneCompletionTrainer:
    """Trainer for 3D Scene Completion model."""

    def __init__(
        self,
        config: dict,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        self.logger.info(f"Config: {config}")

        # Create model
        self.model = self._create_model()
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")

        # Create diffusion
        diffusion_version = config.get('diffusion_version', 'v1')
        if diffusion_version == 'v2':
            self.logger.info("Using MultinomialDiffusion3DV2 (enhanced loss)")
            self.diffusion = MultinomialDiffusion3DV2(
                num_classes=config['num_classes'],
                num_timesteps=config['num_timesteps'],
                beta_min=config.get('beta_min', 0.0001),
                beta_max=config.get('beta_max', 0.1),
                focal_gamma=config.get('focal_gamma', 2.0),
                class_0_weight=config.get('class_0_weight', 0.02),
                occupied_weight=config.get('occupied_weight', 10.0),
                lovasz_weight=config.get('lovasz_weight', 0.3),
                obs_weight_factor=config.get('obs_weight_factor', 2.0),
                auxiliary_loss_weight=config.get('auxiliary_loss_weight', 0.0),
                completion_weight=config.get('completion_weight', 0.0),
                loss_type=config.get('loss_type', 'kl'),
            ).to(device)
        elif diffusion_version == 'gaussian':
            from gssc.models.gaussian_diffusion_3d import GaussianDiffusion3D
            self.logger.info("Using GaussianDiffusion3D (continuous, DiffSSC-style)")
            self.diffusion = GaussianDiffusion3D(
                num_channels=21,  # 1 occ_logit + 20 sem_logits
                num_timesteps=config['num_timesteps'],
                logit_scale=config.get('logit_scale', 5.0),
                sigma_occ=config.get('sigma_occ', 1.0),
                sigma_sem=config.get('sigma_sem', 1.0),
                lambda_p=config.get('lambda_p', 5.0),
                lambda_s=config.get('lambda_s', 4.0),
                use_skewness_reg=config.get('use_skewness_reg', False),
                anisotropic=config.get('anisotropic', False),
                beta_schedule=config.get('beta_schedule', 'linear_diffssc'),
                num_classes=config['num_classes'],
            ).to(device)
        elif diffusion_version == 'gaussian_ve':
            from gssc.models.gaussian_diffusion_ve import GaussianDiffusionVE
            self.logger.info("Using GaussianDiffusionVE (VE diffusion, soft probs, no amplification)")
            self.diffusion = GaussianDiffusionVE(
                num_classes=config['num_classes'],  # 20 channels (soft probs)
                num_timesteps=config['num_timesteps'],
                sigma_min=config.get('sigma_min', 0.01),
                sigma_max=config.get('sigma_max', 1.0),
                sigma_schedule=config.get('sigma_schedule', 'linear'),
                label_smoothing=config.get('label_smoothing', 0.1),
                lambda_reg=config.get('lambda_reg', 5.0),
            ).to(device)
            self.logger.info(f"    sigma: [{config.get('sigma_min', 0.01)}, {config.get('sigma_max', 1.0)}] ({config.get('sigma_schedule', 'linear')})")
            self.logger.info(f"    label_smoothing: {config.get('label_smoothing', 0.1)}")
        elif diffusion_version == 'gaussian_logit':
            from gssc.models.gaussian_diffusion_logit import GaussianDiffusionLogit
            logit_scale = config.get('logit_scale', 3.0)
            sigma_max = config.get('sigma_max', 80.0)
            self.logger.info("Using GaussianDiffusionLogit (VE + EDM preconditioning)")
            self.diffusion = GaussianDiffusionLogit(
                num_classes=config['num_classes'],
                num_timesteps=config['num_timesteps'],
                sigma_min=config.get('sigma_min', 0.01),
                sigma_max=sigma_max,
                sigma_schedule=config.get('sigma_schedule', 'cosine'),
                logit_scale=logit_scale,
                sigma_data=logit_scale,  # EDM: sigma_data = characteristic data scale
            ).to(device)
            self.logger.info(f"    sigma: [{config.get('sigma_min', 0.01)}, {sigma_max}] ({config.get('sigma_schedule', 'cosine')})")
            self.logger.info(f"    logit_scale: {logit_scale}, sigma_data: {logit_scale} (EDM)")
            self.logger.info(f"    EDM c_in at σ_max={sigma_max}: {1.0 / (sigma_max**2 + logit_scale**2)**0.5:.4f}")
        elif diffusion_version == 'gaussian_vp':
            from gssc.models.gaussian_diffusion_3d import GaussianDiffusion3D
            logit_scale = config.get('logit_scale', 5.0)
            beta_schedule = config.get('beta_schedule', 'linear_diffssc')
            lambda_p = config.get('lambda_p', 5.0)
            lambda_s = config.get('lambda_s', 4.0)
            class_0_wt = config.get('class_0_weight', 0.02)
            occupied_wt = config.get('occupied_weight', 10.0)
            lovasz_wt = config.get('lovasz_weight', 0.3)
            obs_wt = config.get('obs_weight_factor', 2.0)
            self.logger.info("Using GaussianDiffusion3D (VP DDPM, 20ch logit encoding)")
            self.diffusion = GaussianDiffusion3D(
                num_channels=20,
                num_timesteps=config['num_timesteps'],
                logit_scale=logit_scale,
                encoding='logit_20ch',
                beta_schedule=beta_schedule,
                num_classes=config['num_classes'],
                lambda_p=lambda_p,
                lambda_s=lambda_s,
                use_skewness_reg=config.get('use_skewness_reg', False),
                class_0_weight=class_0_wt,
                occupied_weight=occupied_wt,
                lovasz_weight=lovasz_wt,
                obs_weight_factor=obs_wt,
            ).to(device)
            self.logger.info(f"    beta_schedule: {beta_schedule}, logit_scale: {logit_scale}")
            self.logger.info(f"    lambda_p: {lambda_p}, lambda_s: {lambda_s}")
            self.logger.info(f"    class_0_weight: {class_0_wt}, occupied_weight: {occupied_wt}")
            self.logger.info(f"    lovasz_weight: {lovasz_wt}, obs_weight_factor: {obs_wt}")
            self.logger.info("    encoding: logit_20ch (20ch, mean-centered)")
        elif diffusion_version == 'factored':
            from gssc.models.factored_diffusion_3d import FactoredDiffusion3D
            self.logger.info("Using FactoredDiffusion3D (K=2 occ + K=20 sem)")
            self.diffusion = FactoredDiffusion3D(
                num_classes=config['num_classes'],
                num_timesteps=config['num_timesteps'],
                beta_max_occ=config.get('beta_max_occ', 0.15),
                beta_max_sem=config.get('beta_max_sem', 0.05),
            ).to(device)
        else:
            self.logger.info("Using MultinomialDiffusion3D (v1)")
            self.diffusion = MultinomialDiffusion3D(
                num_classes=config['num_classes'],
                num_timesteps=config['num_timesteps'],
                loss_type=config.get('loss_type', 'ce'),
            ).to(device)

        # Create EMA
        self.ema = MultinomialDiffusion3DEMA(self.model, decay=config.get('ema_decay', 0.9999))

        # S5: Enable PaSCo's SPCDense3Dv2 at UNet bottleneck
        # Reference: PaSCo (CVPR 2024) applies Dense3D at bottleneck for hallucination
        # This integrates SPCDense3Dv2 INTO the model (not as post-processing)
        self.use_dense_3d = config.get('use_dense_3d', False)
        if self.use_dense_3d:
            # Check if model supports dense3d bottleneck
            if hasattr(self.model, 'enable_dense3d_bottleneck'):
                dense3d_dropout = config.get('dense_3d_dropout', 0.1)
                self.model.enable_dense3d_bottleneck(dropout=dense3d_dropout)
                self.logger.info("S5: Enabled PaSCo's SPCDense3Dv2 at UNet bottleneck")
                self.logger.info(f"    Dropout: {dense3d_dropout}")
                self.logger.info("    Applied at 1:16 resolution (16×16×2 for 256×256×32)")
                # Reinitialize EMA to include newly added dense3d_bottleneck parameters
                self.ema = MultinomialDiffusion3DEMA(self.model, decay=config.get('ema_decay', 0.9999))
                self.logger.info("    EMA reinitialized to include dense3d_bottleneck parameters")
            else:
                self.logger.warning(f"S5: Model {type(self.model).__name__} doesn't support dense3d bottleneck")

        # V2: Initialize auxiliary BEV loss components (FiLM + multi-scale aux BEV)
        self.use_aux_bev = config.get('aux_bev', False) and config.get('model_type', 'full') in ('v2_full', 'v2_lite', 'v3_full', 'v3_ablation', 'v3_coarse2fine', 'v3_c2f_ablation', 'v4_continuous', 'v4_factored', 'v5_ve')
        self.aux_focal = None
        self.aux_lovasz = None
        if self.use_aux_bev:
            from gssc.models.bev_sparse_bev_net import FocalLoss, LovaszSoftmax
            # Class weights: downweight empty (class 0)
            class_weights = torch.ones(config['num_classes'], device=device)
            class_weights[0] = config.get('aux_bev_class_0_weight', 0.02)
            self.aux_focal = FocalLoss(
                gamma=2.0, ignore_index=-100, class_weights=class_weights,
            ).to(device)
            self.aux_lovasz = LovaszSoftmax(ignore_index=0).to(device)  # Lovász optimizes mIoU which excludes class 0
            self.aux_lovasz_weight = config.get('aux_bev_lovasz_weight', 0.3)
            # Fixed weight for aux BEV loss (replaces broken JS3C-Net uncertainty weighting)
            self.aux_bev_weight = config.get('aux_bev_weight', 0.1)
            self.logger.info(f"V2: Auxiliary BEV loss initialized (Focal+Lovász, fixed weight={self.aux_bev_weight})")
            self.logger.info(f"    class_0_weight={class_weights[0]:.3f}, lovász_weight={self.aux_lovasz_weight}")

        # S43: 3D Lovász loss on SSC x_0 predictions
        self.ssc_lovasz = None
        self.ssc_lovasz_weight = config.get('ssc_lovasz_weight', 0.0)
        if self.ssc_lovasz_weight > 0:
            from gssc.losses.lovasz import LovaszSoftmax3D
            self.ssc_lovasz = LovaszSoftmax3D(ignore_index=0).to(device)
            self.logger.info(f"S43: 3D Lovász loss enabled (weight={self.ssc_lovasz_weight})")

        # S48: Direct prediction mode — bypass diffusion, use CE+Lovász directly
        self.direct_prediction = config.get('direct_prediction', False)
        self.dp_use_lifted_init = config.get('dp_lifted_init', False)
        self.dp_use_spsr = config.get('dp_spsr', False)
        if self.direct_prediction:
            # CE loss with class weighting (same as diffusion: low weight for class 0)
            dp_weights = torch.ones(config['num_classes'], device=device)
            dp_weights[0] = config.get('class_0_weight', 0.02)
            self.dp_ce_weights = dp_weights
            # Lovász 3D for direct mIoU optimization
            from gssc.losses.lovasz import LovaszSoftmax3D
            self.dp_lovasz = LovaszSoftmax3D(ignore_index=0).to(device)
            self.dp_lovasz_weight = 0.3  # Same as diffusion lovász weight
            self.logger.info("S48: Direct prediction mode enabled (no diffusion)")
            self.logger.info(f"    Loss: CE(class_0={dp_weights[0]:.3f}) + Lovász(weight={self.dp_lovasz_weight})")

            # Lifted BEV initialization (S3CNet-inspired: BEV → 3D via height priors)
            if self.dp_use_lifted_init:
                from gssc.models.lifting import BEVTo3DLifter
                self.dp_lifter = BEVTo3DLifter(num_classes=config['num_classes'], num_z=32).to(device)
                self.logger.info("    Init: Lifted BEV (S3CNet height priors) instead of uniform 1/K")
            else:
                self.dp_lifter = None

            # SPSR post-processing at eval time
            if self.dp_use_spsr:
                from gssc.models.spatial_propagation import SpatialPropagationRefinement
                self.dp_spsr = SpatialPropagationRefinement(
                    num_classes=config['num_classes'], num_iterations=3,
                    confidence_threshold=0.5, preserve_lidar=True,
                ).to(device)
                self.logger.info("    Eval: SPSR post-processing enabled (3 iterations)")

        # S44: Height-pool BEV auxiliary loss (no BEV conditioning, but BEV as aux target)
        self.use_hp_bev_aux = False
        self.hp_bev_aux_weight = config.get('aux_bev_weight', 0.1)
        if config.get('hp_bev_aux', False):
            from gssc.models.bev_sparse_bev_net import FocalLoss, LovaszSoftmax
            hp_class_weights = torch.ones(config['num_classes'], device=device)
            hp_class_weights[0] = config.get('aux_bev_class_0_weight', 0.02)
            self.hp_bev_focal = FocalLoss(gamma=2.0, ignore_index=-100, class_weights=hp_class_weights).to(device)
            self.hp_bev_lovasz = LovaszSoftmax(ignore_index=0).to(device)
            self.hp_bev_lovasz_weight = config.get('aux_bev_lovasz_weight', 0.3)
            self.use_hp_bev_aux = True
            self.logger.info(f"S44: Height-pool BEV aux loss enabled (weight={self.hp_bev_aux_weight}, class_0={hp_class_weights[0]:.3f}, lovász={self.hp_bev_lovasz_weight})")

        # S40v2/S41: Initialize BEV model for soft/E2E BEV conditioning
        self.bev_model = None
        self.use_soft_bev = config.get('soft_bev', False)
        self.use_e2e_bev = config.get('use_e2e_bev', False)

        if self.use_soft_bev or self.use_e2e_bev:
            bev_ckpt_path = config.get('bev_checkpoint')
            if bev_ckpt_path is None or not os.path.exists(bev_ckpt_path):
                raise ValueError(f"--bev_checkpoint required for soft_bev/e2e_bev, got: {bev_ckpt_path}")
            from gssc.models.bev_sparse_bev_net import FullSparseBEVNet_Deeper
            bev_ckpt = torch.load(bev_ckpt_path, map_location=device, weights_only=False)
            bev_config = bev_ckpt.get('config', {})
            bev_in_channels = bev_config.get('lsk3d_in_channels', 20)
            self.bev_model = FullSparseBEVNet_Deeper(
                num_classes=config['num_classes'],
                in_channels=bev_in_channels,
            ).to(device)
            self.bev_model.load_state_dict(bev_ckpt['model_state_dict'])
            bev_params = sum(p.numel() for p in self.bev_model.parameters())

            if self.use_e2e_bev:
                self.bev_model.eval()  # Keep eval mode: BN uses pretrained running stats (stable with batch_size=1)
                # Gradients still flow through eval-mode BN — only stats tracking and Dropout are affected
                self.logger.info(f"S41: Loaded E2E BEV model from {bev_ckpt_path} ({bev_params:,} params, TRAINABLE, BN frozen)")
                self.logger.info(f"     in_channels={bev_in_channels}, lr_factor={config.get('bev_lr_factor', 0.1)}")
            else:
                self.bev_model.eval()
                for p in self.bev_model.parameters():
                    p.requires_grad = False
                self.logger.info(f"S40v2: Loaded frozen BEV model from {bev_ckpt_path} ({bev_params:,} params, FROZEN)")
                self.logger.info(f"       in_channels={bev_in_channels}, soft BEV (detached)")

        # E2E BEV loss (for S41)
        self.e2e_bev_weight = config.get('e2e_bev_weight', 0.5)

        if self.bev_model is not None and not config.get('use_lsk3d', False):
            self.logger.warning("--soft_bev/--use_e2e_bev requires --use_lsk3d (20ch features). "
                                "BEV model loaded but will NOT be used with 1ch binary lidar!")

        # Create optimizer (all model params including dense3d if enabled)
        params_to_optimize = list(self.model.parameters())

        if self.use_e2e_bev and self.bev_model is not None:
            # S41: Two param groups — SSC at full LR, BEV at reduced LR
            bev_lr_factor = config.get('bev_lr_factor', 0.1)
            self.optimizer = AdamW([
                {'params': params_to_optimize, 'lr': config['lr']},
                {'params': list(self.bev_model.parameters()), 'lr': config['lr'] * bev_lr_factor},
            ], weight_decay=config.get('weight_decay', 0.01))
            self.logger.info(f"S41: Optimizer with 2 param groups: SSC lr={config['lr']}, BEV lr={config['lr'] * bev_lr_factor}")
        else:
            self.optimizer = AdamW(
                params_to_optimize,
                lr=config['lr'],
                weight_decay=config.get('weight_decay', 0.01),
            )

        # Create scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_iterations'],
            eta_min=config['lr'] * 0.01,
        )

        # S2: Initialize lifter for 2D→3D lifting (if enabled)
        self.lifter = None
        if config.get('use_lifting', False):
            from gssc.models.lifting import LiftingModule
            self.lifter = LiftingModule(
                num_classes=config['num_classes'],
                num_z=32,  # SemanticKITTI uses 32 z-slices
                feature_dim=config.get('lifting_feature_dim', 64),
                learnable=config.get('learnable_lifting', False),
            ).to(device)
            # Add lifter encoder params to optimizer
            lifter_params = list(self.lifter.parameters())
            if lifter_params:
                self.optimizer.add_param_group({'params': lifter_params})
                self.logger.info(f"S2: Added lifter to optimizer ({sum(p.numel() for p in lifter_params)} params)")
            self.logger.info("S2: Initialized LiftingModule for 2D→3D lifting")

        # S3: Initialize Enhanced DSKD for knowledge distillation
        # New Strategy: Teacher (GT BEV + Multi-frame) → Student (Pred BEV + Single-frame)
        self.use_dskd = config.get('use_dskd', False)
        self.s3_mode = config.get('s3_mode', None)  # 'teacher', 'intermediate', or 'student'
        self.teacher_model = None
        self.dskd_temperature = config.get('dskd_temperature', 4.0)
        self.gt_bev_prob = config.get('gt_bev_prob', 0.0)  # BEV mixing for student mode
        self.kd_type = config.get('kd_type', 'output')  # 'dskd' (pairwise features) or 'output' (soft labels)

        if self.s3_mode == 'teacher':
            # Teacher mode: GT BEV + Multi-frame LiDAR (no DSKD loss, just train normally)
            self.logger.info("S3 DSKD: TEACHER mode - training with GT BEV + multi-frame LiDAR")
            self.logger.info("         Multi-frame voxels provide 2x density for better scene understanding")

        elif self.s3_mode == 'intermediate':
            # Intermediate mode: GT BEV + Single-frame LiDAR
            # Bridges the multi-frame → single-frame density gap
            self.logger.info("S3 DSKD: INTERMEDIATE mode - training with GT BEV + single-frame LiDAR")
            self.logger.info("         Bridges multi-frame → single-frame gap for curriculum distillation")

        # Initialize DSKD loss (pairwise feature similarity, per SCPNet)
        self.dskd_loss = None

        # IMPORTANT: Enable lifted_features BEFORE creating teacher copy (for DSKD)
        # This ensures the teacher model has the same architecture as the checkpoint
        if config.get('use_lifting', False) or config.get('use_cfg', False):
            if hasattr(self.model, 'enable_lifted_features'):
                feature_dim = config.get('lifted_feature_dim', 64)
                self.model.enable_lifted_features(feature_dim=feature_dim)
                # Add newly created lifted_embed params to optimizer
                if hasattr(self.model, 'lifted_embed') and self.model.lifted_embed is not None:
                    self.optimizer.add_param_group({'params': list(self.model.lifted_embed.parameters())})
                    self.logger.info(f"S2: Added lifted_embed to optimizer ({sum(p.numel() for p in self.model.lifted_embed.parameters())} params)")
                self.logger.info("S2: Initialized LiftingModule for 2D→3D lifting")
                self._lifted_features_enabled = True
            else:
                self._lifted_features_enabled = False
        else:
            self._lifted_features_enabled = False

        if (self.s3_mode == 'student' or self.s3_mode == 'intermediate') and self.use_dskd:
            # Student/Intermediate mode: Use KD from Teacher
            teacher_ckpt = config.get('teacher_checkpoint')
            if teacher_ckpt and os.path.exists(teacher_ckpt):
                import copy
                ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
                # S43: Teacher may have different input channels than student
                # (e.g., teacher=20ch LSK3D, student=30ch LSK3D+geom)
                # Detect teacher's lidar_in_channels from checkpoint stem weight
                teacher_lidar_ch = None
                # Use exact key match to avoid matching stem_bn.weight (BatchNorm1d shape [ch])
                stem_key = 'lidar_encoder.stem.weight'
                if stem_key in ckpt['model_state_dict']:
                    teacher_lidar_ch = ckpt['model_state_dict'][stem_key].shape[-1]  # in_channels (last dim for spconv)
                # Get student's lidar input channels from model
                student_lidar_ch = None
                student_sd = self.model.state_dict()
                if stem_key in student_sd:
                    student_lidar_ch = student_sd[stem_key].shape[-1]  # last dim for spconv
                if teacher_lidar_ch is not None and student_lidar_ch is not None and teacher_lidar_ch != student_lidar_ch:
                    # Create teacher with its own architecture (matching checkpoint)
                    self.logger.info(f"S43 KD: Teacher has {teacher_lidar_ch}ch input, student has {student_lidar_ch}ch")
                    self.teacher_lidar_ch = teacher_lidar_ch
                    teacher_model_class = type(self.model)
                    # Create teacher with matching architecture. These args must match
                    # the checkpoint's training config. Currently hardcoded to S23 teacher
                    # defaults. If using a different teacher architecture, update these.
                    self.teacher_model = teacher_model_class(
                        num_classes=config['num_classes'],
                        base_channels=32,
                        time_emb_dim=128,
                        lidar_base_channels=16,
                        lidar_out_channels=32,
                        lidar_in_channels=teacher_lidar_ch,
                        no_bev=False,  # Teacher (S23) always trained with BEV
                    ).to(self.device)
                else:
                    self.teacher_model = copy.deepcopy(self.model)
                    self.teacher_lidar_ch = None
                # Use non-strict loading: teacher checkpoint may lack new modules
                # (e.g., geom_bev_embed added in student but absent in teacher)
                missing, unexpected = self.teacher_model.load_state_dict(
                    ckpt['model_state_dict'], strict=False)
                if missing:
                    self.logger.info(f"Teacher load: {len(missing)} missing keys (new student modules): "
                                    f"{missing[:3]}...")
                if unexpected:
                    self.logger.warning(f"Teacher load: {len(unexpected)} unexpected keys: "
                                        f"{unexpected[:3]}...")
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                self.teacher_model.eval()

                mode_name = "INTERMEDIATE" if self.s3_mode == 'intermediate' else "STUDENT"
                kd_type = config.get('kd_type', 'output')
                kd_temp = config.get('dskd_temperature', 4.0)
                kd_weight = config.get('dskd_weight', 0.5)

                if kd_type == 'output':
                    # Output-level KD (Hinton-style soft labels)
                    self.logger.info(f"S3 KD: {mode_name} mode - loaded teacher from {teacher_ckpt}")
                    self.logger.info("       Type: OUTPUT-LEVEL (soft labels) - more robust for multi→single frame")
                    self.logger.info(f"       Weight={kd_weight}, Temperature={kd_temp}")
                    if self.s3_mode == 'intermediate':
                        self.logger.info("       Teacher: multi-frame lidar, Student: single-frame lidar")
                else:
                    # DSKD (pairwise feature similarity)
                    if DSKDLoss3D is not None:
                        self.dskd_loss = DSKDLoss3D(
                            normalize=True,
                            temperature=kd_temp,
                            use_occupancy_mask=True,
                            max_samples=4096,
                        )
                        self.logger.info(f"S3 KD: {mode_name} mode - loaded teacher from {teacher_ckpt}")
                        self.logger.info("       Type: DSKD (pairwise features, occupancy-masked)")
                        self.logger.info(f"       Weight={kd_weight}, Temperature={kd_temp}")
                        if self.s3_mode == 'intermediate':
                            self.logger.info("       Teacher: multi-frame lidar, Student: single-frame lidar")
                    else:
                        self.logger.warning("S3 KD: DSKDLoss3D not available - falling back to output-level KD")
                        self.kd_type = 'output'
            else:
                raise ValueError(f"S3 {self.s3_mode} mode with KD requires --teacher_checkpoint to be specified")

        elif self.use_dskd:
            # Legacy mode: simple DSKD without s3_mode
            self.logger.warning("S3: Legacy DSKD mode (consider using --s3_mode teacher/student)")
            teacher_ckpt = config.get('teacher_checkpoint')
            if teacher_ckpt and os.path.exists(teacher_ckpt):
                import copy
                self.teacher_model = copy.deepcopy(self.model)
                ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
                self.teacher_model.load_state_dict(ckpt['model_state_dict'])
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                self.teacher_model.eval()
                # Initialize DSKD loss (SCPNet protocol)
                if DSKDLoss3D is not None:
                    dskd_temp = config.get('dskd_temperature', 1.0)
                    self.dskd_loss = DSKDLoss3D(
                        normalize=True,
                        temperature=dskd_temp,
                        use_occupancy_mask=True,
                        max_samples=4096,
                    )
                self.logger.info(f"S3: Loaded teacher model from {teacher_ckpt}")
            else:
                import copy
                self.teacher_model = copy.deepcopy(self.model)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                self.teacher_model.eval()
                if DSKDLoss3D is not None:
                    dskd_temp = config.get('dskd_temperature', 1.0)
                    self.dskd_loss = DSKDLoss3D(
                        normalize=True,
                        temperature=dskd_temp,
                        use_occupancy_mask=True,
                        max_samples=4096,
                    )
                self.logger.warning("S3: No teacher checkpoint - using frozen initial model")

        # S4: Initialize MIMO wrapper (if enabled)
        # Reference: PaSCo (CVPR 2024) Section 3.2.1-3.2.2
        # Key hyperparameters from PaSCo:
        # - n_subnets=2 (default)
        # - scale_range=0.0 (disabled)
        # - ensemble_method='mean_probs' (average probabilities, not logits)
        # - max_angle=30.0 (from paper)
        # - Training: Different samples per subnet (dataset-level MIMO)
        # - Inference: Same sample with different augmentations (model-level MIMO)
        self.use_mimo = config.get('use_mimo', False)
        self.mimo_wrapper = None
        self.bev_augmenter = None
        self.mimo_use_training_mode = config.get('mimo_use_training_mode', False)
        if self.use_mimo and MIMOSceneCompletion is not None:
            # Create MIMO wrapper with PaSCo-style unified 3D augmentation
            # CRITICAL: Both BEV and LiDAR must be transformed with same T matrix
            self.mimo_wrapper = MIMOSceneCompletion(
                base_model=self.model,
                num_subnets=config.get('mimo_num_subnets', 2),  # PaSCo default: 2
                num_classes=config['num_classes'],
                aug_types=config.get('mimo_aug_types', ['identity', 'noise', 'dropout']),
                ensemble_method=config.get('mimo_ensemble', 'mean_probs'),  # PaSCo default: mean_probs
                use_geometric_aug=True,  # Use PaSCo-style geometric augmentation
                use_unified_3d_aug=True,  # CRITICAL: Transform BOTH BEV and LiDAR with same T
                max_angle=config.get('mimo_max_angle', 30.0),  # PaSCo paper: 30°
                scale_range=config.get('mimo_scale_range', 0.0),  # PaSCo default: 0 (disabled)
            )
            self.logger.info(f"S4: Initialized MIMO with {config.get('mimo_num_subnets', 2)} subnets (PaSCo-style)")
            self.logger.info(f"    Ensemble method: {config.get('mimo_ensemble', 'mean_probs')} (avg probs after softmax)")
            self.logger.info("    Unified 3D augmentation: BEV and LiDAR transformed with same T")
            self.logger.info(f"    Max angle: {config.get('mimo_max_angle', 30.0)}°, Scale range: {config.get('mimo_scale_range', 0.0)}")
            if self.mimo_use_training_mode:
                self.logger.info("    Training mode: Dataset-level MIMO (different samples per subnet)")

        # S1/S2: Reinitialize EMA if lifted_features was enabled earlier (for DSKD compatibility)
        # Note: enable_lifted_features was already called before DSKD setup to ensure teacher model has same architecture
        if getattr(self, '_lifted_features_enabled', False):
            # Reinitialize EMA to include newly added lifted_embed parameter
            self.ema = MultinomialDiffusion3DEMA(self.model, decay=config.get('ema_decay', 0.9999))
            self.logger.info("EMA reinitialized to include lifted_embed parameter")

        # S5: Already initialized above (before optimizer)
        # (Dense3DEnhancer was moved earlier so its params are in optimizer)

        # Create datasets
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_miou = 0.0

    def _sample_train_timesteps(self, B: int) -> torch.Tensor:
        """Sample training timesteps based on config mode.

        Modes:
            uniform: t ~ Uniform[0, num_timesteps) [default, equivalent to old behavior]
            subset: t sampled uniformly from the list in config['train_timesteps_list']
                    (e.g. [99] for T=1, [0,11,22,...,99] for T=10)
            skewed: t ~ full range but one index is upweighted by skew_weight
        """
        T = self.diffusion.num_timesteps
        mode = self.config.get('train_timesteps_mode', 'uniform')

        if mode == 'uniform':
            return torch.randint(0, T, (B,), device=self.device)

        if mode == 'subset':
            ts_str = self.config.get('train_timesteps_list', '')
            if not ts_str:
                return torch.randint(0, T, (B,), device=self.device)
            # Cache parsed list to avoid re-parsing per batch
            if not hasattr(self, '_cached_timestep_subset'):
                subset = [int(x) for x in ts_str.split(',') if x.strip()]
                self._cached_timestep_subset = torch.tensor(subset, device=self.device, dtype=torch.long)
            subset = self._cached_timestep_subset
            idx = torch.randint(0, len(subset), (B,), device=self.device)
            return subset[idx]

        if mode == 'skewed':
            skew_idx = int(self.config.get('train_timesteps_skew_idx', T - 1))
            skew_weight = float(self.config.get('train_timesteps_skew_weight', 3.0))
            # Cache weights tensor
            if not hasattr(self, '_cached_skew_weights'):
                w = torch.ones(T, device=self.device)
                w[skew_idx] = skew_weight
                w = w / w.sum()
                self._cached_skew_weights = w
            w = self._cached_skew_weights
            return torch.multinomial(w, B, replacement=True)

        # Fallback
        return torch.randint(0, T, (B,), device=self.device)

    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger('SceneCompletion')
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = self.output_dir / 'training.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _create_model(self) -> nn.Module:
        """Create the model for SemanticKITTI (256×256×32).

        Model variants:
        - exp_1: Dense Conv3d for LiDAR (full, lite)
        - exp_2: Sparse spconv for LiDAR (sparse_full, sparse_lite)
        - S4_MIMO: PaSCo-style MIMO with N separate decoder heads (mimo_full, mimo_lite)
        """
        config = self.config
        model_type = config.get('model_type', 'full')

        if model_type == 'lite':
            # exp_1 lite: Dense Conv3d, 16 → 32 → 64 → 128 channels
            model = SceneCompletionUNetLite(
                num_classes=config['num_classes'],
                base_channels=16,
                time_emb_dim=64,
            ).to(self.device)
            self.logger.info("exp_1 lite: SceneCompletionUNetLite (8M params)")

        elif model_type == 'full':
            # exp_1 full: Dense Conv3d, 32 → 64 → 128 → 256 channels
            model = SceneCompletionUNet(
                num_classes=config['num_classes'],
                base_channels=32,
                time_emb_dim=128,
            ).to(self.device)
            self.logger.info("exp_1 full: SceneCompletionUNet (32M params)")

        elif model_type == 'sparse_lite':
            # exp_2 lite: Sparse spconv, 16 → 32 → 64 → 128 channels
            model = SceneCompletionUNetSparseLite(
                num_classes=config['num_classes'],
                base_channels=16,
                time_emb_dim=64,
                lidar_base_channels=8,
                lidar_out_channels=16,
            ).to(self.device)
            self.logger.info("exp_2 lite: SceneCompletionUNetSparseLite (9M params)")

        elif model_type == 'sparse_full':
            # exp_2 full: Sparse spconv, 32 → 64 → 128 → 256 channels
            lidar_in_ch = 20 if config.get('use_lsk3d', False) else 1
            tsdf_bev = config.get('tsdf_bev', False)
            if config.get('geom_dir'):
                if tsdf_bev:
                    lidar_in_ch = 29  # S46: 20 LSK3D + 5 height + 3 normals + 1 intensity (no TSDF)
                else:
                    lidar_in_ch = 30  # S43: 20 LSK3D + 5 height + 3 normals + 1 intensity + 1 TSDF
            geom_bev_channels = 4 if tsdf_bev else 0
            ssc_cond_channels = config['num_classes'] if config.get('scpnet_pred_dir') else 0
            no_bev = config.get('no_bev', False)
            densify_nn = config.get('densify_nn', False)
            distance_gate = config.get('distance_gate', False)
            lidar_film = config.get('lidar_film', False)
            ssc_multiscale = config.get('ssc_multiscale', False)
            obs_mask_channel = config.get('obs_mask_channel', False)
            model = SceneCompletionUNetSparse(
                num_classes=config['num_classes'],
                base_channels=32,
                time_emb_dim=128,
                lidar_base_channels=16,
                lidar_out_channels=32,
                lidar_in_channels=lidar_in_ch,
                no_bev=no_bev,
                densify_nn=densify_nn,
                add_distance_gate=distance_gate,
                lidar_film_mode=lidar_film,
                geom_bev_channels=geom_bev_channels,
                ssc_cond_channels=ssc_cond_channels,
                ssc_multiscale=ssc_multiscale,
                obs_mask_channel=obs_mask_channel,
            ).to(self.device)
            extra_info = ""
            if lidar_in_ch == 20:
                extra_info += " [LSK3DNet 20ch]"
            if no_bev:
                extra_info += " [no BEV]"
            if densify_nn:
                extra_info += " [NN densify]"
            if distance_gate:
                extra_info += " [dist gate]"
            if lidar_film:
                extra_info += " [FiLM]"
            if tsdf_bev:
                extra_info += " [TSDF BEV 4ch]"
            if config.get('geom_dir'):
                extra_info += f" [{lidar_in_ch}ch sparse]"
            self.logger.info(f"exp_2 full: SceneCompletionUNetSparse (35M params){extra_info}")

        elif model_type == 'mimo_lite':
            # S4 MIMO lite: PaSCo-style with N separate decoder heads
            if MIMOSceneCompletionUNetLite is None:
                raise ImportError("MIMOSceneCompletionUNetLite not available - check mimo_scene_unet.py")
            n_subnets = config.get('mimo_num_subnets', 3)
            model = MIMOSceneCompletionUNetLite(
                num_classes=config['num_classes'],
                n_subnets=n_subnets,
                base_channels=16,
                time_emb_dim=64,
                lidar_base_channels=8,
                lidar_out_channels=16,
            ).to(self.device)
            self.logger.info(f"S4 MIMO lite: MIMOSceneCompletionUNetLite (n_subnets={n_subnets})")

        elif model_type == 'mimo_full':
            # S4 MIMO full: PaSCo-style with N separate decoder heads
            if MIMOSceneCompletionUNet is None:
                raise ImportError("MIMOSceneCompletionUNet not available - check mimo_scene_unet.py")
            n_subnets = config.get('mimo_num_subnets', 3)
            model = MIMOSceneCompletionUNet(
                num_classes=config['num_classes'],
                n_subnets=n_subnets,
                base_channels=32,
                time_emb_dim=128,
                lidar_base_channels=16,
                lidar_out_channels=32,
            ).to(self.device)
            self.logger.info(f"S4 MIMO full: MIMOSceneCompletionUNet (n_subnets={n_subnets})")

        elif model_type == 'v2_full':
            # V2 full: FiLM conditioning + multi-scale aux BEV (no cascade)
            lidar_in_ch = 20 if config.get('use_lsk3d', False) else 1
            aux_bev = config.get('aux_bev', True)
            model = V2ModelWrapper(SceneCompletionUNetV2(
                num_classes=config['num_classes'],
                base_channels=32,
                time_emb_dim=128,
                lidar_in_channels=lidar_in_ch,
                lidar_base_channels=16,
                lidar_out_channels=32,
                aux_bev=aux_bev,
            )).to(self.device)
            extra_info = ""
            if lidar_in_ch == 20:
                extra_info += " [LSK3DNet 20ch]"
            if aux_bev:
                extra_info += " [aux BEV]"
            self.logger.info(f"V2 full: SceneCompletionUNetV2 (FiLM + multi-scale aux BEV){extra_info}")

        elif model_type == 'v2_lite':
            # V2 lite: FiLM conditioning + multi-scale aux BEV (smaller)
            lidar_in_ch = 20 if config.get('use_lsk3d', False) else 1
            aux_bev = config.get('aux_bev', True)
            model = V2ModelWrapper(SceneCompletionUNetV2(
                num_classes=config['num_classes'],
                base_channels=24,
                time_emb_dim=96,
                lidar_in_channels=lidar_in_ch,
                lidar_base_channels=12,
                lidar_out_channels=24,
                aux_bev=aux_bev,
            )).to(self.device)
            extra_info = ""
            if lidar_in_ch == 20:
                extra_info += " [LSK3DNet 20ch]"
            if aux_bev:
                extra_info += " [aux BEV]"
            self.logger.info(f"V2 lite: SceneCompletionUNetV2 (FiLM + multi-scale aux BEV){extra_info}")

        elif model_type in ('v3_full', 'v3_ablation', 'v3_coarse2fine', 'v3_c2f_ablation'):
            # V3: Internal BEV or CoarseToFine + sparse FiLM (fixes V2 collapse)
            lidar_in_ch = 20 if config.get('use_lsk3d', False) else 1
            aux_bev = config.get('aux_bev', True)
            no_sparse_film = model_type in ('v3_ablation', 'v3_c2f_ablation')
            dense_bev_ch = config.get('dense_bev_channels', 32)
            bev_comp_layers = config.get('bev_completion_layers', 5)
            # V3.0 uses InternalBEV, V3.1 uses CoarseToFine
            if model_type in ('v3_coarse2fine', 'v3_c2f_ablation'):
                dense_mode = 'coarse2fine'
            else:
                dense_mode = 'internal_bev'
            cfg_drop = config.get('cfg_drop_prob', 0.0)
            model = V3ModelWrapper(SceneCompletionUNetV3(
                num_classes=config['num_classes'],
                base_channels=32,
                time_emb_dim=128,
                lidar_in_channels=lidar_in_ch,
                lidar_base_channels=16,
                lidar_out_channels=32,
                dense_bev_channels=dense_bev_ch,
                bev_completion_layers=bev_comp_layers,
                aux_bev=aux_bev,
                no_sparse_film=no_sparse_film,
                dense_mode=dense_mode,
                cfg_drop_prob=cfg_drop,
            )).to(self.device)
            extra_info = ""
            if lidar_in_ch == 20:
                extra_info += " [LSK3DNet 20ch]"
            if aux_bev:
                extra_info += " [aux BEV]"
            if no_sparse_film:
                extra_info += " [no sparse FiLM]"
            if dense_mode == 'coarse2fine':
                extra_info += " [CoarseToFine]"
            if cfg_drop > 0:
                extra_info += f" [CFG drop={cfg_drop}]"
            self.logger.info(f"V3 {model_type}: SceneCompletionUNetV3 ({dense_mode} + FiLM){extra_info}")

        elif model_type in ('v4_continuous', 'v4_factored'):
            # V4: Factored representation on V3.1 backbone (S13/S14)
            from gssc.models.scene_unet_v3 import V4ModelWrapper
            lidar_in_ch = 20 if config.get('use_lsk3d', False) else 1
            aux_bev = config.get('aux_bev', True)
            cfg_drop = config.get('cfg_drop_prob', 0.0)
            fuse_time = config.get('fuse_time_cond', False)
            dense_bev_ch = config.get('dense_bev_channels', 32)

            if model_type == 'v4_continuous':
                in_channels = 21   # 1 occ logit + 20 sem logits
                out_channels = 21  # predict 21-dim noise
            else:  # v4_factored
                in_channels = 22   # 2 occ one-hot + 20 sem one-hot
                out_channels = 22  # 2 occ logits + 20 sem logits

            model = V4ModelWrapper(SceneCompletionUNetV3(
                num_classes=config['num_classes'],
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=32,
                time_emb_dim=128,
                lidar_in_channels=lidar_in_ch,
                lidar_base_channels=16,
                lidar_out_channels=32,
                dense_bev_channels=dense_bev_ch,
                dense_mode='coarse2fine',
                aux_bev=aux_bev,
                no_sparse_film=False,
                cfg_drop_prob=cfg_drop,
                fuse_time_cond=fuse_time,
            )).to(self.device)
            extra_info = f" [in={in_channels}ch, out={out_channels}ch]"
            if lidar_in_ch == 20:
                extra_info += " [LSK3DNet 20ch]"
            if aux_bev:
                extra_info += " [aux BEV]"
            if fuse_time:
                extra_info += " [fused time+cond]"
            if cfg_drop > 0:
                extra_info += f" [CFG drop={cfg_drop}]"
            self.logger.info(f"V4 {model_type}: SceneCompletionUNetV3 (CoarseToFine + FiLM){extra_info}")

        elif model_type == 'v5_ve':
            # V5: VE Gaussian diffusion with soft probability encoding (S16+)
            # Key differences from V4:
            # - VE diffusion (no signal scaling, no 1/alpha_bar amplification)
            # - 20 channels (soft probs), not 21 (logit-encoded occ + sem)
            # - Implicit occupancy (class 0 = empty)
            from gssc.models.scene_unet_v3 import V4ModelWrapper
            lidar_in_ch = 20 if config.get('use_lsk3d', False) else 1
            aux_bev = config.get('aux_bev', True)
            cfg_drop = config.get('cfg_drop_prob', 0.0)
            fuse_time = config.get('fuse_time_cond', False)
            dense_bev_ch = config.get('dense_bev_channels', 32)

            in_channels = 20   # 20 soft probability channels
            out_channels = 20  # predict 20-dim noise

            cond_3d_ch = config.get('cond_3d_channels', 0)

            model = V4ModelWrapper(SceneCompletionUNetV3(
                num_classes=config['num_classes'],
                in_channels=in_channels,
                out_channels=out_channels,
                base_channels=32,
                time_emb_dim=128,
                lidar_in_channels=lidar_in_ch,
                lidar_base_channels=16,
                lidar_out_channels=32,
                dense_bev_channels=dense_bev_ch,
                dense_mode='coarse2fine',
                aux_bev=aux_bev,
                no_sparse_film=False,
                cfg_drop_prob=cfg_drop,
                fuse_time_cond=fuse_time,
                cond_3d_channels=cond_3d_ch,
            )).to(self.device)
            extra_info = f" [in={in_channels}ch (soft probs), out={out_channels}ch]"
            if lidar_in_ch == 20:
                extra_info += " [LSK3DNet 20ch]"
            if cond_3d_ch > 0:
                extra_info += f" [cond_3d={cond_3d_ch}ch]"
            if aux_bev:
                extra_info += " [aux BEV]"
            if fuse_time:
                extra_info += " [fused time+cond]"
            if cfg_drop > 0:
                extra_info += f" [CFG drop={cfg_drop}]"
            self.logger.info(f"V5 VE: SceneCompletionUNetV3 (VE diffusion + soft probs){extra_info}")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return model

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create dataloader.

        S4 MIMO Training Mode (PaSCo-style):
        - When mimo_use_training_mode=True, wraps dataset with MIMODatasetWrapper
        - Training: Each subnet receives a DIFFERENT sample (dataset-level MIMO)
        - Validation: Each subnet receives the SAME sample (then different augs at model level)

        Reference: PaSCo CVPR 2024, Section 3.2.1-3.2.2
        """
        config = self.config

        if split == 'train':
            sequences = config['train_sequences']
            augment = True
            shuffle = True
        else:
            sequences = config['val_sequences']
            augment = False
            shuffle = False

        # Use quantized dataset if available (much faster loading)
        use_quantized = config.get('use_quantized', False)
        s3_mode = config.get('s3_mode', None)

        # S3 DSKD: Use specialized dataset for teacher/student mode
        if s3_mode is not None and S3DSKDDataset is not None:
            # S3DSKDDataset expects data_root to be parent of SemanticKITTI_3D and dataset_SemanticKITTI_SSC
            # If data_root contains 'dataset_SemanticKITTI_SSC', use parent directory
            s3_data_root = config['data_root']
            if 'dataset_SemanticKITTI_SSC' in s3_data_root:
                s3_data_root = str(Path(s3_data_root).parent)
            # LSK3DNet 3D features dir
            lsk3d_dir = config.get('lsk3d_dir', None)
            if config.get('use_lsk3d', False) and lsk3d_dir is None:
                lsk3d_dir = str(Path(s3_data_root) / 'SemanticKITTI_3D' / '256_lsk3d_3d')

            no_tsdf_sparse = config.get('tsdf_bev', False)
            dataset = S3DSKDDataset(
                data_root=s3_data_root,
                sequences=sequences,
                resolution=256,
                mode=s3_mode,  # 'teacher', 'intermediate', or 'student'
                pred_bev_dir=config.get('pred_bev_dir', None),
                multi_frame_dir='SemanticKITTI_3D/256_multi_frame',
                augment=augment,
                use_rectified_labels=False,
                gt_bev_prob=config.get('gt_bev_prob', 0.0) if split == 'train' else 0.0,
                lsk3d_dir=lsk3d_dir if config.get('use_lsk3d', False) else None,
                geom_dir=config.get('geom_dir'),
                no_tsdf_sparse=no_tsdf_sparse,
                scpnet_pred_dir=config.get('scpnet_pred_dir'),
                talos_pred_dir=config.get('talos_pred_dir'),
                bev_cold_dir=config.get('bev_cold_dir'),
                force_single_frame_lidar=config.get('force_single_frame_lidar', False),
            )
            if split == 'train':
                self.logger.info(f"S3 DSKD: Using S3DSKDDataset in {s3_mode} mode")
                if config.get('use_lsk3d', False):
                    self.logger.info(f"         Using LSK3DNet 3D features (20ch) from {lsk3d_dir}")
                if config.get('geom_dir'):
                    geom_ch = 9 if no_tsdf_sparse else 10
                    self.logger.info(f"         Using geometric features ({geom_ch}ch) from {config['geom_dir']}")
                if no_tsdf_sparse:
                    self.logger.info("         TSDF → BEV projection (4ch column stats, not in sparse encoder)")
                if config.get('no_bev', False):
                    self.logger.info("         BEV conditioning DISABLED (S6 mode)")
                if s3_mode == 'teacher':
                    if config.get('force_single_frame_lidar', False):
                        self.logger.info("         Loading SINGLE-FRAME voxels (force_single_frame_lidar=True, overrides teacher-mode default)")
                    else:
                        self.logger.info("         Loading multi-frame voxels (2x density)")
                elif s3_mode == 'intermediate':
                    self.logger.info("         Loading single-frame voxels with GT BEV")
                elif s3_mode == 'student':
                    gt_prob = config.get('gt_bev_prob', 0.0)
                    if gt_prob > 0:
                        self.logger.info(f"         BEV mixing enabled: {gt_prob*100:.0f}% GT, {(1-gt_prob)*100:.0f}% Pred")
        elif use_quantized:
            dataset = SemanticKITTI3DQuantizedDataset(
                data_root=config['data_root'],
                sequences=sequences,
                augment=augment,
            )
        else:
            dataset = SemanticKITTI3DDataset(
                data_root=config['data_root'],
                sequences=sequences,
                augment=augment,
                waffleiron_root=config.get('waffleiron_root'),
                lsk3d_3d_root=config.get('lsk3d_3d_root'),  # S18: SDEdit init
                densify_nn=config.get('densify_nn', False),  # S25-S27
            )

        # S4: MIMO Training Mode - wrap dataset with MIMODatasetWrapper
        # Reference: PaSCo CVPR 2024 - kitti_dataset.py lines 126-140
        # Training: Each subnet receives a DIFFERENT sample from the dataset
        # Validation: Each subnet receives the SAME sample (augs applied at model level)
        model_type = config.get('model_type', 'full')
        is_mimo_model = model_type in ['mimo_lite', 'mimo_full']

        use_mimo_training = (
            is_mimo_model or (
                config.get('use_mimo', False) and
                config.get('mimo_use_training_mode', False)
            )
        ) and MIMODatasetWrapper is not None

        if use_mimo_training:
            n_subnets = config.get('mimo_num_subnets', 3)  # PaSCo default: 3
            apply_crop = split == 'train'  # 80% crop for training only
            # CRITICAL FIX: Enable training augmentation to match inference TTA
            # This fixes the train/inference mismatch that caused ~0.69% mIoU
            train_aug = config.get('mimo_train_aug', True)  # Default: enabled

            dataset = MIMODatasetWrapper(
                base_dataset=dataset,
                n_subnets=n_subnets,
                split=split,  # 'train' for different samples, 'val' for same sample
                apply_crop=apply_crop,  # PaSCo-style 80% random crop
                train_aug=train_aug,  # NEW: Apply augmentation during training
                use_continuous_aug=True,  # PaSCo-style continuous rotation
                max_angle=config.get('mimo_max_angle', 30.0),  # ±30° rotation
                max_translation=(0.6, 0.6, 0.4),  # PaSCo default translation
            )
            if split == 'train':
                self.logger.info(f"S4 MIMO Training: Using MIMODatasetWrapper (n_subnets={n_subnets})")
                self.logger.info("    Each subnet sees a DIFFERENT sample per batch (PaSCo-style)")
                self.logger.info("    80% random crop applied per sample")
                self.logger.info(f"    Training augmentation: {'ENABLED (±30° rotation)' if train_aug else 'DISABLED'}")

            # Use custom collate function for MIMO batches with channel concatenation
            from functools import partial
            concatenate_channels = is_mimo_model  # Channel concat for MIMO UNet model
            collate_fn = partial(
                mimo_collate_fn,
                n_subnets=n_subnets,
                concatenate_channels=concatenate_channels,
            )

            loader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=shuffle,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                drop_last=(split == 'train'),
                collate_fn=collate_fn,
            )
            if split == 'train' and is_mimo_model:
                self.logger.info("    Channel concatenation: ENABLED (single forward pass)")
        else:
            loader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=shuffle,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                drop_last=(split == 'train'),
            )

        return loader

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step.

        For MIMO models (mimo_lite, mimo_full):
        - Batch contains channel-concatenated inputs: lidar [B, N, H, W, D], bev [B, N, H, W]
        - GT scene is stacked: [B, N, H, W, D]
        - Model returns N outputs from N decoder heads
        - Loss is averaged across all N heads
        """
        self.model.train()

        # Check if this is a MIMO batch (channel-concatenated)
        is_mimo_batch = batch.get('channel_concatenated', False)
        model_type = self.config.get('model_type', 'full')
        is_mimo_model = model_type in ['mimo_lite', 'mimo_full']

        if is_mimo_model and is_mimo_batch:
            # S4 MIMO: PaSCo-style training with N separate decoder heads
            return self._train_step_mimo(batch)

        # Standard (non-MIMO) training path
        # Move to device
        lidar = batch['lidar'].to(self.device)  # [B, 1, H, W, D]
        gt_scene = batch['gt_scene'].to(self.device)  # [B, H, W, D]
        bev = batch['bev'].to(self.device)  # [B, H, W]

        # B1: Derive BEV from SCPNet 3D prediction (height-pool)
        if self.config.get('bev_from_base') and 'scpnet_pred' in batch:
            # Topmost non-empty class along Z (matches GT BEV computation)
            scp = batch['scpnet_pred'].to(self.device)
            bev = torch.zeros(scp.shape[0], scp.shape[1], scp.shape[2],
                               dtype=torch.long, device=self.device)
            for z in range(scp.shape[3] - 1, -1, -1):
                layer = scp[:, :, :, z]
                mask = (layer > 0) & (bev == 0)
                bev[mask] = layer[mask]  # [B, H, W]

        # S20: Replace binary LiDAR with LSK3DNet 3D soft probs (20ch)
        # Analogous to DiffSSC's Cylinder3D features — rich semantic conditioning
        if self.config.get('use_lsk3d', False) and 'lsk3d_3d_probs' in batch:
            lidar = batch['lsk3d_3d_probs'].to(self.device)  # [B, 20, H, W, D]

        # S48: Direct prediction — bypass diffusion entirely
        if self.direct_prediction:
            return self._train_step_direct(lidar, gt_scene, bev, batch)

        # S40v2/S41: Compute soft BEV from BEV model on-the-fly
        bev_loss_e2e = None
        if self.bev_model is not None and lidar.shape[1] > 1:
            # Convert dense LSK3D [B,C,H,W,D] → sparse for BEV model's spconv input
            # S43 FIX: BEV model was trained with 20ch (LSK3D only). When lidar has
            # 30ch (LSK3D+geom), we must use only the first 20 channels for both
            # occupancy detection and feature extraction. TSDF-only voxels (ch 20-29)
            # have zero LSK3D features and would confuse the BEV model.
            bev_in_ch = self.bev_model.encoder.stem.weight.shape[-1]  # 20 for V5_LSK3D
            lidar_for_bev = lidar[:, :bev_in_ch]  # [B, 20, H, W, D] — only LSK3D channels
            B_bev = lidar.shape[0]
            bev_coords_list = []
            bev_feats_list = []
            with torch.no_grad():
                occ_mask = (lidar_for_bev.abs().sum(dim=1) > 1e-6)  # [B, H, W, D]
            for b in range(B_bev):
                nz = occ_mask[b].nonzero()  # [N_b, 3]
                feats = lidar_for_bev[b, :, nz[:, 0], nz[:, 1], nz[:, 2]].T  # [N_b, 20]
                batch_col = torch.full((nz.shape[0], 1), b, device=self.device, dtype=torch.int32)
                bev_coords_list.append(torch.cat([batch_col, nz.int()], dim=1))
                bev_feats_list.append(feats)
            sparse_coords = torch.cat(bev_coords_list, dim=0)
            sparse_feats = torch.cat(bev_feats_list, dim=0)

            if self.use_e2e_bev:
                # S41: gradient flow — don't detach
                bev_logits = self.bev_model(sparse_feats, sparse_coords, B_bev)  # [B, 20, H, W]
                bev = F.softmax(bev_logits, dim=1)  # soft BEV with gradients
                # BEV supervised loss against GT BEV
                gt_bev_for_loss = batch.get('gt_bev', batch['bev']).to(self.device).long()
                bev_loss_e2e = F.cross_entropy(bev_logits, gt_bev_for_loss)
            else:
                # S40v3: frozen BEV model, hard argmax BEV (format-compatible with KD teacher)
                # Soft BEV (softmax) caused KD divergence because teacher's bev_embed was trained
                # on one-hot BEV — different format creates conflicting gradient objectives
                with torch.no_grad():
                    bev_logits = self.bev_model(sparse_feats, sparse_coords, B_bev)
                    bev = bev_logits.argmax(dim=1)  # [B, H, W] hard class indices

        # S3 DSKD: Get multi-frame lidar for teacher if available (INTERMEDIATE mode)
        # SCPNet: Teacher uses multi-frame (dense), Student uses single-frame (sparse)
        lidar_multi = batch.get('lidar_multi', None)
        if lidar_multi is not None:
            lidar_multi = lidar_multi.to(self.device)

        # S2: Optional 2D→3D lifting for additional conditioning
        # IMPORTANT: Compute lifting BEFORE CFG dropout so we have features to potentially drop
        lifted_3d = None
        lifted_features = None
        if self.config.get('use_lifting', False) and self.lifter is not None:
            # Convert BEV to one-hot for lifting (handles soft BEV [B,K,H,W] too)
            if bev.dim() == 4:
                bev_one_hot = bev  # Already soft [B, K, H, W]
            else:
                bev_one_hot = F.one_hot(bev.long(), self.config['num_classes']).float()
                bev_one_hot = bev_one_hot.permute(0, 3, 1, 2)  # [B, C, H, W]
            lifted_3d, lifted_features = self.lifter(bev_one_hot)
            # lifted_3d: [B, H, W, Z] semantic labels
            # lifted_features: [B, feature_dim, H, W, Z] encoded features

        # S1: Classifier-Free Guidance - randomly drop LIFTED condition (NOT BEV!)
        # Key design decision (from architecture_improvement_plan.md):
        # - Drop ONLY the lifted condition (p=0.1)
        # - Keep BEV and LiDAR conditions ALWAYS present
        # - This is hierarchical CFG - some conditions more important than others
        if self.config.get('use_cfg', False) and lifted_features is not None:
            cfg_drop_prob = self.config.get('cfg_drop_prob', 0.1)
            if torch.rand(1).item() < cfg_drop_prob:
                # Zero out LIFTED condition for this batch (not BEV!)
                lifted_features = torch.zeros_like(lifted_features)

        # Sample timesteps (supports uniform/subset/skewed modes for ablations)
        B = lidar.shape[0]
        t = self._sample_train_timesteps(B)

        # Compute loss
        # cond_3d: LSK3DNet 3D predictions as conditioning input channels
        cond_3d = None
        if 'lsk3d_3d_probs' in batch and self.config.get('cond_3d_channels', 0) > 0:
            cond_3d = batch['lsk3d_3d_probs'].to(self.device)

        loss_kwargs = dict(lifted_features=lifted_features)
        if cond_3d is not None:
            loss_kwargs['cond_3d'] = cond_3d
        # S25-S27: Pass precomputed NN indices for densification
        if 'nn_indices' in batch:
            loss_kwargs['nn_indices'] = batch['nn_indices'].to(self.device)
        # S46: TSDF BEV features (4ch column stats)
        if 'tsdf_bev' in batch:
            loss_kwargs['geom_bev'] = batch['tsdf_bev'].to(self.device)
        # SCPNet SSC prediction conditioning (Phase 2 refinement)
        if 'scpnet_pred' in batch:
            scpnet_labels = batch['scpnet_pred'].to(self.device)  # [B, H, W, D] int64
            scpnet_onehot = F.one_hot(scpnet_labels, self.config['num_classes']).float()
            loss_kwargs['ssc_pred'] = scpnet_onehot.permute(0, 4, 1, 2, 3)  # [B, K, H, W, D]
        # B2: Observation mask for model input
        if self.config.get('obs_mask_channel', False):
            if lidar.shape[1] == 1:
                loss_kwargs['obs_mask'] = (lidar > 0).float()
            else:
                loss_kwargs['obs_mask'] = (lidar.abs().sum(dim=1, keepdim=True) > 0).float()
        # B4: Cold Diffusion — pass SCPNet pred as noise target
        if self.config.get('cold_diffusion') and 'scpnet_pred' in batch:
            _scp = batch['scpnet_pred'].to(self.device)
            loss_kwargs['x_scpnet'] = F.one_hot(_scp.long(), self.config['num_classes']).float().permute(0, 4, 1, 2, 3)
        losses = self.diffusion.training_losses(
            self.model,
            gt_scene,
            t,
            bev,
            lidar,
            **loss_kwargs,
        )

        loss_ssc = losses['loss']

        # S43: 3D Lovász loss on x_0 predictions (directly optimizes mIoU)
        # Note: Applied at all timesteps including high-noise ones where predictions
        # are noisy. This is consistent with the main CE/KL loss behavior. The gradient
        # signal from high-noise steps is weak but not harmful.
        ssc_lovasz_loss_value = None
        if self.ssc_lovasz is not None and 'x_0_logits' in losses:
            x_0_logits = losses['x_0_logits']  # [B, C, H, W, D]
            lovasz_loss = self.ssc_lovasz(x_0_logits, gt_scene)
            ssc_lovasz_loss_value = lovasz_loss.item()
            loss_ssc = loss_ssc + self.ssc_lovasz_weight * lovasz_loss

        # V2: Multi-scale auxiliary BEV loss (FiLM conditioning, fixed weight)
        aux_bev_loss_value = None
        # Resolve GT BEV target for aux loss (must be [B, H, W] class indices)
        gt_bev_for_aux = None
        if 'gt_bev' in batch:
            gt_bev_for_aux = batch['gt_bev'].to(self.device).long()
        elif bev.dim() == 3:
            gt_bev_for_aux = bev.long()  # Hard BEV [B,H,W], safe to use
        # Skip aux BEV loss if no valid GT target (e.g. soft BEV without gt_bev in batch)

        if self.use_aux_bev and hasattr(self.model, 'last_aux_bev') and self.model.last_aux_bev is not None and gt_bev_for_aux is not None:
            aux_bev_pred = self.model.last_aux_bev  # [B, 20, H, W]
            # Proven BEV loss: Focal(gamma=2, class_0=0.02) + Lovász(0.3)
            loss_bev_focal = self.aux_focal(aux_bev_pred, gt_bev_for_aux)
            loss_bev_lovasz = self.aux_lovasz(aux_bev_pred, gt_bev_for_aux)
            loss_bev = loss_bev_focal + self.aux_lovasz_weight * loss_bev_lovasz
            aux_bev_loss_value = loss_bev.item()
            # Fixed weight combination
            loss = loss_ssc + self.aux_bev_weight * loss_bev
        else:
            loss = loss_ssc

        # S44: Height-pool BEV auxiliary loss
        hp_bev_loss_value = None
        if self.use_hp_bev_aux and 'x_0_logits' in losses and gt_bev_for_aux is not None:
            x_0_logits_hp = losses['x_0_logits']  # [B, 20, H, W, D]
            bev_pred = x_0_logits_hp.max(dim=-1)[0]  # [B, 20, H, W] — height-pool
            hp_focal = self.hp_bev_focal(bev_pred, gt_bev_for_aux)
            hp_lovasz = self.hp_bev_lovasz(bev_pred, gt_bev_for_aux)
            hp_bev_loss = hp_focal + self.hp_bev_lovasz_weight * hp_lovasz
            hp_bev_loss_value = hp_bev_loss.item()
            loss = loss + self.hp_bev_aux_weight * hp_bev_loss

        # S41: End-to-End BEV supervised loss (gradient flows back through BEV model)
        if bev_loss_e2e is not None:
            loss = loss + self.e2e_bev_weight * bev_loss_e2e

        # S3: Knowledge Distillation from Teacher (multi-frame) to Student (single-frame)
        # Two modes: 'dskd' (pairwise features) or 'output' (soft labels)
        kd_loss_value = None
        if self.use_dskd and self.teacher_model is not None:
            kd_weight = self.config.get('dskd_weight', 0.5)
            kd_type = self.kd_type

            # Get noisy x_t and bev_onehot for the forward pass
            x_t_onehot = losses.get('x_t_onehot', None)
            bev_onehot = losses.get('bev_onehot', None)

            if x_t_onehot is None or bev_onehot is None:
                if not hasattr(self, '_kd_warning_logged'):
                    self.logger.warning("KD skipped: x_t_onehot or bev_onehot not available")
                    self._kd_warning_logged = True
            else:
                # Determine inputs for teacher vs student
                # INTERMEDIATE mode: Same BEV (GT), different LiDAR (multi vs single)
                # STUDENT mode: Different BEV (GT vs Pred), same LiDAR (single)
                teacher_lidar = lidar_multi if lidar_multi is not None else lidar
                student_lidar = lidar

                # S43: If teacher has fewer input channels, truncate lidar for teacher
                if hasattr(self, 'teacher_lidar_ch') and self.teacher_lidar_ch is not None:
                    if teacher_lidar.shape[1] > self.teacher_lidar_ch:
                        teacher_lidar = teacher_lidar[:, :self.teacher_lidar_ch]

                # Determine teacher BEV
                if self.s3_mode == 'student' and 'gt_bev' in batch:
                    gt_bev = batch['gt_bev'].to(self.device)
                    teacher_bev_onehot = F.one_hot(gt_bev.long(), self.config['num_classes']).float()
                    teacher_bev_onehot = teacher_bev_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]
                else:
                    teacher_bev_onehot = bev_onehot

                # Compute lifted features for KD forward passes
                # IMPORTANT: Teacher's lifted_embed was NEVER trained (random init, same
                # bugs #2/#4 affected teacher training). Passing trained lifter features
                # through teacher's random lifted_embed produces wrong noise, corrupting
                # teacher soft labels. So pass None to teacher — skips lifted_embed entirely.
                teacher_lifted = None  # Teacher ignores lifted features (random lifted_embed)
                student_lifted = lifted_features  # From pred BEV (already computed above)

                if kd_type == 'output':
                    # OUTPUT-LEVEL KD (Hinton-style soft labels)
                    # Match teacher's soft predictions, not internal features
                    # This is more robust for multi-frame → single-frame distillation
                    T = self.dskd_temperature  # Temperature for softening

                    # Get teacher predictions (with multi-frame LiDAR + GT lifted features)
                    with torch.no_grad():
                        teacher_logits = self.teacher_model(
                            x_t_onehot, t, teacher_bev_onehot, teacher_lidar,
                            lifted_features=teacher_lifted
                        )  # [B, C, H, W, D]

                    # Reuse student logits from main forward pass (avoids double forward)
                    student_logits = losses['x_0_logits']  # [B, C, H, W, D]

                    # Soft label KD: KL divergence between softened predictions
                    # L_kd = KL(softmax(student/T) || softmax(teacher/T)) * T^2
                    teacher_soft = F.softmax(teacher_logits / T, dim=1)
                    student_log_soft = F.log_softmax(student_logits / T, dim=1)

                    # KL divergence: sum over classes (dim=1), then mean over spatial
                    # F.kl_div computes: target * (log(target) - input)
                    # With reduction='none', we get element-wise KL, then sum over classes
                    kl_per_voxel = F.kl_div(
                        student_log_soft, teacher_soft, reduction='none'
                    ).sum(dim=1)  # [B, H, W, D] - sum over class dimension
                    kd_loss_value = kl_per_voxel.mean() * (T * T)  # Mean over spatial, scale by T^2

                    loss = loss + kd_weight * kd_loss_value

                elif kd_type == 'dskd' and self.dskd_loss is not None:
                    # DSKD: Pairwise feature similarity (per SCPNet)
                    if hasattr(self.model, 'forward_features'):
                        # Get student features (pass lifted_features + geom_bev to match main forward)
                        student_geom_bev = batch.get('tsdf_bev', None)
                        if student_geom_bev is not None:
                            student_geom_bev = student_geom_bev.to(self.device)
                        _, student_features = self.model.forward_features(
                            x_t_onehot, t, bev_onehot, student_lidar,
                            lifted_features=student_lifted,
                            geom_bev=student_geom_bev,
                        )
                        # Get teacher features (None for lifted — teacher's lifted_embed is random)
                        with torch.no_grad():
                            _, teacher_features = self.teacher_model.forward_features(
                                x_t_onehot, t, teacher_bev_onehot, teacher_lidar,
                                lifted_features=teacher_lifted
                            )
                        # Pairwise similarity matching
                        kd_loss_value = self.dskd_loss(
                            student_features,
                            teacher_features.detach(),
                            student_occupancy=student_lidar,
                            teacher_occupancy=teacher_lidar,
                        )
                        loss = loss + kd_weight * kd_loss_value
                    else:
                        if not hasattr(self, '_dskd_warning_logged'):
                            self.logger.warning("DSKD skipped: model has no forward_features method")
                            self._dskd_warning_logged = True

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (include BEV model if E2E)
        all_params = list(self.model.parameters()) + (list(self.lifter.parameters()) if self.lifter is not None else [])
        if self.use_e2e_bev and self.bev_model is not None:
            all_params += list(self.bev_model.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Update EMA
        self.ema.update()

        # Build return metrics
        result = {
            'loss': loss.item(),
            'accuracy': losses['accuracy'].item(),
            'lr': self.scheduler.get_last_lr()[0],
        }

        # Add V2 enhanced metrics if available
        if 'kl_loss' in losses:
            result['kl_loss'] = losses['kl_loss'].item()
        if 'lovasz_loss' in losses:
            result['lovasz_loss'] = losses['lovasz_loss'].item()
        if 'auxiliary_loss' in losses:
            result['auxiliary_loss'] = losses['auxiliary_loss'].item()
        if 'occupied_accuracy' in losses:
            result['occupied_accuracy'] = losses['occupied_accuracy'].item()

        # Gaussian diffusion metrics (S14+)
        if 'mse_loss' in losses:
            result['mse_loss'] = losses['mse_loss'].item()
        if 'reg_loss' in losses:
            result['reg_loss'] = losses['reg_loss'].item()
        if 'lovasz_loss' in losses:
            val = losses['lovasz_loss']
            result['lovasz_loss'] = val.item() if torch.is_tensor(val) else float(val)

        # Factored diffusion metrics (S13)
        if 'occ_loss' in losses:
            result['occ_loss'] = losses['occ_loss'].item()
        if 'sem_loss' in losses:
            result['sem_loss'] = losses['sem_loss'].item()

        # S3: KD loss (either DSKD or output-level)
        if kd_loss_value is not None:
            result['dskd_loss'] = kd_loss_value.item() if hasattr(kd_loss_value, 'item') else kd_loss_value

        # S43: 3D Lovász loss metrics
        if ssc_lovasz_loss_value is not None:
            result['ssc_lovasz_loss'] = ssc_lovasz_loss_value

        # S44: Height-pool BEV aux loss metrics
        if hp_bev_loss_value is not None:
            result['hp_bev_loss'] = hp_bev_loss_value

        # V2: Aux BEV loss metrics
        if aux_bev_loss_value is not None:
            result['aux_bev_loss'] = aux_bev_loss_value
            result['ssc_loss'] = loss_ssc.item()

        # S41: E2E BEV loss metrics
        if bev_loss_e2e is not None:
            result['e2e_bev_loss'] = bev_loss_e2e.item()

        return result

    def _train_step_direct(self, lidar, gt_scene, bev, batch):
        """S48: Direct prediction training step — no diffusion.

        Single forward pass with uniform init as x_t, t=0.
        Loss: CE + Lovász + optional output KD.
        """
        B = lidar.shape[0]
        num_classes = self.config['num_classes']
        H, W, D = gt_scene.shape[1], gt_scene.shape[2], gt_scene.shape[3]

        # S40v2/S41: Compute BEV from BEV model on-the-fly (same logic as standard path)
        bev_loss_e2e = None
        if self.bev_model is not None and lidar.shape[1] > 1:
            bev_in_ch = self.bev_model.encoder.stem.weight.shape[-1]
            lidar_for_bev = lidar[:, :bev_in_ch]
            bev_coords_list = []
            bev_feats_list = []
            with torch.no_grad():
                occ_mask = (lidar_for_bev.abs().sum(dim=1) > 1e-6)
            for b in range(B):
                nz = occ_mask[b].nonzero()
                feats = lidar_for_bev[b, :, nz[:, 0], nz[:, 1], nz[:, 2]].T
                batch_col = torch.full((nz.shape[0], 1), b, device=self.device, dtype=torch.int32)
                bev_coords_list.append(torch.cat([batch_col, nz.int()], dim=1))
                bev_feats_list.append(feats)
            sparse_coords = torch.cat(bev_coords_list, dim=0)
            sparse_feats = torch.cat(bev_feats_list, dim=0)
            if self.use_e2e_bev:
                bev_logits = self.bev_model(sparse_feats, sparse_coords, B)
                bev = F.softmax(bev_logits, dim=1)
                gt_bev_for_loss = batch.get('gt_bev', batch['bev']).to(self.device).long()
                bev_loss_e2e = F.cross_entropy(bev_logits, gt_bev_for_loss)
            else:
                with torch.no_grad():
                    bev_logits = self.bev_model(sparse_feats, sparse_coords, B)
                    bev = bev_logits.argmax(dim=1)

        # Prepare BEV as one-hot for model input
        if bev.dim() == 4:
            bev_onehot = bev
        else:
            bev_onehot = F.one_hot(bev.long(), num_classes).float().permute(0, 3, 1, 2)

        # Initialize x_init: SCPNet (cold diffusion), lifted BEV (S3CNet), or uniform 1/K
        if self.config.get('cold_diffusion') and 'scpnet_pred' in batch:
            # B5: Use SCPNet pred as input — model learns to map SCPNet→GT
            # Compatible with cold diffusion Stage 2 (at t≈99, x_t ≈ SCPNet)
            _scp = batch['scpnet_pred'].to(self.device)
            x_init = F.one_hot(_scp.long(), num_classes).float().permute(0, 4, 1, 2, 3)
        elif self.dp_use_lifted_init and self.dp_lifter is not None:
            # Lift BEV to 3D using class-specific height priors → soft one-hot init
            x_init = self.dp_lifter.forward_soft(bev_onehot)  # [B, K, H, W, D]
        else:
            # Uniform init: equal probability for all classes (no diffusion noise)
            x_init = torch.ones(B, num_classes, H, W, D, device=self.device) / num_classes

        # t=0 (no noise)
        t = torch.zeros(B, device=self.device, dtype=torch.long)

        # Forward pass
        model_kwargs = {}
        if 'tsdf_bev' in batch:
            model_kwargs['geom_bev'] = batch['tsdf_bev'].to(self.device)
        if 'scpnet_pred' in batch:
            scpnet_labels = batch['scpnet_pred'].to(self.device)
            scpnet_onehot = F.one_hot(scpnet_labels, num_classes).float()
            model_kwargs['ssc_pred'] = scpnet_onehot.permute(0, 4, 1, 2, 3)
        logits = self.model(x_init, t, bev_onehot, lidar, **model_kwargs)  # [B, K, H, W, D]

        # CE loss (class-weighted, same weighting as diffusion)
        logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)
        target_flat = gt_scene.reshape(-1).long()
        ce_loss = F.cross_entropy(logits_flat, target_flat, weight=self.dp_ce_weights)

        # Lovász loss (directly optimizes mIoU)
        lovasz_loss = self.dp_lovasz(logits, gt_scene)
        loss = ce_loss + self.dp_lovasz_weight * lovasz_loss

        # S41: E2E BEV loss
        if bev_loss_e2e is not None:
            loss = loss + self.e2e_bev_weight * bev_loss_e2e

        # Output-level KD from teacher
        kd_loss_value = None
        if self.use_dskd and self.teacher_model is not None and self.kd_type == 'output':
            kd_weight = self.config.get('dskd_weight', 0.5)
            T = self.config.get('dskd_temperature', 4.0)

            # Teacher forward pass (same uniform init + t=0)
            teacher_bev = batch.get('teacher_bev', bev)
            if isinstance(teacher_bev, torch.Tensor):
                teacher_bev = teacher_bev.to(self.device)
            if teacher_bev.dim() == 3:
                teacher_bev_oh = F.one_hot(teacher_bev.long(), num_classes).float().permute(0, 3, 1, 2)
            else:
                teacher_bev_oh = teacher_bev

            with torch.no_grad():
                teacher_logits = self.teacher_model(x_init, t, teacher_bev_oh, lidar, **model_kwargs)

            # Soft label KD (Hinton-style)
            student_log_soft = F.log_softmax(logits / T, dim=1)
            teacher_soft = F.softmax(teacher_logits / T, dim=1)
            kl_per_voxel = F.kl_div(student_log_soft, teacher_soft, reduction='none').sum(dim=1)
            kd_loss_value = kl_per_voxel.mean() * (T * T)
            loss = loss + kd_weight * kd_loss_value

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        all_params = list(self.model.parameters())
        if self.use_e2e_bev and self.bev_model is not None:
            all_params += list(self.bev_model.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()

        # Metrics
        pred = logits.argmax(dim=1)
        accuracy = (pred == gt_scene).float().mean()

        result = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'ce_loss': ce_loss.item(),
            'lovasz_loss': lovasz_loss.item(),
        }
        if kd_loss_value is not None:
            result['dskd_loss'] = kd_loss_value.item()
        if bev_loss_e2e is not None:
            result['e2e_bev_loss'] = bev_loss_e2e.item()
        return result

    def _train_step_mimo(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        MIMO training step following PaSCo architecture exactly.

        PaSCo MIMO Architecture (from decoder_v3.py and ensembler.py):
        1. Input: Channel-concatenated LiDAR [B, N, H, W, D], stacked BEV [B, N, H, W]
        2. GT: Stacked scenes [B, N, H, W, D] - one per subnet
        3. Model: Single forward pass → N decoder head outputs
        4. Loss: Averaged across N heads (each head predicts its corresponding GT)
        5. Ensemble: mean_probs for inference (softmax first, then average)

        Reference: PaSCo CVPR 2024
        - decoder_v3.py lines 130-136: N separate completion heads
        - ensembler.py: ensemble_sem_compl() uses mean of softmax probs
        - kitti_dataset.py: 80% random crop, different samples per subnet
        """
        n_subnets = batch.get('n_subnets', 3)
        num_classes = self.config['num_classes']

        # Move to device
        # lidar: [B, N, H, W, D] channel-concatenated
        # bev: [B, N, H, W] stacked
        # gt_scene: [B, N, H, W, D] stacked
        lidar = batch['lidar'].to(self.device)
        bev = batch['bev'].to(self.device)
        gt_scene = batch['gt_scene'].to(self.device)

        if self.config.get('bev_from_base') and 'scpnet_pred' in batch:
            # Topmost non-empty class along Z (matches GT BEV computation)
            scp = batch['scpnet_pred'].to(self.device)
            bev = torch.zeros(scp.shape[0], scp.shape[1], scp.shape[2],
                               dtype=torch.long, device=self.device)
            for z in range(scp.shape[3] - 1, -1, -1):
                layer = scp[:, :, :, z]
                mask = (layer > 0) & (bev == 0)
                bev[mask] = layer[mask]

        B = lidar.shape[0]
        _H, _W, _D = lidar.shape[2], lidar.shape[3], lidar.shape[4]

        # Sample timesteps (same for all subnets in batch; uses configurable sampling mode)
        t = self._sample_train_timesteps(B)

        # Create noisy x_t for each subnet's GT and concatenate
        # PaSCo processes all N samples together in single forward pass
        x_t_list = []
        x_0_onehot_list = []

        for i in range(n_subnets):
            gt_i = gt_scene[:, i]  # [B, H, W, D]

            # Convert to one-hot
            x_0_onehot = F.one_hot(gt_i.long(), num_classes).float()  # [B, H, W, D, C]
            x_0_onehot = x_0_onehot.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
            x_0_onehot_list.append(x_0_onehot)

            # Add noise using diffusion transition
            x_t_onehot = self.diffusion.q_sample(x_0_onehot, t)  # [B, C, H, W, D]
            x_t_list.append(x_t_onehot)

        # Channel-concatenate noisy voxels for single forward pass
        # Result: [B, N*C, H, W, D]
        x_t_concat = torch.cat(x_t_list, dim=1)
        torch.cat(x_0_onehot_list, dim=1)

        # Forward through MIMO model
        # Model expects: x_t [B, N*C, H, W, D], t [B], bev [B, N, H, W], lidar [B, N, H, W, D]
        # WaffleIron is optional (None uses zeros fallback)
        waffleiron = batch.get('waffleiron', None)
        if waffleiron is not None and isinstance(waffleiron, torch.Tensor):
            waffleiron = waffleiron.to(self.device)
        else:
            waffleiron = None  # Not available, model uses fallback
        outputs = self.model(x_t_concat, t, bev, lidar, waffleiron=waffleiron)

        # Get N decoder head outputs
        head_outputs = outputs['head_outputs']  # List of N tensors [B, C, H, W, D]

        # Compute loss for each head against corresponding GT
        total_loss = 0.0
        total_acc = 0.0
        head_losses = []

        for i, logits in enumerate(head_outputs):
            gt_i = gt_scene[:, i]  # [B, H, W, D]

            # Cross-entropy loss
            # logits: [B, C, H, W, D] → [B, C, -1] → [-1, C]
            # target: [B, H, W, D] → [-1]
            logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)
            target_flat = gt_i.reshape(-1)

            # Class weights (downweight class 0 = empty)
            class_0_weight = self.config.get('class_0_weight', 0.02)
            weights = torch.ones(num_classes, device=self.device)
            weights[0] = class_0_weight

            loss_i = F.cross_entropy(logits_flat, target_flat, weight=weights)
            head_losses.append(loss_i)
            total_loss += loss_i

            # Accuracy
            pred = logits.argmax(dim=1)  # [B, H, W, D]
            acc_i = (pred == gt_i).float().mean()
            total_acc += acc_i

        # Average loss across heads
        avg_loss = total_loss / n_subnets
        avg_acc = total_acc / n_subnets

        # Backward
        self.optimizer.zero_grad()
        avg_loss.backward()

        # Gradient clipping
        all_params = list(self.model.parameters()) + (list(self.lifter.parameters()) if self.lifter is not None else [])
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Update EMA
        self.ema.update()

        result = {
            'loss': avg_loss.item(),
            'accuracy': avg_acc.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'mimo_n_heads': n_subnets,
        }

        # Add per-head losses for monitoring
        for i, hl in enumerate(head_losses):
            result[f'head_{i}_loss'] = hl.item()

        return result

    @torch.no_grad()
    def validate(self, full_sampling: bool = False) -> dict[str, float]:
        """
        Validate model with SemanticKITTI SSC metrics.

        Args:
            full_sampling: If True, run full diffusion sampling for mIoU.
                          If False, only compute loss (faster, for frequent validation).
        """
        self.model.eval()
        if self.lifter is not None:
            self.lifter.eval()
        if self.bev_model is not None:
            self.bev_model.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        # SSC metrics evaluator
        ssc_metrics = SSCMetrics(num_classes=self.config['num_classes']) if full_sampling else None
        ssc_metrics_spsr = SSCMetrics(num_classes=self.config['num_classes']) if (full_sampling and self.dp_use_spsr) else None
        # SCPNet baseline metrics on the SAME samples (for direct comparison)
        scpnet_baseline_metrics = SSCMetrics(num_classes=self.config['num_classes']) if full_sampling else None

        # For mIoU evaluation, use a random subset to speed up
        # FIXED seed selects FRAME indices (not batch indices) for cross-experiment consistency
        miou_samples = self.config.get('miou_samples', 100)
        total_val_samples = len(self.val_loader.dataset)
        batch_size = self.config['batch_size']
        total_batches = len(self.val_loader)

        if full_sampling and miou_samples < total_val_samples:
            # Select frame indices directly — consistent regardless of batch_size
            rng = np.random.RandomState(42)
            miou_frame_indices = set(rng.choice(total_val_samples, min(miou_samples, total_val_samples), replace=False))
            # Determine which batches contain selected frames
            miou_batch_indices = set(idx // batch_size for idx in miou_frame_indices)
        else:
            miou_frame_indices = set(range(total_val_samples))
            miou_batch_indices = set(range(total_batches))

        miou_sample_count = 0

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
            lidar = batch['lidar'].to(self.device)
            gt_scene = batch['gt_scene'].to(self.device)
            bev = batch['bev'].to(self.device)

            if self.config.get('bev_from_base') and 'scpnet_pred' in batch:
                # Topmost non-empty class along Z (consistent with training)
                scp = batch['scpnet_pred'].to(self.device)
                bev = torch.zeros(scp.shape[0], scp.shape[1], scp.shape[2],
                                   dtype=torch.long, device=self.device)
                for z in range(scp.shape[3] - 1, -1, -1):
                    layer = scp[:, :, :, z]
                    mask = (layer > 0) & (bev == 0)
                    bev[mask] = layer[mask]

            # S20: Replace binary LiDAR with LSK3DNet 3D soft probs (20ch)
            if self.config.get('use_lsk3d', False) and 'lsk3d_3d_probs' in batch:
                lidar = batch['lsk3d_3d_probs'].to(self.device)

            # S40v2/S41: Compute soft BEV from BEV model for validation
            if self.bev_model is not None and lidar.shape[1] > 1:
                # S43 FIX: Truncate to BEV model's expected channels (20ch)
                bev_in_ch = self.bev_model.encoder.stem.weight.shape[-1]
                lidar_for_bev = lidar[:, :bev_in_ch]
                B_bev = lidar.shape[0]
                bev_coords_list = []
                bev_feats_list = []
                occ_mask = (lidar_for_bev.abs().sum(dim=1) > 1e-6)
                for b in range(B_bev):
                    nz = occ_mask[b].nonzero()
                    feats = lidar_for_bev[b, :, nz[:, 0], nz[:, 1], nz[:, 2]].T
                    batch_col = torch.full((nz.shape[0], 1), b, device=self.device, dtype=torch.int32)
                    bev_coords_list.append(torch.cat([batch_col, nz.int()], dim=1))
                    bev_feats_list.append(feats)
                sparse_coords = torch.cat(bev_coords_list, dim=0)
                sparse_feats = torch.cat(bev_feats_list, dim=0)
                bev_logits = self.bev_model(sparse_feats, sparse_coords, B_bev)
                if self.use_e2e_bev:
                    bev = F.softmax(bev_logits, dim=1)  # [B, 20, H, W] soft BEV (E2E)
                else:
                    bev = bev_logits.argmax(dim=1)  # [B, H, W] hard class indices (frozen)

            model_type = self.config.get('model_type', 'full')
            is_mimo_model = model_type in ['mimo_lite', 'mimo_full']
            is_mimo_batch = batch.get('channel_concatenated', False)

            # For MIMO validation loss, we use the full MIMO forward pass
            # (same approach as training to get consistent metrics)
            if is_mimo_model and is_mimo_batch:
                n_subnets = batch.get('n_subnets', 3)
                B = lidar.shape[0]

                # Sample timesteps (configurable sampling mode)
                t = self._sample_train_timesteps(B)

                # Create noisy x_t for each subnet's GT and concatenate
                x_t_list = []
                for i in range(n_subnets):
                    gt_i = gt_scene[:, i]  # [B, H, W, D]
                    x_0_onehot = F.one_hot(gt_i.long(), self.config['num_classes']).float()
                    x_0_onehot = x_0_onehot.permute(0, 4, 1, 2, 3)  # [B, C, H, W, D]
                    x_t_onehot = self.diffusion.q_sample(x_0_onehot, t)
                    x_t_list.append(x_t_onehot)

                x_t_concat = torch.cat(x_t_list, dim=1)

                # Forward through MIMO model (waffleiron optional)
                waffleiron = batch.get('waffleiron', None)
                if waffleiron is not None and isinstance(waffleiron, torch.Tensor):
                    waffleiron = waffleiron.to(self.device)
                else:
                    waffleiron = None  # Not available, model uses fallback
                outputs = self.model(x_t_concat, t, bev, lidar, waffleiron=waffleiron)

                # Compute loss for each head
                total_head_loss = 0.0
                total_head_acc = 0.0
                class_0_weight = self.config.get('class_0_weight', 0.02)
                weights = torch.ones(self.config['num_classes'], device=self.device)
                weights[0] = class_0_weight

                for i, logits in enumerate(outputs['head_outputs']):
                    gt_i = gt_scene[:, i]
                    logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, self.config['num_classes'])
                    target_flat = gt_i.reshape(-1)
                    loss_i = F.cross_entropy(logits_flat, target_flat, weight=weights)
                    total_head_loss += loss_i

                    pred = logits.argmax(dim=1)
                    acc_i = (pred == gt_i).float().mean()
                    total_head_acc += acc_i

                avg_loss = total_head_loss / n_subnets
                avg_acc = total_head_acc / n_subnets

                total_loss += avg_loss.item()
                total_acc += avg_acc.item()
            elif self.direct_prediction:
                # S48: Direct prediction validation — single forward pass
                B = lidar.shape[0]
                num_classes = self.config['num_classes']
                H, W, D_dim = gt_scene.shape[1], gt_scene.shape[2], gt_scene.shape[3]

                if bev.dim() == 4:
                    bev_onehot = bev
                else:
                    bev_onehot = F.one_hot(bev.long(), num_classes).float().permute(0, 3, 1, 2)

                # Use lifted BEV init if enabled
                if self.dp_use_lifted_init and self.dp_lifter is not None:
                    x_init = self.dp_lifter.forward_soft(bev_onehot)
                else:
                    x_init = torch.ones(B, num_classes, H, W, D_dim, device=self.device) / num_classes
                t = torch.zeros(B, device=self.device, dtype=torch.long)

                val_kwargs = {}
                if 'tsdf_bev' in batch:
                    val_kwargs['geom_bev'] = batch['tsdf_bev'].to(self.device)
                logits = self.model(x_init, t, bev_onehot, lidar, **val_kwargs)

                logits_flat = logits.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)
                target_flat = gt_scene.reshape(-1).long()
                val_loss = F.cross_entropy(logits_flat, target_flat, weight=self.dp_ce_weights)
                pred = logits.argmax(dim=1)
                val_acc = (pred == gt_scene).float().mean()

                total_loss += val_loss.item()
                total_acc += val_acc.item()
            else:
                # Standard validation
                # Compute lifted features (same as training)
                val_lifted = None
                if self.config.get('use_lifting', False) and self.lifter is not None:
                    if bev.dim() == 4:
                        bev_one_hot = bev
                    else:
                        bev_one_hot = F.one_hot(bev.long(), self.config['num_classes']).float()
                        bev_one_hot = bev_one_hot.permute(0, 3, 1, 2)
                    _, val_lifted = self.lifter(bev_one_hot)

                B = lidar.shape[0]
                t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
                val_nn_indices = batch.get('nn_indices', None)
                if val_nn_indices is not None:
                    val_nn_indices = val_nn_indices.to(self.device)
                val_kwargs = {}
                if 'tsdf_bev' in batch:
                    val_kwargs['geom_bev'] = batch['tsdf_bev'].to(self.device)
                if 'scpnet_pred' in batch:
                    _scp = batch['scpnet_pred'].to(self.device)
                    _scp_oh = F.one_hot(_scp.long(), self.config['num_classes']).float()
                    val_kwargs['ssc_pred'] = _scp_oh.permute(0, 4, 1, 2, 3)
                if self.config.get('obs_mask_channel', False):
                    if lidar.shape[1] == 1:
                        val_kwargs['obs_mask'] = (lidar > 0).float()
                    else:
                        val_kwargs['obs_mask'] = (lidar.abs().sum(dim=1, keepdim=True) > 0).float()
                if self.config.get('cold_diffusion') and 'scpnet_pred' in batch:
                    _scp_v = batch['scpnet_pred'].to(self.device)
                    val_kwargs['x_scpnet'] = F.one_hot(_scp_v.long(), self.config['num_classes']).float().permute(0, 4, 1, 2, 3)
                losses = self.diffusion.training_losses(
                    self.model,
                    gt_scene,
                    t,
                    bev,
                    lidar,
                    lifted_features=val_lifted,
                    nn_indices=val_nn_indices,
                    **val_kwargs,
                )
                total_loss += losses['loss'].item()
                total_acc += losses['accuracy'].item()

            num_batches += 1

            # Full diffusion sampling for mIoU (only on fixed-seed selected subset)
            if full_sampling and batch_idx in miou_batch_indices:
              # Determine which samples in this batch are selected
              B_cur = gt_scene.shape[0]
              selected_in_batch = []
              for b_i in range(B_cur):
                  frame_idx = batch_idx * batch_size + b_i
                  if frame_idx in miou_frame_indices:
                      selected_in_batch.append(b_i)
              if not selected_in_batch:
                  continue  # No selected frames in this batch

              # Invalid mask for official-style eval (excludes unobservable voxels)
              inv_mask = batch.get('invalid_mask', None)
              if inv_mask is not None:
                  inv_mask = inv_mask.cpu().numpy()

              # Compute SCPNet/TALoS baseline on selected samples only
              if scpnet_baseline_metrics is not None and 'scpnet_pred' in batch:
                  for b_i in selected_in_batch:
                      inv_i = inv_mask[b_i] if inv_mask is not None else None
                      scpnet_baseline_metrics.update(
                          batch['scpnet_pred'].cpu().numpy()[b_i],
                          batch['gt_scene'].cpu().numpy()[b_i],
                          invalid_mask=inv_i,
                      )
              try:
                # S48: Direct prediction — single forward pass for mIoU (no diffusion sampling needed)
                if self.direct_prediction:
                    B = lidar.shape[0]
                    num_classes = self.config['num_classes']
                    H, W, D_dim = gt_scene.shape[1], gt_scene.shape[2], gt_scene.shape[3]

                    if bev.dim() == 4:
                        bev_onehot = bev
                    else:
                        bev_onehot = F.one_hot(bev.long(), num_classes).float().permute(0, 3, 1, 2)

                    # Use lifted BEV init if enabled, else uniform
                    if self.dp_use_lifted_init and self.dp_lifter is not None:
                        x_init = self.dp_lifter.forward_soft(bev_onehot)
                    else:
                        x_init = torch.ones(B, num_classes, H, W, D_dim, device=self.device) / num_classes
                    t_zero = torch.zeros(B, device=self.device, dtype=torch.long)

                    sample_kwargs = {}
                    if 'tsdf_bev' in batch:
                        sample_kwargs['geom_bev'] = batch['tsdf_bev'].to(self.device)
                    if 'scpnet_pred' in batch:
                        _scp = batch['scpnet_pred'].to(self.device)
                        _scp_oh = F.one_hot(_scp.long(), num_classes).float()
                        sample_kwargs['ssc_pred'] = _scp_oh.permute(0, 4, 1, 2, 3)
                    logits = self.model(x_init, t_zero, bev_onehot, lidar, **sample_kwargs)
                    pred_scene = logits.argmax(dim=1)  # [B, H, W, D]

                    # Always track raw (non-SPSR) metrics
                    ssc_metrics.update(
                        pred_scene.cpu().numpy(),
                        gt_scene.cpu().numpy(),
                        invalid_mask=inv_mask,
                    )

                    # Additionally track SPSR metrics if enabled
                    if ssc_metrics_spsr is not None and hasattr(self, 'dp_spsr'):
                        if lidar.shape[1] == 1:
                            lidar_obs = (lidar > 0).float()
                        else:
                            lidar_obs = (lidar.abs().sum(dim=1, keepdim=True) > 1e-6).float()
                        pred_probs = F.softmax(logits, dim=1)
                        pred_spsr = self.dp_spsr(pred_scene.clone(), lidar_obs=lidar_obs, pred_probs=pred_probs)
                        ssc_metrics_spsr.update(
                            pred_spsr.cpu().numpy(),
                            gt_scene.cpu().numpy(),
                        )

                    miou_sample_count += B
                    continue  # Skip diffusion sampling below

                model_type = self.config.get('model_type', 'full')
                is_mimo_model = model_type in ['mimo_lite', 'mimo_full']
                is_mimo_batch = batch.get('channel_concatenated', False)

                # S4 MIMO: Use sample_mimo() for MIMO models
                if is_mimo_model and is_mimo_batch:
                    # For MIMO validation, inputs are stacked: lidar [B, N, H, W, D], bev [B, N, H, W]
                    # gt_scene is [B, N, H, W, D] but all N are same sample, use [:, 0]
                    n_subnets = batch.get('n_subnets', 3)
                    gt_scene_single = gt_scene[:, 0]  # [B, H, W, D]

                    # Extract single sample for sampling (all N are same)
                    lidar_single = lidar[:, 0:1]  # [B, 1, H, W, D]
                    bev_single = bev[:, 0]  # [B, H, W]

                    B = lidar_single.shape[0]
                    H, W, D = gt_scene_single.shape[1], gt_scene_single.shape[2], gt_scene_single.shape[3]

                    # Use MIMO sampling - DO NOT use TTA!
                    # PaSCo reference: validation uses N IDENTICAL copies (not TTA)
                    # - Training: N different scenes merged (NOT N augmented copies of same scene)
                    # - Validation: N identical copies, outputs averaged
                    # Testing confirmed: TTA hurts (80% acc without vs 53% with TTA)
                    # Get waffleiron for MIMO sampling (required by sample_mimo)
                    # MIMO dataset returns waffleiron as [B, N, C, H, W] but sample_mimo expects [B, C, H, W]
                    # For validation, all N subnets have identical waffleiron, so take the first one
                    waffleiron_sample = batch.get('waffleiron', None)
                    if waffleiron_sample is not None and isinstance(waffleiron_sample, torch.Tensor):
                        waffleiron_sample = waffleiron_sample.to(self.device)
                        # Extract first subnet's waffleiron: [B, N, C, H, W] -> [B, C, H, W]
                        if waffleiron_sample.dim() == 5:
                            waffleiron_sample = waffleiron_sample[:, 0]  # Take first subnet

                    # PaSCo MIMO validation: Same scene with N different augmentations
                    # Each head sees scene from different viewpoint, ensemble averages
                    # Previous use_tta=False was WRONG - caused train/val mismatch:
                    # - Training: Each head sees DIFFERENT scene (diverse inputs)
                    # - Val with use_tta=False: Each head sees IDENTICAL scene (no diversity)
                    # - Result: Heads specialized for other scenes produce garbage
                    # Fix: use_tta=True gives each head different viewpoint of same scene
                    pred_scene = self.diffusion.sample_mimo(
                        self.model,
                        bev_single,
                        lidar_single,
                        waffleiron=waffleiron_sample,
                        n_subnets=n_subnets,
                        shape=(B, H, W, D),
                        device=self.device,
                        show_progress=False,
                        use_tta=True,  # PaSCo-style: N augmentations + inverse transforms + ensemble
                        use_continuous_tta=True,  # PaSCo uses continuous rotation
                        tta_max_angle=30.0,  # PaSCo default: ±30°
                        tta_max_translation=(0.6, 0.6, 0.4),  # PaSCo default
                    )

                    # Update SSC metrics
                    ssc_metrics.update(
                        pred_scene.cpu().numpy(),
                        gt_scene_single.cpu().numpy(),
                        invalid_mask=inv_mask,
                    )
                    miou_sample_count += B
                else:
                    # Standard (non-MIMO) sampling
                    B = lidar.shape[0]
                    H, W, D = gt_scene.shape[1], gt_scene.shape[2], gt_scene.shape[3]

                    # S25-S27: Get precomputed NN indices for densification
                    sample_nn_indices = batch.get('nn_indices', None)
                    if sample_nn_indices is not None:
                        sample_nn_indices = sample_nn_indices.to(self.device)

                    # S2: Use SDEdit if lifting is enabled
                    use_sdedit = self.config.get('use_sdedit', False) and self.lifter is not None
                    if use_sdedit:
                        # Compute lifted 3D and features from BEV
                        if bev.dim() == 4:
                            bev_one_hot = bev
                        else:
                            bev_one_hot = F.one_hot(bev.long(), self.config['num_classes']).float()
                            bev_one_hot = bev_one_hot.permute(0, 3, 1, 2)  # [B, C, H, W]
                        lifted_3d, sdedit_lifted = self.lifter(bev_one_hot)

                        # SDEdit: Start from lifted 3D + noise
                        sdedit_start = self.config.get('sdedit_start_step', 50)
                        pred_scene = self.diffusion.sample_sdedit(
                            self.model,
                            bev,
                            lidar,
                            lifted_3d,
                            shape=(B, H, W, D),
                            device=self.device,
                            start_timestep=sdedit_start,
                            show_progress=False,
                            lifted_features=sdedit_lifted,
                        )
                    else:
                        # Compute lifted features for sampling
                        sample_lifted = None
                        if self.config.get('use_lifting', False) and self.lifter is not None:
                            if bev.dim() == 4:
                                bev_oh = bev
                            else:
                                bev_oh = F.one_hot(bev.long(), self.config['num_classes']).float()
                                bev_oh = bev_oh.permute(0, 3, 1, 2)
                            _, sample_lifted = self.lifter(bev_oh)

                        # Use CFG sampling if available
                        cfg_scale = self.config.get('cfg_guidance_scale', 1.0)
                        cfg_drop = self.config.get('cfg_drop_prob', 0.0)
                        model_type = self.config.get('model_type', 'full')
                        diffusion_version = self.config.get('diffusion_version', 'v1')

                        is_v3_cfg = model_type in ('v3_coarse2fine', 'v3_c2f_ablation') and cfg_drop > 0 and cfg_scale > 1.0
                        is_v4 = model_type in ('v4_continuous', 'v4_factored')
                        is_v5_ve = model_type == 'v5_ve'

                        # Disable CFG for V3/V4 models during early training.
                        # CFG requires a well-trained unconditional model (cfg_drop_prob=0.1
                        # means only 10% of steps train uncond path). Using CFG with an
                        # under-trained uncond model produces garbage: guided = uncond + w*(cond-uncond)
                        # amplifies noise when uncond output is random. Non-CFG sampling uses
                        # the conditional model directly, which works immediately.
                        # TODO: Re-enable CFG after sufficient training (e.g. >50K steps).
                        use_cfg_at_eval = False  # Disabled: uncond model under-trained early on

                        if is_v5_ve and diffusion_version == 'gaussian_ve':
                            # S16+: VE Gaussian diffusion — use DPM-Solver++ with SDEdit
                            dpm_steps = self.config.get('dpm_solver_steps', 50)
                            # For VE diffusion, we use sample_dpm_solver which handles
                            # initialization from uniform probs (no SDEdit by default)
                            pred_scene = self.diffusion.sample_dpm_solver(
                                self.model,
                                bev,
                                lidar,
                                shape=(B, H, W, D),
                                device=self.device,
                                num_steps=dpm_steps,
                                show_progress=False,
                            )
                        elif diffusion_version == 'gaussian_logit':
                            # S17+: Logit-space VE Gaussian diffusion
                            dpm_steps = self.config.get('dpm_solver_steps', 50)
                            sdedit_start = self.config.get('sdedit_start_step', 500)

                            # cond_3d: LSK3DNet 3D predictions as conditioning
                            eval_cond_3d = None
                            if 'lsk3d_3d_probs' in batch and self.config.get('cond_3d_channels', 0) > 0:
                                eval_cond_3d = batch['lsk3d_3d_probs'].to(self.device)

                            # SDEdit init (if configured): start from noised predictions
                            init_probs = None
                            if sdedit_start < self.diffusion.num_timesteps - 1:
                                if 'lsk3d_3d_probs' in batch:
                                    init_probs = batch['lsk3d_3d_probs'].to(self.device)
                                elif self.lifter is not None:
                                    if bev.dim() == 4:
                                        bev_oh = bev
                                    else:
                                        bev_oh = F.one_hot(bev.long(), self.config['num_classes']).float()
                                        bev_oh = bev_oh.permute(0, 3, 1, 2)
                                    lifted_3d, _ = self.lifter(bev_oh)
                                    init_probs = lifted_3d

                            pred_scene = self.diffusion.sample_dpm_solver(
                                self.model,
                                bev,
                                lidar,
                                shape=(B, H, W, D),
                                device=self.device,
                                cond_3d=eval_cond_3d,
                                init_probs=init_probs,
                                start_timestep=sdedit_start if init_probs is not None else None,
                                num_steps=dpm_steps,
                                guidance_scale=cfg_scale,
                                show_progress=False,
                            )
                        elif diffusion_version == 'gaussian_vp':
                            # S21: VP DDPM with 20ch logit encoding
                            # Pure-noise generation: model completes scene from
                            # Gaussian noise conditioned on LSK3DNet features.
                            #
                            # SDEdit is WRONG for scene completion:
                            # - LSK3DNet only predicts at ~1% of voxels (observed)
                            # - Unobserved voxels encoded as class 0 (empty)
                            # - At t=200, noise too mild to override "empty" bias
                            # - Result: recall locked at 11.8% (= LSK3DNet baseline)
                            #
                            # Pure-noise generation works because:
                            # - Dense conditioning (coarse2fine) provides layout at ALL voxels
                            # - Sparse FiLM provides high-quality features at observed voxels
                            # - Model trained to denoise from pure noise → generates full scene
                            dpm_steps = self.config.get('dpm_solver_steps', 50)

                            # CFG: requires well-trained unconditional model.
                            # With cfg_drop_prob=0.1, uncond gets trained 10% of steps.
                            # Disable CFG until unconditional model has enough training.
                            cfg_min_step = self.config.get('cfg_min_step', 50000)
                            if self.global_step < cfg_min_step:
                                eval_cfg_scale = 1.0
                            else:
                                eval_cfg_scale = cfg_scale

                            if cfg_drop > 0 and eval_cfg_scale > 1.0:
                                # CFG + DPM-Solver++ from pure noise
                                pred_scene = self.diffusion.sample_cfg(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    guidance_scale=eval_cfg_scale,
                                    num_steps=dpm_steps,
                                    show_progress=False,
                                )
                            else:
                                # DPM-Solver++ without CFG
                                pred_scene = self.diffusion.sample_dpm_solver(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    num_steps=dpm_steps,
                                    show_progress=False,
                                )
                        elif is_v4 and diffusion_version == 'gaussian':
                            # S14/S15: Gaussian diffusion — use DPM-Solver++
                            dpm_steps = self.config.get('dpm_solver_steps', 50)
                            if use_cfg_at_eval and cfg_drop > 0 and cfg_scale > 1.0:
                                pred_scene = self.diffusion.sample_cfg(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    guidance_scale=cfg_scale,
                                    num_steps=dpm_steps,
                                    show_progress=False,
                                )
                            else:
                                pred_scene = self.diffusion.sample_dpm_solver(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    num_steps=dpm_steps,
                                    show_progress=False,
                                )
                        elif is_v4 and diffusion_version == 'factored':
                            # S13: Factored discrete — ancestral sampling
                            if use_cfg_at_eval and cfg_drop > 0 and cfg_scale > 1.0:
                                pred_scene = self.diffusion.sample_cfg(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    guidance_scale=cfg_scale,
                                    show_progress=False,
                                )
                            else:
                                pred_scene = self.diffusion.sample(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    show_progress=False,
                                )
                        elif is_v3_cfg:
                            # V3.1: non-CFG sampling (plain sample)
                            if use_cfg_at_eval:
                                pred_scene = self.diffusion.sample_cfg(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    lifted_features=sample_lifted,
                                    guidance_scale=cfg_scale,
                                    cfg_target='lidar',
                                    show_progress=False,
                                    nn_indices=sample_nn_indices,
                                )
                            else:
                                pred_scene = self.diffusion.sample(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    lifted_features=sample_lifted,
                                    show_progress=False,
                                    nn_indices=sample_nn_indices,
                                )
                        elif sample_lifted is not None and cfg_scale > 1.0:
                            pred_scene = self.diffusion.sample_cfg(
                                self.model,
                                bev,
                                lidar,
                                shape=(B, H, W, D),
                                device=self.device,
                                lifted_features=sample_lifted,
                                guidance_scale=cfg_scale,
                                show_progress=False,
                                nn_indices=sample_nn_indices,
                            )
                        else:
                            sample_kwargs = {}
                            if 'tsdf_bev' in batch:
                                sample_kwargs['geom_bev'] = batch['tsdf_bev'].to(self.device)
                            if 'scpnet_pred' in batch:
                                _scp = batch['scpnet_pred'].to(self.device)
                                _scp_oh = F.one_hot(_scp.long(), self.config['num_classes']).float()
                                sample_kwargs['ssc_pred'] = _scp_oh.permute(0, 4, 1, 2, 3)
                            if self.config.get('obs_mask_channel', False):
                                if lidar.shape[1] == 1:
                                    sample_kwargs['obs_mask'] = (lidar > 0).float()
                                else:
                                    sample_kwargs['obs_mask'] = (lidar.abs().sum(dim=1, keepdim=True) > 0).float()
                            # B2: Repaint sampling — anchor observed voxels
                            if self.config.get('use_repaint') and 'scpnet_pred' in batch:
                                _obs = (lidar[:, :1] > 0).float() if lidar.shape[1] == 1 else \
                                       (lidar.abs().sum(dim=1, keepdim=True) > 0).float()
                                # Remove obs_mask from kwargs to avoid duplicate with explicit param
                                repaint_kwargs = {k: v for k, v in sample_kwargs.items() if k != 'obs_mask'}
                                pred_scene = self.diffusion.sample_repaint(
                                    self.model, bev, lidar,
                                    known_labels=batch['scpnet_pred'].to(self.device),
                                    obs_mask=_obs,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    show_progress=False,
                                    nn_indices=sample_nn_indices,
                                    **repaint_kwargs,
                                )
                            # B4/B5: structured-source forward — always use S2D2 correction sampling
                            elif self.config.get('cold_diffusion') and 'scpnet_pred' in batch:
                                _scp = batch['scpnet_pred'].to(self.device)
                                _scp_oh = F.one_hot(_scp.long(), self.config['num_classes']).float()
                                sample_kwargs['x_scpnet'] = _scp_oh.permute(0, 4, 1, 2, 3)
                                correction_steps = self.config.get(
                                    'correction_eval_steps',
                                    self.config.get('algo2_eval_steps', 100),
                                )
                                pred_scene = self.diffusion.sample_algo2(
                                    self.model, bev, lidar,
                                    scpnet_pred=_scp,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    n_steps=correction_steps,
                                    show_progress=False,
                                    **sample_kwargs,
                                )
                            else:
                                pred_scene = self.diffusion.sample(
                                    self.model,
                                    bev,
                                    lidar,
                                    shape=(B, H, W, D),
                                    device=self.device,
                                    lifted_features=sample_lifted,
                                    show_progress=False,
                                    nn_indices=sample_nn_indices,
                                    **sample_kwargs,
                                )
                    # pred_scene: [B, H, W, D] class indices

                    # Update SSC metrics (only for selected frames in batch)
                    for b_i in selected_in_batch:
                        inv_i = inv_mask[b_i] if inv_mask is not None else None
                        ssc_metrics.update(
                            pred_scene.cpu().numpy()[b_i],
                            gt_scene.cpu().numpy()[b_i],
                            invalid_mask=inv_i,
                        )
                    miou_sample_count += len(selected_in_batch)
              except Exception as e:
                    logging.warning(f"mIoU sampling failed for batch {batch_idx}: {e}")
                    continue

        results = {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_acc / num_batches,
        }

        # Add SSC metrics if full sampling
        if full_sampling and ssc_metrics is not None:
            iou_results = ssc_metrics.get_iou()
            results['mIoU'] = iou_results['mIoU']
            results['iou_completion'] = iou_results['iou_completion']
            results['precision'] = iou_results['precision']
            results['recall'] = iou_results['recall']
            results['class_iou'] = iou_results.get('class_iou', {})

        # Add SCPNet baseline on same samples for direct comparison
        if full_sampling and scpnet_baseline_metrics is not None:
            try:
                scpnet_results = scpnet_baseline_metrics.get_iou()
                results['scpnet_mIoU'] = scpnet_results['mIoU']
                results['scpnet_iou_completion'] = scpnet_results['iou_completion']
            except Exception:
                pass  # No SCPNet predictions available

        # Add SPSR metrics if tracked
        if full_sampling and ssc_metrics_spsr is not None:
            spsr_results = ssc_metrics_spsr.get_iou()
            results['mIoU_spsr'] = spsr_results['mIoU']

        # Restore train mode
        if self.lifter is not None:
            self.lifter.train()
        # BEV model stays in eval() always — BN must use pretrained running stats
        # (batch_size=1 makes train-mode BN stats unreliable)

        return results

    def _get_curriculum_gt_bev_prob(self) -> float:
        """Compute current gt_bev_prob based on curriculum schedule."""
        config = self.config
        if not config.get('curriculum', False):
            return config.get('gt_bev_prob', 0.0)

        start_prob = config.get('curriculum_start_prob', 1.0)
        end_prob = config.get('curriculum_end_prob', 0.0)
        warmup_frac = config.get('curriculum_warmup_frac', 0.6)
        total_steps = config['num_iterations']
        warmup_steps = int(total_steps * warmup_frac)

        if self.global_step >= warmup_steps:
            return end_prob
        else:
            # Linear interpolation from start_prob to end_prob
            progress = self.global_step / max(warmup_steps, 1)
            return start_prob + (end_prob - start_prob) * progress

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")

        # Log curriculum settings
        if self.config.get('curriculum', False):
            self.logger.info(
                f"Curriculum learning ENABLED: gt_bev_prob "
                f"{self.config['curriculum_start_prob']:.2f} → {self.config['curriculum_end_prob']:.2f} "
                f"over first {self.config['curriculum_warmup_frac']*100:.0f}% of training "
                f"({int(self.config['num_iterations'] * self.config['curriculum_warmup_frac'])} steps)"
            )

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.config['num_iterations'], initial=self.global_step, desc="Training")

        while self.global_step < self.config['num_iterations']:
            # Curriculum learning: update gt_bev_prob dynamically
            if self.config.get('curriculum', False):
                new_prob = self._get_curriculum_gt_bev_prob()
                dataset = self.train_loader.dataset
                # If wrapped by MIMODatasetWrapper, access the base dataset
                if hasattr(dataset, 'base_dataset'):
                    dataset = dataset.base_dataset
                if hasattr(dataset, 'gt_bev_prob'):
                    dataset.gt_bev_prob = new_prob

            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            metrics = self.train_step(batch)
            self.global_step += 1

            # Update progress bar
            pbar.update(1)
            postfix = {
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'lr': f"{metrics['lr']:.2e}",
            }
            if 'dskd_loss' in metrics:
                postfix['dskd'] = f"{metrics['dskd_loss']:.4f}"
            if 'aux_bev_loss' in metrics:
                postfix['bev'] = f"{metrics['aux_bev_loss']:.4f}"
            pbar.set_postfix(postfix)

            # Log
            if self.global_step % self.config.get('log_interval', 100) == 0:
                log_msg = f"Step {self.global_step}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}"

                # Add V2 metrics if available
                if 'occupied_accuracy' in metrics:
                    log_msg += f", occ_acc={metrics['occupied_accuracy']:.4f}"
                if 'kl_loss' in metrics:
                    log_msg += f", kl={metrics['kl_loss']:.4f}"
                if 'lovasz_loss' in metrics:
                    log_msg += f", lovasz={metrics['lovasz_loss']:.4f}"
                if 'auxiliary_loss' in metrics:
                    log_msg += f", aux={metrics['auxiliary_loss']:.4f}"
                # S48: Direct prediction CE loss
                if 'ce_loss' in metrics:
                    log_msg += f", ce={metrics['ce_loss']:.4f}"
                # S43: Log 3D Lovász loss
                if 'ssc_lovasz_loss' in metrics:
                    log_msg += f", lov3d={metrics['ssc_lovasz_loss']:.4f}"
                # S44: Log height-pool BEV aux loss
                if 'hp_bev_loss' in metrics:
                    log_msg += f", hpbev={metrics['hp_bev_loss']:.4f}"
                # S3 DSKD: Log DSKD loss if computed
                if 'dskd_loss' in metrics:
                    log_msg += f", dskd={metrics['dskd_loss']:.4f}"
                # S41: Log E2E BEV loss
                if 'e2e_bev_loss' in metrics:
                    log_msg += f", e2e_bev={metrics['e2e_bev_loss']:.4f}"
                # V2: Log aux BEV loss and fixed weight
                if 'aux_bev_loss' in metrics:
                    log_msg += f", ssc={metrics['ssc_loss']:.4f}, bev={metrics['aux_bev_loss']:.4f}"
                    log_msg += f", bev_w={self.aux_bev_weight:.2f}"
                # S14+: Gaussian diffusion components
                if 'mse_loss' in metrics:
                    log_msg += f", mse={metrics['mse_loss']:.4f}, reg={metrics.get('reg_loss', 0):.4f}"
                    if 'lovasz_loss' in metrics and metrics['lovasz_loss'] > 0:
                        log_msg += f", lov={metrics['lovasz_loss']:.4f}"
                # S13: Factored diffusion components
                if 'occ_loss' in metrics:
                    log_msg += f", occ={metrics['occ_loss']:.4f}, sem={metrics['sem_loss']:.4f}"

                log_msg += f", lr={metrics['lr']:.2e}"
                # Curriculum: log current gt_bev_prob
                if self.config.get('curriculum', False):
                    cur_prob = self._get_curriculum_gt_bev_prob()
                    log_msg += f", gt_bev_prob={cur_prob:.3f}"
                self.logger.info(log_msg)

            # Validate (loss-based, frequent) — use TRAINING weights (not EMA)
            # EMA with decay=0.9999 takes ~20K steps to converge; at step 1000
            # it's 90% random init, producing misleading val_loss (especially
            # with multi-channel conditioning like 20ch LSK3DNet).
            if self.global_step % self.config.get('val_interval', 1000) == 0:
                val_metrics = self.validate(full_sampling=False)
                self.logger.info(
                    f"Step {self.global_step}: val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_acc={val_metrics['val_accuracy']:.4f}"
                )

                # Save best model based on loss
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint('best.pt')
                    self.logger.info(f"New best model saved! val_loss={self.best_val_loss:.4f}")

            # Full mIoU evaluation (expensive, less frequent)
            # Run BOTH training weights and EMA weights to track both.
            miou_interval = self.config.get('miou_interval', 5000)
            miou_first = self.config.get('miou_first', 0)  # Run early mIoU at this step
            run_miou = (self.global_step % miou_interval == 0) or \
                       (miou_first > 0 and self.global_step == miou_first)
            if run_miou:
                class_names = {1:'car',2:'bicycle',3:'motorcycle',4:'truck',5:'other-veh',
                    6:'person',7:'bicyclist',8:'motorcyclist',9:'road',10:'parking',
                    11:'sidewalk',12:'other-gnd',13:'building',14:'fence',
                    15:'vegetation',16:'trunk',17:'terrain',18:'pole',19:'traffic-sign'}

                # --- 1) mIoU with TRAINING weights (reliable, especially early) ---
                self.logger.info(f"Running mIoU evaluation at step {self.global_step} (training weights)...")
                train_metrics = self.validate(full_sampling=True)
                spsr_str = ""
                if 'mIoU_spsr' in train_metrics:
                    spsr_str = f", SPSR={train_metrics['mIoU_spsr']*100:.2f}%"
                scpnet_str = ""
                if 'scpnet_mIoU' in train_metrics:
                    delta = (train_metrics['mIoU'] - train_metrics['scpnet_mIoU']) * 100
                    scpnet_str = f" (SCPNet={train_metrics['scpnet_mIoU']*100:.2f}%, Δ={delta:+.2f}%)"
                self.logger.info(
                    f"Step {self.global_step} [mIoU-train]: "
                    f"mIoU={train_metrics['mIoU']*100:.2f}%{spsr_str}{scpnet_str}, "
                    f"IoU_Cmpl={train_metrics['iou_completion']*100:.2f}%, "
                    f"Prec={train_metrics['precision']*100:.2f}%, "
                    f"Recall={train_metrics['recall']*100:.2f}%"
                )
                class_iou = train_metrics.get('class_iou', {})
                if class_iou:
                    parts = [f"{class_names.get(c,'?')}={v*100:.1f}" for c,v in sorted(class_iou.items())]
                    self.logger.info(f"  Per-class: {' '.join(parts)}")

                # Save best mIoU based on training weights (more reliable)
                if train_metrics['mIoU'] > self.best_miou:
                    self.best_miou = train_metrics['mIoU']
                    self.save_checkpoint('best_miou.pt')
                    self.logger.info(f"New best mIoU model saved! mIoU={self.best_miou*100:.2f}%")

                # --- 2) mIoU with EMA weights (for comparison / generation quality) ---
                self.logger.info(f"Running mIoU evaluation at step {self.global_step} (EMA weights)...")
                self.ema.apply_shadow()
                ema_metrics = self.validate(full_sampling=True)
                ema_spsr_str = ""
                if 'mIoU_spsr' in ema_metrics:
                    ema_spsr_str = f", SPSR={ema_metrics['mIoU_spsr']*100:.2f}%"
                ema_scpnet_str = ""
                if 'scpnet_mIoU' in ema_metrics:
                    ema_delta = (ema_metrics['mIoU'] - ema_metrics['scpnet_mIoU']) * 100
                    ema_scpnet_str = f" (SCPNet={ema_metrics['scpnet_mIoU']*100:.2f}%, Δ={ema_delta:+.2f}%)"
                self.logger.info(
                    f"Step {self.global_step} [mIoU-EMA]: "
                    f"mIoU={ema_metrics['mIoU']*100:.2f}%{ema_spsr_str}{ema_scpnet_str}, "
                    f"IoU_Cmpl={ema_metrics['iou_completion']*100:.2f}%, "
                    f"Prec={ema_metrics['precision']*100:.2f}%, "
                    f"Recall={ema_metrics['recall']*100:.2f}%"
                )
                ema_class_iou = ema_metrics.get('class_iou', {})
                if ema_class_iou:
                    parts = [f"{class_names.get(c,'?')}={v*100:.1f}" for c,v in sorted(ema_class_iou.items())]
                    self.logger.info(f"  Per-class: {' '.join(parts)}")
                self.ema.restore()

            # Save checkpoint (every 20000 steps to reduce storage usage)
            if self.global_step % self.config.get('save_interval', 20000) == 0:
                self.save_checkpoint(f'step_{self.global_step}.pt')

        pbar.close()
        self.save_checkpoint('final.pt')
        self.logger.info("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save checkpoint.

        Always saves training weights as model_state_dict, even if EMA is
        currently applied (detects via non-empty ema.backup and patches the
        state dict so optimizer state stays aligned with saved weights).
        """
        path = self.output_dir / filename
        model_sd = self.model.state_dict()
        if self.ema.backup:
            # EMA is applied: model has EMA weights, backup has training weights.
            # Patch model_state_dict to contain training weights so optimizer
            # state (which tracks training weight gradients) stays consistent.
            for name, val in self.ema.backup.items():
                if name in model_sd:
                    model_sd[name] = val.clone()
        ckpt_dict = {
            'global_step': self.global_step,
            'model_state_dict': model_sd,
            'ema_shadow': {k: v.clone() for k, v in self.ema.shadow.items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_miou': self.best_miou,
            'lifter_state_dict': self.lifter.state_dict() if self.lifter is not None else None,
            'bev_model_state_dict': self.bev_model.state_dict() if self.bev_model is not None and self.use_e2e_bev else None,
            'config': self.config,
        }
        torch.save(ckpt_dict, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str, weights_only: bool = False):
        """Load checkpoint.

        Args:
            path: Path to checkpoint file.
            weights_only: If True, only load model weights and EMA shadow
                         (for fine-tuning with fresh optimizer/scheduler/step).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # When using LSK3DNet (20ch) but loading from binary LiDAR (1ch) checkpoint,
        # the lidar_encoder stem weights won't match. strict=False only handles
        # missing/unexpected keys, NOT shape mismatches. We must manually filter
        # out shape-mismatched keys before loading.
        use_lsk3d = self.config.get('use_lsk3d', False)
        model_type = self.config.get('model_type', 'full')
        is_v2 = model_type in ('v2_full', 'v2_lite', 'v3_full', 'v3_ablation', 'v3_coarse2fine', 'v3_c2f_ablation')
        has_cond_3d = self.config.get('cond_3d_channels', 0) > 0
        has_new_params = self.config.get('distance_gate', False) or self.config.get('lidar_film', False)
        if (use_lsk3d or is_v2 or has_cond_3d or has_new_params) and weights_only:
            # Filter out keys with shape mismatches
            ckpt_state = checkpoint['model_state_dict']
            current_state = self.model.state_dict()
            filtered_state = {}
            skipped_keys = []
            for k, v in ckpt_state.items():
                if k in current_state and current_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    skipped_keys.append(k)
            missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
            if skipped_keys:
                self.logger.info(f"Partial load: {len(skipped_keys)} keys skipped (shape mismatch): "
                                 f"{skipped_keys[:5]}...")
            if missing:
                self.logger.info(f"Partial load: {len(missing)} missing keys (randomly initialized): "
                                 f"{[k for k in missing[:5]]}...")
            if unexpected:
                self.logger.info(f"Partial load: {len(unexpected)} unexpected keys (skipped): "
                                 f"{[k for k in unexpected[:5]]}...")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'ema_shadow' in checkpoint:
            if (use_lsk3d or is_v2 or has_cond_3d or has_new_params) and weights_only:
                # EMA shadow may also have shape mismatches
                current_shadow = {k: v.clone() for k, v in self.ema.shadow.items()}
                for k, v in checkpoint['ema_shadow'].items():
                    if k in current_shadow and current_shadow[k].shape == v.shape:
                        current_shadow[k] = v
                self.ema.shadow = current_shadow
            else:
                self.ema.shadow = checkpoint['ema_shadow']

        if weights_only:
            self.logger.info(f"Loaded model weights from {path} (fine-tuning mode, fresh optimizer/scheduler)")
        else:
            ckpt_opt = checkpoint['optimizer_state_dict']
            if len(ckpt_opt['param_groups']) != len(self.optimizer.param_groups):
                self.logger.warning(
                    f"Optimizer param group mismatch: checkpoint has {len(ckpt_opt['param_groups'])}, "
                    f"current has {len(self.optimizer.param_groups)}. Using fresh optimizer.")
            else:
                self.optimizer.load_state_dict(ckpt_opt)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_miou = checkpoint.get('best_miou', 0.0)
            if self.lifter is not None and checkpoint.get('lifter_state_dict') is not None:
                self.lifter.load_state_dict(checkpoint['lifter_state_dict'])
            if self.bev_model is not None and self.use_e2e_bev and checkpoint.get('bev_model_state_dict') is not None:
                self.bev_model.load_state_dict(checkpoint['bev_model_state_dict'])
                self.logger.info("Loaded E2E BEV model weights from checkpoint")
            self.logger.info(f"Loaded checkpoint from step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description='Train 3D Scene Completion on SemanticKITTI (256×256×32)')

    # Data
    parser.add_argument('--data_root', type=str, default='datasets/SemanticKITTI')
    parser.add_argument('--output_dir', type=str, default='outputs/scene_completion')
    parser.add_argument('--waffleiron_root', type=str, default=None,
                        help='Path to precomputed WaffleIron BEV features (e.g., datasets/SemanticKITTI_3D/256_waffleiron)')

    # Model
    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'lite', 'sparse_full', 'sparse_lite', 'mimo_full', 'mimo_lite',
                                 'v2_full', 'v2_lite', 'v3_full', 'v3_ablation',
                                 'v3_coarse2fine', 'v3_c2f_ablation',
                                 'v4_continuous', 'v4_factored', 'v5_ve'],
                        help='exp_1: full/lite (dense Conv3d), exp_2: sparse_full/sparse_lite (spconv), '
                             'S4: mimo_full/mimo_lite (PaSCo-style MIMO with N decoder heads), '
                             'V2: v2_full/v2_lite (FiLM conditioning + multi-scale aux BEV, no cascade), '
                             'V3: v3_full (internal BEV + sparse FiLM), v3_ablation (internal BEV only), '
                             'V3.1: v3_coarse2fine (CoarseToFine + FiLM + CFG), v3_c2f_ablation (no FiLM), '
                             'V4: v4_continuous (21ch Gaussian diffusion), v4_factored (22ch factored discrete), '
                             'V5: v5_ve (20ch VE diffusion with soft probs, no amplification)')
    parser.add_argument('--num_classes', type=int, default=20)

    # Diffusion
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'kl', 'ce_direct'])
    # Ablation: restrict/skew the set of training timesteps (keeps num_timesteps=100 for inference)
    parser.add_argument('--train_timesteps_mode', type=str, default='uniform',
                        choices=['uniform', 'subset', 'skewed'],
                        help='uniform: sample t uniformly from [0, num_timesteps). '
                             'subset: sample only from --train_timesteps_list (for T=1/T=10/T=50 ablations). '
                             'skewed: sample all timesteps but bias toward specific values.')
    parser.add_argument('--train_timesteps_list', type=str, default='',
                        help='Comma-separated list of timesteps to train on (subset mode). '
                             'Example: "99" for T=1, "0,11,22,33,44,55,66,77,88,99" for T=10.')
    parser.add_argument('--train_timesteps_skew_idx', type=int, default=99,
                        help='Which timestep to upweight in skewed mode.')
    parser.add_argument('--train_timesteps_skew_weight', type=float, default=3.0,
                        help='Weight for the skewed timestep (relative to others weight=1).')

    # Diffusion V2 parameters (enhanced loss)
    parser.add_argument('--diffusion_version', type=str, default='v1',
                        choices=['v1', 'v2', 'gaussian', 'gaussian_ve', 'gaussian_logit', 'gaussian_vp', 'factored'],
                        help='v1: standard CE loss, v2: enhanced loss (focal+lovasz+class-balanced), '
                             'gaussian: VP Gaussian (broken - 158x amplification), '
                             'gaussian_ve: VE Gaussian (soft probs - simplex violation), '
                             'gaussian_logit: VE Gaussian in LOGIT space (correct), '
                             'factored: factored discrete (K=2 occ + K=20 sem)')
    parser.add_argument('--beta_max', type=float, default=0.1,
                        help='Max beta for noise schedule (0.1 for stronger noise, 0.02 for weak)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (higher = more focus on hard examples)')
    parser.add_argument('--class_0_weight', type=float, default=0.02,
                        help='Weight for class 0 (empty) - very low to focus on semantics')
    parser.add_argument('--occupied_weight', type=float, default=10.0,
                        help='Weight multiplier for occupied voxels (non-class-0)')
    parser.add_argument('--lovasz_weight', type=float, default=0.3,
                        help='Weight for Lovasz loss component (IoU optimization)')
    parser.add_argument('--obs_weight_factor', type=float, default=2.0,
                        help='Extra weight for LiDAR-observed voxels (0 to disable)')
    parser.add_argument('--auxiliary_loss_weight', type=float, default=0.0,
                        help='Weight for auxiliary x0 reconstruction loss (paper: 0.05)')

    # Training (adjusted for SemanticKITTI 256×256×32)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (4 recommended for 256×256×32 on 80GB GPU)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    # Logging
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1000,
                        help='Validation interval (loss-based, fast)')
    parser.add_argument('--miou_interval', type=int, default=5000,
                        help='Full mIoU evaluation interval (slow, requires sampling)')
    parser.add_argument('--miou_first', type=int, default=0,
                        help='Run first mIoU eval at this step (0=disabled, e.g. 1000 for early check)')
    parser.add_argument('--miou_samples', type=int, default=100,
                        help='Number of samples for mIoU evaluation (subset of val set)')
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--num_workers', type=int, default=4)

    # Resume / Fine-tune
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (loads optimizer/scheduler/step)')
    parser.add_argument('--finetune', type=str, default=None,
                        help='Path to checkpoint for fine-tuning (loads model weights only, fresh optimizer)')

    # Quantized data (faster loading)
    parser.add_argument('--use_quantized', action='store_true',
                        help='Use quantized numpy data from SemanticKITTI_3D/256/')

    # S1: Classifier-Free Guidance (CFG)
    parser.add_argument('--use_cfg', action='store_true',
                        help='S1: Use Classifier-Free Guidance for lifted condition')
    parser.add_argument('--cfg_drop_prob', type=float, default=0.0,
                        help='CFG conditioning dropout probability during training (default: 0.0, set to 0.1 for CFG)')
    parser.add_argument('--cfg_guidance_scale', type=float, default=1.5,
                        help='CFG guidance scale for inference (default: 1.5)')

    # S1/S2: Lifted features dimension
    parser.add_argument('--lifted_feature_dim', type=int, default=64,
                        help='Dimension of lifted features from LiftingModule (default: 64)')

    # S2: 2D→3D Lifting
    parser.add_argument('--use_lifting', action='store_true',
                        help='S2: Use 2D→3D lifting as additional conditioning')
    parser.add_argument('--use_sdedit', action='store_true',
                        help='S2: Use SDEdit-style sampling starting from lifted 3D')
    parser.add_argument('--sdedit_start_step', type=int, default=50,
                        help='SDEdit start timestep (default: 50 out of 100)')
    parser.add_argument('--p_init_train', type=float, default=0.0,
                        help='[DEPRECATED] SDEdit training probability. Use --cond_3d_channels instead.')
    parser.add_argument('--cond_3d_channels', type=int, default=0,
                        help='Number of 3D conditioning channels (e.g. 20 for LSK3DNet probs). '
                             'Concatenated to model input alongside noisy x_t.')

    # S3: DSKD for Scene Completion (Enhanced: GT BEV + Multi-frame Teacher → Pred BEV + Single-frame Student)
    parser.add_argument('--use_dskd', action='store_true',
                        help='S3: Use DSKD to distill GT BEV knowledge to Pred BEV model')
    parser.add_argument('--dskd_weight', type=float, default=0.5,
                        help='Weight for DSKD loss (default: 0.5 for KL divergence)')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher model checkpoint (trained with GT BEV + multi-frame)')
    parser.add_argument('--s3_mode', type=str, default=None, choices=['teacher', 'intermediate', 'student'],
                        help='S3 DSKD mode: teacher (GT BEV + multi-frame), intermediate (GT BEV + single-frame), '
                             'or student (Pred BEV + single-frame). Use curriculum: teacher→intermediate→student')
    parser.add_argument('--pred_bev_dir', type=str, default=None,
                        help='Directory containing predicted BEV maps (required for student mode)')
    parser.add_argument('--dskd_temperature', type=float, default=4.0,
                        help='Temperature for soft distillation (default: 4.0)')
    parser.add_argument('--kd_type', type=str, default='output', choices=['dskd', 'output'],
                        help='KD type: dskd (pairwise features - SCPNet) or output (soft labels - Hinton). '
                             'Default: output (more robust for multi-frame→single-frame)')
    parser.add_argument('--gt_bev_prob', type=float, default=0.0,
                        help='BEV mixing: probability of using GT BEV in student mode (default: 0.0). '
                             'Recommended: 0.2-0.3 for curriculum distillation')

    # S40v2/S41: Soft BEV and End-to-End BEV training
    parser.add_argument('--soft_bev', action='store_true',
                        help='S40v2: Use soft BEV (softmax) from frozen BEV model instead of hard argmax')
    parser.add_argument('--use_e2e_bev', action='store_true',
                        help='S41: End-to-end BEV+SSC training with gradient flow through BEV model')
    parser.add_argument('--bev_checkpoint', type=str, default=None,
                        help='Path to BEV model checkpoint (required for --soft_bev or --use_e2e_bev)')
    parser.add_argument('--bev_lr_factor', type=float, default=0.1,
                        help='S41: LR multiplier for BEV model (default: 0.1, prevents forgetting)')
    parser.add_argument('--e2e_bev_weight', type=float, default=0.5,
                        help='S41: Weight for BEV supervised loss in E2E training (default: 0.5)')

    # Curriculum learning: gradually transition from GT BEV to Pred BEV
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning: linearly decrease gt_bev_prob from '
                             'curriculum_start_prob to curriculum_end_prob over first curriculum_warmup_frac of training')
    parser.add_argument('--curriculum_start_prob', type=float, default=1.0,
                        help='Starting GT BEV probability (default: 1.0 = 100% GT BEV)')
    parser.add_argument('--curriculum_end_prob', type=float, default=0.0,
                        help='Ending GT BEV probability (default: 0.0 = 100% Pred BEV)')
    parser.add_argument('--curriculum_warmup_frac', type=float, default=0.6,
                        help='Fraction of total training for curriculum transition (default: 0.6)')

    # S5/S6: LSK3DNet 3D features
    parser.add_argument('--use_lsk3d', action='store_true',
                        help='S5/S6: Replace binary LiDAR with LSK3DNet 3D soft logits (20ch)')
    parser.add_argument('--lsk3d_dir', type=str, default=None,
                        help='LSK3DNet 3D features directory (default: auto-detect from data_root)')
    parser.add_argument('--geom_dir', type=str, default=None,
                        help='S43: Geometric features directory (height/normals/intensity/TSDF)')
    parser.add_argument('--tsdf_bev', action='store_true',
                        help='S46: Route TSDF via BEV projection (4ch column stats) instead of sparse encoder')
    parser.add_argument('--ssc_lovasz_weight', type=float, default=0.0,
                        help='S43: Weight for 3D Lovász loss on x_0 predictions (default: 0, set to 0.1)')
    parser.add_argument('--scpnet_pred_dir', '--base_pred_dir', dest='scpnet_pred_dir',
                        type=str, default=None,
                        help='Base-model 3D predictions directory for refinement conditioning. '
                             '`--scpnet_pred_dir` is the historical name; `--base_pred_dir` is '
                             'the v1.1.0+ alias for cross-base support (SCPNet / JS3C-Net / ...).')
    parser.add_argument('--base_kind', type=str, default='scpnet',
                        choices=['scpnet', 'js3c'],
                        help='Base model kind for cross-base evaluation. Pure label — does not '
                             'affect inference logic; the model treats predictions identically. '
                             'Default scpnet maintains backwards compatibility with v1.0.0 CLIs.')
    parser.add_argument('--talos_pred_dir', type=str, default=None,
                        help='TALoS TTA predictions dir (overrides SCPNet for real seqs, falls back for synthetic)')
    parser.add_argument('--bev_cold_dir', type=str, default=None,
                        help='S2D2 refined BEV predictions dir (replaces SCPNet-derived BEV)')
    parser.add_argument('--no_bev', action='store_true',
                        help='S6: Disable BEV conditioning entirely (use only LSK3DNet 3D)')
    parser.add_argument('--bev_from_base', action='store_true',
                        help='Derive BEV from base-model 3D prediction (height-pool) instead of GT/pred BEV. '
                             'The base is whichever `--base_kind` (and `--base_pred_dir`) is wired in '
                             '(SCPNet, JS3C-Net, ...).')
    parser.add_argument('--bev_from_scpnet', action='store_true',
                        help='DEPRECATED v1.1.1 — alias for --bev_from_base, kept for back-compat. '
                             'Removed in v2.0.0.')
    parser.add_argument('--sdedit_from_scpnet', action='store_true',
                        help='SDEdit eval: start sampling from SCPNet prediction + noise')
    parser.add_argument('--sdedit_scpnet_t_start', type=int, default=50,
                        help='SDEdit start timestep when using SCPNet init (default 50/100)')
    parser.add_argument('--ssc_multiscale', action='store_true',
                        help='B1c: Add multi-scale SCPNet 3D conditioning (summed with bev_emb)')
    parser.add_argument('--obs_mask_channel', action='store_true',
                        help='B2: Add observation mask as extra model input channel')
    parser.add_argument('--use_repaint', action='store_true',
                        help='B2: Use repaint sampling during eval (anchor observed voxels)')
    parser.add_argument('--completion_weight', type=float, default=0.0,
                        help='B2: Extra weight for unobserved voxels in loss (e.g., 2.0)')
    parser.add_argument('--cold_diffusion', action='store_true',
                        help='B4: Cold Diffusion — use SCPNet pred as noise target instead of uniform')
    parser.add_argument('--force_single_frame_lidar', action='store_true',
                        help='Force single-frame LiDAR voxels in teacher mode (overrides the '
                             'multi-frame default inherited from the DSKD teacher code path). '
                             'Use this to train cold-diffusion models with single-frame input '
                             'matching the deployment-time lidar stream.')
    parser.add_argument('--algo2_eval_steps', type=int, default=100,
                        help='B5: Number of S2D2 correction-sampling steps during eval (default: 100)')
    parser.add_argument('--train_sequences', type=str, default=None,
                        help='B6: Comma-separated train sequences (e.g., 00,01,...,10,synthetic)')
    parser.add_argument('--hp_bev_aux', action='store_true',
                        help='S44: Height-pool BEV aux loss from x_0 predictions')

    # S18: LSK3DNet 3D predictions for SDEdit init (replaces BEV-lifted)
    parser.add_argument('--lsk3d_3d_root', type=str, default=None,
                        help='S18: Path to LSK3DNet sparse 3D predictions (npz) for SDEdit init')

    # V2: Auxiliary BEV head (for v2_full/v2_lite model types)
    parser.add_argument('--aux_bev', action='store_true', default=True,
                        help='V2: Enable multi-scale auxiliary BEV head (default: True)')
    parser.add_argument('--no_aux_bev', action='store_true',
                        help='V2: Disable auxiliary BEV head (S8 ablation)')
    parser.add_argument('--aux_bev_class_0_weight', type=float, default=0.02,
                        help='V2: Class 0 weight for aux BEV focal loss (default: 0.02, matches SSC diffusion)')
    parser.add_argument('--aux_bev_lovasz_weight', type=float, default=0.3,
                        help='V2: Lovász weight for aux BEV loss (default: 0.3)')
    parser.add_argument('--aux_bev_weight', type=float, default=0.1,
                        help='V2/V3: Fixed weight for aux BEV loss (default: 0.1)')

    # V3: Internal BEV completion (for v3_full/v3_ablation model types)
    parser.add_argument('--dense_bev_channels', type=int, default=32,
                        help='V3: Number of channels for internal BEV completion (default: 32)')
    parser.add_argument('--bev_completion_layers', type=int, default=5,
                        help='V3: Number of 2D conv layers in BEV completion network (default: 5)')
    parser.add_argument('--no_sparse_film', action='store_true',
                        help='V3: Disable sparse FiLM conditioning (v3_ablation uses dense only)')

    # V4: Factored representation (S13/S14)
    parser.add_argument('--fuse_time_cond', action='store_true',
                        help='V4/S13/S14: DiffSSC-style concat(condition, timestep) → multiplicative gate')

    # V4 Continuous Gaussian diffusion parameters (S14)
    parser.add_argument('--logit_scale', type=float, default=5.0,
                        help='S14: Scale for logit encoding (sigmoid(5)=0.993)')
    parser.add_argument('--sigma_occ', type=float, default=1.0,
                        help='S14: Anisotropic noise scale for occupancy (DiffSSC Paper Eq. 3)')
    parser.add_argument('--sigma_sem', type=float, default=1.0,
                        help='S14: Anisotropic noise scale for semantics (1.0=isotropic, 0.2=paper)')
    parser.add_argument('--lambda_p', type=float, default=5.0,
                        help='S14: Spatial (occupancy) regularization weight (DiffSSC Paper Eq. 5)')
    parser.add_argument('--lambda_s', type=float, default=4.0,
                        help='S14: Semantic regularization weight (DiffSSC Paper Eq. 5)')
    parser.add_argument('--use_skewness_reg', action='store_true',
                        help='S14: Add skewness term to semantic regularization (DiffSSC Paper Eq. 5)')
    parser.add_argument('--anisotropic', action='store_true', default=False,
                        help='S14: Use anisotropic noise matrix W (DiffSSC Paper Eq. 3, NOT in code)')
    parser.add_argument('--no_anisotropic', action='store_true',
                        help='S14: Disable anisotropic noise (use isotropic, matches DiffSSC code)')
    parser.add_argument('--beta_schedule', type=str, default='linear_diffssc',
                        choices=['linear_diffssc', 'cosine_diffssc', 'cosine_nd', 'linear'],
                        help='S14: Beta schedule: linear_diffssc (DiffSSC code, proven), cosine_diffssc (Paper Eq. 6), cosine_nd, linear')
    parser.add_argument('--dpm_solver_steps', type=int, default=50,
                        help='S14: Inference steps for SDE-DPM-Solver++ (DiffSSC uses 50)')

    # V4 Factored discrete diffusion parameters (S13)
    parser.add_argument('--beta_max_occ', type=float, default=0.15,
                        help='S13: Max beta for occupancy diffusion (aggressive)')
    parser.add_argument('--beta_max_sem', type=float, default=0.05,
                        help='S13: Max beta for semantic diffusion (gentle)')

    # S4: MIMO (Multi-Input Multi-Output)
    parser.add_argument('--use_mimo', action='store_true',
                        help='S4: Use MIMO with multiple augmented BEV inputs')
    parser.add_argument('--mimo_num_subnets', type=int, default=3,
                        help='Number of MIMO subnets (default: 3, PaSCo paper default)')
    parser.add_argument('--mimo_aug_types', type=str, default='identity,noise,dropout',
                        help='Comma-separated augmentation types (identity,noise,dropout,smooth,rot90,flip_h)')
    parser.add_argument('--mimo_ensemble', type=str, default='mean_probs',
                        choices=['mean_logits', 'mean_probs', 'vote'],
                        help='MIMO ensemble method (default: mean_probs, PaSCo default)')
    parser.add_argument('--mimo_scale_range', type=float, default=0.0,
                        help='MIMO scale augmentation range (default: 0.0 = disabled, PaSCo default)')
    parser.add_argument('--mimo_max_angle', type=float, default=30.0,
                        help='MIMO max rotation angle in degrees (default: 30.0, PaSCo paper)')
    parser.add_argument('--mimo_use_training_mode', action='store_true',
                        help='Use PaSCo-style dataset-level MIMO for training (different samples per subnet)')

    # S5: PaSCo's SPCDense3Dv2 at UNet bottleneck
    parser.add_argument('--use_dense_3d', action='store_true',
                        help='S5: Enable PaSCo SPCDense3Dv2 at bottleneck for dense hallucination')
    parser.add_argument('--dense_3d_dropout', type=float, default=0.1,
                        help='Dropout for SPCDense3Dv2 bottleneck (default: 0.1)')

    # S25-S27: Sparse conditioning densification
    parser.add_argument('--densify_nn', action='store_true',
                        help='S25: NN densification — fill zero voxels with nearest-neighbor features')
    parser.add_argument('--distance_gate', action='store_true',
                        help='S26: Add learned distance-dependent gate to densified features')
    parser.add_argument('--lidar_film', action='store_true',
                        help='S27: Replace additive LiDAR conditioning with multiplicative FiLM')

    # S48: Direct prediction (no diffusion)
    parser.add_argument('--direct_prediction', action='store_true',
                        help='S48: Bypass diffusion entirely. Single forward pass with uniform init, '
                             'CE+Lovász loss. 50x faster inference.')
    parser.add_argument('--dp_lifted_init', action='store_true',
                        help='S48+: Use lifted BEV (S3CNet height priors) as init instead of uniform 1/K')
    parser.add_argument('--dp_spsr', action='store_true',
                        help='S48+: Apply SPSR post-processing at eval time (3×3×3 neighborhood voting)')

    # S17: Logit-space VE Gaussian diffusion args
    parser.add_argument('--sigma_min', type=float, default=0.01,
                        help='VE diffusion: minimum sigma (default: 0.01)')
    parser.add_argument('--sigma_max', type=float, default=2.0,
                        help='VE diffusion: maximum sigma (default: 2.0)')
    parser.add_argument('--sigma_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'exponential'],
                        help='VE diffusion: sigma schedule type (default: cosine)')

    parser.add_argument('--seed', type=int, default=42,
                        help=('Random seed. Default: 42, the verified reproducible '
                              'recipe — converges to 38.05%% val 1-step mIoU on the '
                              'migrated codebase, within ~0.5%% of the paper headline '
                              '38.54%%. Pass --seed=None to disable seeding (not '
                              'recommended; produces a different trajectory).'))

    args = parser.parse_args()

    # Seed only when explicitly requested. The published gssc_31k_mf headline
    # checkpoint was trained without seeding, so leaving args.seed=None is what
    # reproduces the paper trajectory in distribution.
    if args.seed is not None:
        import os
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Config for SemanticKITTI (256×256×32)
    config = {
        'data_root': args.data_root,
        'output_dir': args.output_dir,
        'waffleiron_root': args.waffleiron_root,
        'model_type': args.model_type,
        'num_classes': args.num_classes,
        'num_timesteps': args.num_timesteps,
        'loss_type': args.loss_type,
        # Timestep sampling ablation (for T=1/T=10/T=50/skewed experiments)
        'train_timesteps_mode': args.train_timesteps_mode,
        'train_timesteps_list': args.train_timesteps_list,
        'train_timesteps_skew_idx': args.train_timesteps_skew_idx,
        'train_timesteps_skew_weight': args.train_timesteps_skew_weight,
        # V2 diffusion parameters
        'diffusion_version': args.diffusion_version,
        'beta_max': args.beta_max,
        'focal_gamma': args.focal_gamma,
        'class_0_weight': args.class_0_weight,
        'occupied_weight': args.occupied_weight,
        'lovasz_weight': args.lovasz_weight,
        'obs_weight_factor': args.obs_weight_factor,
        'auxiliary_loss_weight': args.auxiliary_loss_weight,
        # Training
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'num_iterations': args.num_iterations,
        'ema_decay': args.ema_decay,
        'log_interval': args.log_interval,
        'val_interval': args.val_interval,
        'miou_interval': args.miou_interval,
        'miou_first': args.miou_first,
        'miou_samples': args.miou_samples,
        'save_interval': args.save_interval,
        'num_workers': args.num_workers,
        'train_sequences': args.train_sequences.split(',') if args.train_sequences else ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
        'val_sequences': ['08'],
        'use_quantized': args.use_quantized,

        # S1: Classifier-Free Guidance
        'use_cfg': args.use_cfg,
        'cfg_drop_prob': args.cfg_drop_prob,
        'cfg_guidance_scale': args.cfg_guidance_scale,
        'lifted_feature_dim': args.lifted_feature_dim,

        # S2: 2D→3D Lifting
        'use_lifting': args.use_lifting,
        'use_sdedit': args.use_sdedit,
        'sdedit_start_step': args.sdedit_start_step,
        'p_init_train': args.p_init_train,
        'cond_3d_channels': args.cond_3d_channels,

        # S3: DSKD for Scene Completion (Enhanced Strategy)
        'use_dskd': args.use_dskd,
        'dskd_weight': args.dskd_weight,
        'teacher_checkpoint': args.teacher_checkpoint,
        's3_mode': args.s3_mode,
        'pred_bev_dir': args.pred_bev_dir,
        'dskd_temperature': args.dskd_temperature,
        'gt_bev_prob': args.gt_bev_prob,
        'kd_type': args.kd_type,

        # S40v2/S41: Soft BEV and E2E BEV training
        'soft_bev': args.soft_bev,
        'use_e2e_bev': args.use_e2e_bev,
        'bev_checkpoint': args.bev_checkpoint,
        'bev_lr_factor': args.bev_lr_factor,
        'e2e_bev_weight': args.e2e_bev_weight,

        # Curriculum learning
        'curriculum': args.curriculum,
        'curriculum_start_prob': args.curriculum_start_prob,
        'curriculum_end_prob': args.curriculum_end_prob,
        'curriculum_warmup_frac': args.curriculum_warmup_frac,

        # S5/S6: LSK3DNet 3D features
        'use_lsk3d': args.use_lsk3d,
        'lsk3d_dir': args.lsk3d_dir,
        'geom_dir': args.geom_dir,
        'tsdf_bev': args.tsdf_bev,
        'ssc_lovasz_weight': args.ssc_lovasz_weight,
        'scpnet_pred_dir': args.scpnet_pred_dir,
        'talos_pred_dir': args.talos_pred_dir,
        'bev_cold_dir': args.bev_cold_dir,
        'no_bev': args.no_bev,
        'bev_from_base': resolve_bev_from_base(
            bev_from_base=args.bev_from_base,
            bev_from_scpnet=args.bev_from_scpnet,
        ),
        'sdedit_from_scpnet': args.sdedit_from_scpnet,
        'sdedit_scpnet_t_start': args.sdedit_scpnet_t_start,
        'ssc_multiscale': args.ssc_multiscale,
        'obs_mask_channel': args.obs_mask_channel,
        'use_repaint': args.use_repaint,
        'completion_weight': args.completion_weight,
        'cold_diffusion': args.cold_diffusion,
        'force_single_frame_lidar': args.force_single_frame_lidar,
        'hp_bev_aux': args.hp_bev_aux,

        # S18: LSK3DNet 3D predictions for SDEdit init
        'lsk3d_3d_root': args.lsk3d_3d_root,

        # V2: Auxiliary BEV head
        'aux_bev': args.aux_bev and not args.no_aux_bev,
        'aux_bev_class_0_weight': args.aux_bev_class_0_weight,
        'aux_bev_lovasz_weight': args.aux_bev_lovasz_weight,
        'aux_bev_weight': args.aux_bev_weight,

        # V3: Internal BEV completion / V3.1: CoarseToFine
        'dense_bev_channels': args.dense_bev_channels,
        'bev_completion_layers': args.bev_completion_layers,

        # V4: Factored representation (S13/S14)
        'fuse_time_cond': args.fuse_time_cond,

        # V4 Continuous Gaussian diffusion (S14)
        'logit_scale': args.logit_scale,
        'sigma_occ': args.sigma_occ,
        'sigma_sem': args.sigma_sem,
        'lambda_p': args.lambda_p,
        'lambda_s': args.lambda_s,
        'use_skewness_reg': args.use_skewness_reg,
        'anisotropic': args.anisotropic and not args.no_anisotropic,
        'beta_schedule': args.beta_schedule,
        'dpm_solver_steps': args.dpm_solver_steps,

        # V4 Factored discrete diffusion (S13)
        'beta_max_occ': args.beta_max_occ,
        'beta_max_sem': args.beta_max_sem,

        # S4: MIMO (PaSCo-style)
        # Reference: PaSCo CVPR 2024, Section 3.2.1-3.2.2
        # Key hyperparameters: n_subnets=2, scale_range=0, ensemble=mean_probs
        'use_mimo': args.use_mimo,
        'mimo_num_subnets': args.mimo_num_subnets,
        'mimo_aug_types': args.mimo_aug_types.split(','),
        'mimo_ensemble': args.mimo_ensemble,
        'mimo_scale_range': args.mimo_scale_range,  # PaSCo default: 0 (disabled)
        'mimo_max_angle': args.mimo_max_angle,  # PaSCo paper: 30 degrees
        'mimo_use_training_mode': args.mimo_use_training_mode,  # Dataset-level MIMO

        # S5: PaSCo's SPCDense3Dv2 at bottleneck
        'use_dense_3d': args.use_dense_3d,
        'dense_3d_dropout': args.dense_3d_dropout,

        # S17: Logit-space VE Gaussian diffusion
        'sigma_min': args.sigma_min,
        'sigma_max': args.sigma_max,
        'sigma_schedule': args.sigma_schedule,

        # S25-S27: Sparse conditioning densification
        'densify_nn': args.densify_nn,
        'distance_gate': args.distance_gate,
        'lidar_film': args.lidar_film,

        # S48: Direct prediction (no diffusion)
        'direct_prediction': args.direct_prediction,
        'dp_lifted_init': args.dp_lifted_init,
        'dp_spsr': args.dp_spsr,
    }

    # Curriculum learning: auto-configure student mode
    if args.curriculum:
        if not config.get('s3_mode'):
            config['s3_mode'] = 'student'
            print("Curriculum: auto-set s3_mode=student")
        if not config.get('pred_bev_dir'):
            # data_root is usually 'datasets/dataset_SemanticKITTI_SSC' or 'datasets/'
            dr = Path(config['data_root'])
            if 'dataset_SemanticKITTI_SSC' in str(dr):
                pred_dir = dr.parent / 'predicted_bev'
            else:
                pred_dir = dr / 'predicted_bev'
            config['pred_bev_dir'] = str(pred_dir)
            print(f"Curriculum: auto-set pred_bev_dir={config['pred_bev_dir']}")
        # Set initial gt_bev_prob to curriculum start (dataset will be updated dynamically)
        config['gt_bev_prob'] = config['curriculum_start_prob']
        print(f"Curriculum: gt_bev_prob will schedule {config['curriculum_start_prob']:.2f} → "
              f"{config['curriculum_end_prob']:.2f} over {config['curriculum_warmup_frac']*100:.0f}% of training")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create trainer
    trainer = SceneCompletionTrainer(config, device)

    # Resume or fine-tune if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.finetune:
        trainer.load_checkpoint(args.finetune, weights_only=True)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
