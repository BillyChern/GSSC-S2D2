"""
DiffSSC Dataset: Joint geometry + semantics point cloud dataset.

For each frame:
- Partial cloud: sparse LiDAR scan + LSK3DNet 14-TTA semantic probs → [N_part, 23]
- Full cloud: clean_map cropped to ego frame + semantic logits → [N_full, 23]

Both have 23 channels: 3 xyz + 20 semantic (logits for full, probs for partial).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .diffssc_utils import semantic_encode


# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
VELODYNE_ROOT = PROJECT_ROOT / "datasets" / "kitti_odometry" / "dataset" / "sequences"
SSC_ROOT = PROJECT_ROOT / "datasets" / "dataset_SemanticKITTI_SSC" / "sequences"
GT_ROOT = PROJECT_ROOT / "datasets" / "diffssc_gt"
LSK3D_ROOT = PROJECT_ROOT / "datasets" / "SemanticKITTI_3D" / "256_lsk3d_3d_tta"
SSC_GT_ROOT = PROJECT_ROOT / "datasets" / "SemanticKITTI_3D" / "256"

# SemanticKITTI label filtering (static objects only)
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}
LEARNING_MAP_ARRAY = np.zeros(300, dtype=np.uint8)
for _orig, _mapped in LEARNING_MAP.items():
    if _orig < 300:
        LEARNING_MAP_ARRAY[_orig] = _mapped

# Point cloud range for SSC evaluation grid
POINT_CLOUD_RANGE = (0, -25.6, -2, 51.2, 25.6, 4.4)


def parse_calibration(calib_path):
    """Parse KITTI calibration file, return Tr (velodyne-to-camera) as 4x4."""
    calib = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, content = line.strip().split(":", 1)
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key.strip()] = pose
    return calib


def load_poses(calib_path, poses_path):
    """Load poses, converting from camera frame to velodyne frame using Tr."""
    calibration = parse_calibration(calib_path)
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    poses = []
    with open(poses_path) as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(Tr_inv @ pose @ Tr)
    return poses


class DiffSSCDataset(Dataset):
    """Point cloud dataset for DiffSSC training/evaluation.

    Args:
        seqs: list of sequence numbers (ints)
        split: 'train' or 'val'
        num_points_full: number of points for complete cloud
        num_points_part: number of points for partial cloud
        max_range: maximum distance from ego in meters
        resolution: voxel resolution for sparse tensors
        num_classes: number of semantic classes
        semantic_scale: scale for semantic logit encoding
        augment: whether to apply data augmentation
    """
    def __init__(
        self,
        seqs,
        split='train',
        num_points_full=180000,
        num_points_part=18000,
        max_range=50.0,
        resolution=0.05,
        num_classes=20,
        semantic_scale=5.0,
        augment=True,
        use_bev=False,
    ):
        super().__init__()
        self.split = split
        self.num_points_full = num_points_full
        self.num_points_part = num_points_part
        self.max_range = max_range
        self.resolution = resolution
        self.num_classes = num_classes
        self.use_bev = use_bev
        self.semantic_scale = semantic_scale
        self.augment = augment and (split == 'train')

        # Pre-load clean maps and poses for each sequence
        self.cache_maps = {}   # seq_str → [M, 4] float32 (x, y, z, label)
        self.cache_poses = {}  # seq_str → list of 4x4 arrays
        self.frames = []       # list of (seq_str, frame_id) tuples

        for seq in seqs:
            seq_str = f"{seq:02d}"

            # Load clean_map with semantic labels
            map_path = GT_ROOT / seq_str / "clean_map.npy"
            if not map_path.exists():
                print(f"Warning: clean_map not found for seq {seq_str}, skipping")
                continue
            self.cache_maps[seq_str] = np.load(str(map_path))

            # Load poses
            calib_path = str(VELODYNE_ROOT / seq_str / "calib.txt")
            poses_path = str(SSC_ROOT / seq_str / "poses.txt")
            self.cache_poses[seq_str] = load_poses(calib_path, poses_path)

            # Enumerate frames
            velodyne_dir = VELODYNE_ROOT / seq_str / "velodyne"
            bin_files = sorted(velodyne_dir.glob("*.bin"))
            for bf in bin_files:
                frame_id = int(bf.stem)
                if frame_id < len(self.cache_poses[seq_str]):
                    self.frames.append((seq_str, frame_id))

        print(f"DiffSSCDataset: {len(self.frames)} frames from seqs {[f'{s:02d}' for s in seqs]}")

    def __len__(self):
        return len(self.frames)

    def _load_partial_scan(self, seq_str, frame_id):
        """Load sparse LiDAR scan, filter to static points within range.

        Returns:
            p_part_xyz: [N, 3] float32, filtered partial scan points (ego frame)
        """
        bin_path = VELODYNE_ROOT / seq_str / "velodyne" / f"{frame_id:06d}.bin"
        label_path = SSC_ROOT / seq_str / "labels" / f"{frame_id:06d}.label"

        pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)[:, :3]

        if label_path.exists():
            labels_raw = np.fromfile(str(label_path), dtype=np.uint32).reshape(-1) & 0xFFFF
            # Keep static points only
            static_mask = (labels_raw > 1) & (labels_raw < 252)
            pts = pts[static_mask]

        # Distance filter
        dist = np.sqrt((pts ** 2).sum(axis=-1))
        valid = (dist > 3.5) & (dist < self.max_range)
        pts = pts[valid]

        # Height filter
        pts = pts[pts[:, 2] > -4.0]

        return pts

    def _load_lsk3d_features(self, seq_str, frame_id):
        """Load LSK3DNet 14-TTA features for this frame.

        Returns:
            coords: [N, 3] uint8 voxel coordinates
            probs: [N, 20] float16 semantic probabilities
        """
        feat_path = LSK3D_ROOT / seq_str / f"{frame_id:06d}_lsk3d_3d.npz"
        if not feat_path.exists():
            return None, None
        data = np.load(str(feat_path))
        return data['coords'], data['probs']

    def _crop_full_to_ego(self, seq_str, frame_id):
        """Crop clean_map to ego frame for this frame.

        Transforms world-frame points to ego frame using inverse pose,
        filters by max_range and height.

        Returns:
            p_full_xyz: [N, 3] float32, complete scene points (ego frame)
            p_full_labels: [N] uint8, semantic labels (1-19)
        """
        p_map = self.cache_maps[seq_str]  # [M, 4] (x, y, z, label)
        pose = self.cache_poses[seq_str][frame_id]  # 4x4

        # Ego position in world frame
        R = pose[:3, :3]
        t = pose[:3, 3]

        # FAST bounding box pre-filter (per-axis, avoids sqrt on all 38M pts)
        mr = self.max_range
        box_mask = (
            (np.abs(p_map[:, 0] - t[0]) < mr) &
            (np.abs(p_map[:, 1] - t[1]) < mr) &
            (np.abs(p_map[:, 2] - t[2]) < mr)
        )
        p_crop = p_map[box_mask]

        if len(p_crop) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.uint8)

        # Exact distance filter on the smaller set
        diff = p_crop[:, :3] - t
        dist_sq = (diff ** 2).sum(axis=-1)
        mask = dist_sq < (mr * mr)
        p_crop = p_crop[mask]
        diff = diff[mask]

        if len(p_crop) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.uint8)

        labels = p_crop[:, 3].astype(np.uint8)

        # Transform world → ego: x_ego = (x_world - t) @ R
        # (R is orthogonal, so R_inv = R.T, and for row vectors: x @ R = (R.T @ x.T).T)
        xyz_ego = diff @ R

        # Height filter
        valid = xyz_ego[:, 2] > -4.0
        xyz_ego = xyz_ego[valid]
        labels = labels[valid]

        # Remove class 0 (unlabeled) from full cloud
        valid_cls = labels > 0
        xyz_ego = xyz_ego[valid_cls]
        labels = labels[valid_cls]

        return xyz_ego.astype(np.float32), labels

    def _attach_lsk3d_to_partial(self, p_part_xyz, lsk3d_coords, lsk3d_probs):
        """Attach LSK3DNet semantic probs to partial scan points via voxel grid lookup.

        Args:
            p_part_xyz: [N_part, 3] partial scan in ego frame (meters)
            lsk3d_coords: [N_lsk, 3] uint8 voxel coordinates (in 256 grid)
            lsk3d_probs: [N_lsk, 20] float16 semantic probabilities

        Returns:
            p_part_feats: [N_part, 23] (xyz + 20 semantic probs)
        """
        uniform = 1.0 / self.num_classes
        if lsk3d_coords is None or len(lsk3d_coords) == 0:
            probs = np.full((len(p_part_xyz), self.num_classes), uniform, dtype=np.float32)
            return np.hstack([p_part_xyz, probs])

        lsk_probs_f = lsk3d_probs.astype(np.float32)

        # Build flat lookup array: voxel flat index → LSK3D index (vectorized)
        MAX_FLAT = 256 * 256 * 32  # 2M entries, 8MB as int32
        lsk_flat = (lsk3d_coords[:, 0].astype(np.int32) * (256 * 32) +
                    lsk3d_coords[:, 1].astype(np.int32) * 32 +
                    lsk3d_coords[:, 2].astype(np.int32))
        lookup_arr = np.full(MAX_FLAT, -1, dtype=np.int32)
        lookup_arr[lsk_flat] = np.arange(len(lsk_flat), dtype=np.int32)

        # Convert partial points to voxel coordinates (matching SSC grid)
        x_min, y_min, z_min = POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[2]
        voxel_size = 0.2  # SSC grid voxel size
        part_voxels = np.floor((p_part_xyz - np.array([x_min, y_min, z_min])) / voxel_size).astype(np.int32)
        part_voxels = np.clip(part_voxels, [0, 0, 0], [255, 255, 31])
        part_flat = part_voxels[:, 0] * (256 * 32) + part_voxels[:, 1] * 32 + part_voxels[:, 2]

        # Vectorized lookup (no Python loop)
        matched_idx = lookup_arr[part_flat]
        has_match = matched_idx >= 0

        N_part = len(p_part_xyz)
        probs = np.full((N_part, self.num_classes), uniform, dtype=np.float32)
        probs[has_match] = lsk_probs_f[matched_idx[has_match]]

        return np.hstack([p_part_xyz, probs])

    def _sample_points(self, pts, n_target):
        """Sample or repeat points to exactly n_target.

        Args:
            pts: [N, F] array
            n_target: target number of points

        Returns:
            sampled: [n_target, F] array
        """
        N = len(pts)
        if N == 0:
            return np.zeros((n_target, pts.shape[1] if pts.ndim > 1 else 3), dtype=np.float32)

        if N >= n_target:
            # Subsample
            idx = np.random.choice(N, n_target, replace=False)
            return pts[idx]
        else:
            # Repeat + subsample remainder
            repeats = int(np.ceil(n_target / N))
            idx = np.random.permutation(N)
            pts_shuffled = pts[idx]
            pts_repeated = np.tile(pts_shuffled, (repeats, 1))[:n_target]
            return pts_repeated

    def _augment(self, p_full, p_part):
        """Joint augmentation: random rotation + flip.

        Applied to xyz channels only (first 3), keeping semantics unchanged.

        Returns:
            p_full, p_part: augmented point clouds
            aug_params: dict with 'rot_k' (0-3), 'flip_x' (bool), 'flip_y' (bool)
                        for applying the same transform to the BEV grid
        """
        # Random rotation around Z (k * 90 degrees)
        rot_k = np.random.choice([0, 1, 2, 3])
        angle = rot_k * np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]], dtype=np.float32)

        p_full[:, :3] = p_full[:, :3] @ rot.T
        p_part[:, :3] = p_part[:, :3] @ rot.T

        # Random flip X
        flip_x = np.random.rand() > 0.5
        if flip_x:
            p_full[:, 0] *= -1
            p_part[:, 0] *= -1

        # Random flip Y
        flip_y = np.random.rand() > 0.5
        if flip_y:
            p_full[:, 1] *= -1
            p_part[:, 1] *= -1

        aug_params = {'rot_k': rot_k, 'flip_x': flip_x, 'flip_y': flip_y}
        return p_full, p_part, aug_params

    def __getitem__(self, idx):
        seq_str, frame_id = self.frames[idx]

        # ─── Partial scan ─────────────────────────────────────
        p_part_xyz = self._load_partial_scan(seq_str, frame_id)
        lsk_coords, lsk_probs = self._load_lsk3d_features(seq_str, frame_id)
        p_part_feats = self._attach_lsk3d_to_partial(p_part_xyz, lsk_coords, lsk_probs)
        # [N_part, 23]: xyz + 20 LSK3DNet probs

        # ─── Complete cloud ───────────────────────────────────
        p_full_xyz, p_full_labels = self._crop_full_to_ego(seq_str, frame_id)
        # Encode labels to logits: [N, 20]
        if len(p_full_labels) > 0:
            labels_torch = torch.from_numpy(p_full_labels.astype(np.int64))
            sem_logits = semantic_encode(labels_torch, self.num_classes, self.semantic_scale).numpy()
        else:
            sem_logits = np.zeros((0, self.num_classes), dtype=np.float32)
        p_full_feats = np.hstack([p_full_xyz, sem_logits])
        # [N_full, 23]: xyz + 20 semantic logits

        # ─── Filter full cloud to SSC evaluation grid ─────────
        # Concentrates all training points in the forward hemisphere
        # where evaluation actually happens, dramatically improving
        # voxel coverage (from ~16% to ~100%)
        x_min, y_min, z_min = POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[2]
        x_max, y_max, z_max = POINT_CLOUD_RANGE[3], POINT_CLOUD_RANGE[4], POINT_CLOUD_RANGE[5]
        eval_mask = ((p_full_feats[:, 0] >= x_min) & (p_full_feats[:, 0] <= x_max) &
                     (p_full_feats[:, 1] >= y_min) & (p_full_feats[:, 1] <= y_max) &
                     (p_full_feats[:, 2] >= z_min) & (p_full_feats[:, 2] <= z_max))
        p_full_eval = p_full_feats[eval_mask]
        if len(p_full_eval) >= 100:
            p_full_feats = p_full_eval  # Use only eval-grid points
        # else: keep all points as fallback (shouldn't happen normally)

        # ─── Sample to fixed sizes ────────────────────────────
        p_full_feats = self._sample_points(p_full_feats, self.num_points_full)
        p_part_feats = self._sample_points(p_part_feats, self.num_points_part)

        # ─── Augmentation ─────────────────────────────────────
        aug_params = None
        if self.augment:
            p_full_feats, p_part_feats, aug_params = self._augment(p_full_feats, p_part_feats)

        # ─── Normalization stats (from full cloud xyz) ────────
        p_mean = p_full_feats[:, :3].mean(axis=0)  # [3]
        p_std = max(p_full_feats[:, :3].std(), 1e-6)  # scalar
        p_std = np.array([p_std, p_std, p_std], dtype=np.float32)  # [3]

        # ─── Load GT SSC scene for evaluation ─────────────────
        gt_scene_path = SSC_GT_ROOT / seq_str / f"{frame_id:06d}_gt_scene.npy"
        if gt_scene_path.exists():
            gt_scene = np.load(str(gt_scene_path))  # [256, 256, 32] uint8
        else:
            gt_scene = np.zeros((256, 256, 32), dtype=np.uint8)

        result = {
            'pcd_full': torch.from_numpy(p_full_feats).float(),    # [N_full, 23]
            'pcd_part': torch.from_numpy(p_part_feats).float(),    # [N_part, 23]
            'mean': torch.from_numpy(p_mean).float(),              # [3]
            'std': torch.from_numpy(p_std).float(),                # [3]
            'gt_scene': torch.from_numpy(gt_scene).long(),         # [256, 256, 32]
            'filename': f"{seq_str}/{frame_id:06d}",
        }

        # ─── Load BEV for S30 conditioning ───────────────────
        if self.use_bev:
            bev_path = SSC_GT_ROOT / seq_str / f"{frame_id:06d}_bev.npy"
            if bev_path.exists():
                bev = np.load(str(bev_path))  # [256, 256] uint8
            else:
                bev = np.zeros((256, 256), dtype=np.uint8)
            # Apply same augmentation to BEV grid to stay aligned with point cloud
            if aug_params is not None:
                if aug_params['rot_k'] > 0:
                    bev = np.rot90(bev, k=aug_params['rot_k']).copy()
                if aug_params['flip_x']:
                    bev = np.flip(bev, axis=0).copy()
                if aug_params['flip_y']:
                    bev = np.flip(bev, axis=1).copy()
            result['bev'] = torch.from_numpy(bev).long()  # [256, 256]

        return result


def diffssc_collate_fn(batch):
    """Collate function for DiffSSCDataset.

    Stacks all tensors, collects filenames.
    """
    result = {
        'pcd_full': torch.stack([b['pcd_full'] for b in batch]),    # [B, N_full, 23]
        'pcd_part': torch.stack([b['pcd_part'] for b in batch]),    # [B, N_part, 23]
        'mean': torch.stack([b['mean'] for b in batch]),            # [B, 3]
        'std': torch.stack([b['std'] for b in batch]),              # [B, 3]
        'gt_scene': torch.stack([b['gt_scene'] for b in batch]),    # [B, 256, 256, 32]
        'filename': [b['filename'] for b in batch],
    }
    if 'bev' in batch[0]:
        result['bev'] = torch.stack([b['bev'] for b in batch])  # [B, 256, 256]
    return result
