"""
S3 DSKD Dataset for Scene Completion.

Supports three training modes for curriculum distillation:
1. TEACHER mode: GT BEV + Multi-frame LiDAR (preprocessed, 2x density)
   - Best teacher model, maximum conditioning power (49.01% mIoU)
2. INTERMEDIATE mode: GT BEV + Single-frame LiDAR (NEW)
   - Bridges multi-frame → single-frame gap
   - Distills from TEACHER mode, teaches to STUDENT mode
3. STUDENT mode: Pred BEV + Single-frame LiDAR (inference-time reality)
   - Final deployment model (~27% mIoU BEV input)
   - Supports BEV mixing (randomly use GT BEV with probability p)

Curriculum Distillation Strategy:
1. Train TEACHER (GT BEV + Multi-frame) → 49% mIoU (done)
2. Train INTERMEDIATE with DSKD from TEACHER → ~45% mIoU expected
3. Train STUDENT with DSKD from INTERMEDIATE + BEV mixing → ~35% mIoU expected

Reference: SCPNet CVPR 2023 - uses pairwise feature similarity for DSKD
"""

import multiprocessing
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class S3DSKDDataset(Dataset):
    """
    Dataset for S3 DSKD training with Teacher/Intermediate/Student modes.

    Teacher Mode:
        - Uses GT BEV (100% accurate)
        - Uses Multi-frame LiDAR (preprocessed, 2x density)

    Intermediate Mode (NEW):
        - Uses GT BEV (100% accurate)
        - Uses Single-frame LiDAR (same as inference)
        - Bridges the multi-frame → single-frame density gap

    Student Mode:
        - Uses Pred BEV (from best BEV model, ~27% mIoU)
        - Uses Single-frame LiDAR (what we have at inference)
        - Supports BEV mixing: randomly use GT BEV with probability gt_bev_prob
    """

    def __init__(
        self,
        data_root: str,
        sequences: list[str],
        resolution: int = 256,
        mode: str = 'teacher',  # 'teacher', 'intermediate', or 'student'
        pred_bev_dir: str | None = None,  # Required for student mode
        multi_frame_dir: str = 'SemanticKITTI_3D/256_multi_frame',
        augment: bool = False,
        use_rectified_labels: bool = False,  # SCPNet-style rectified labels
        gt_bev_prob: float = 0.0,  # BEV mixing: probability of using GT BEV in student mode
        lsk3d_dir: str | None = None,  # LSK3DNet 3D features dir (for S5/S6)
        geom_dir: str | None = None,  # S43: Geometric features dir (height/normals/intensity/TSDF)
        no_tsdf_sparse: bool = False,  # S46: Don't put TSDF in sparse tensor, compute TSDF BEV instead
        scpnet_pred_dir: str | None = None,  # SCPNet 3D predictions for refinement conditioning
        talos_pred_dir: str | None = None,  # TALoS TTA predictions (overrides SCPNet for real seqs)
        bev_cold_dir: str | None = None,  # S2D2 refined BEV predictions dir
        load_raw_lidar: bool = False,  # Load raw .bin pointcloud for Cylinder3D features
        force_single_frame_lidar: bool = False,  # Force single-frame lidar even in teacher mode
    ):
        """
        Args:
            data_root: Root directory containing datasets
            sequences: List of sequence IDs
            resolution: BEV resolution (256 for production)
            mode: 'teacher' (GT BEV + multi-frame), 'intermediate' (GT BEV + single-frame),
                  or 'student' (Pred BEV + single-frame)
            pred_bev_dir: Directory containing predicted BEV maps (required for student mode)
            multi_frame_dir: Directory containing preprocessed multi-frame voxels
            augment: Whether to apply data augmentation
            use_rectified_labels: Use SCPNet-style rectified labels (removes ghost trails
                                  from dynamic objects). Expected +10-25% on dynamic classes.
            gt_bev_prob: Probability of using GT BEV instead of Pred BEV in student mode.
                        This implements "BEV mixing" to bridge the GT→Pred gap gradually.
                        Recommended: 0.2-0.3 for curriculum distillation.
            load_raw_lidar: Load raw .bin pointcloud and compute 7D features for Cylinder3D.
                           Required for SCPNet denoiser mode.
        """
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.mode = mode
        self.augment = augment
        self.use_rectified_labels = use_rectified_labels
        # Use multiprocessing.Value so DataLoader workers see updates from main process
        self._gt_bev_prob = multiprocessing.Value('d', gt_bev_prob)
        self.lsk3d_dir = Path(lsk3d_dir) if lsk3d_dir else None
        self.geom_dir = Path(geom_dir) if geom_dir else None
        self.no_tsdf_sparse = no_tsdf_sparse
        self.scpnet_pred_dir = Path(scpnet_pred_dir) if scpnet_pred_dir else None
        self.talos_pred_dir = Path(talos_pred_dir) if talos_pred_dir else None
        self.bev_cold_dir = Path(bev_cold_dir) if bev_cold_dir else None
        self.load_raw_lidar = load_raw_lidar
        self.force_single_frame_lidar = force_single_frame_lidar

        # SCPNet voxelization config (from semantickitti-multiscan.yaml)
        if load_raw_lidar:
            self.scpnet_grid_size = [256, 256, 32]
            self.scpnet_max_vol = [50.0, 50.0, 3.0]
            self.scpnet_min_vol = [-50.0, -50.0, -5.0]

        assert mode in ['teacher', 'intermediate', 'student'], \
            f"Mode must be 'teacher', 'intermediate', or 'student', got {mode}"

        if mode == 'student' and pred_bev_dir is None:
            raise ValueError("pred_bev_dir is required for student mode")

        # Paths
        self.voxels_root = self.data_root / f'SemanticKITTI_3D/{resolution}'
        self.multi_frame_root = self.data_root / multi_frame_dir
        self.ssc_root = self.data_root / 'dataset_SemanticKITTI_SSC'
        self.pred_bev_root = Path(pred_bev_dir) if pred_bev_dir else None

        # Build sample list
        self.samples = []
        rectified_count = 0
        for seq in sequences:
            voxels_dir = self.voxels_root / seq
            ssc_voxels_dir = self.ssc_root / 'sequences' / seq / 'voxels'

            if not voxels_dir.exists():
                print(f"Warning: Voxels dir not found for seq {seq}: {voxels_dir}")
                continue

            # Find all samples with BEV and scene labels
            bev_files = sorted(voxels_dir.glob('*_bev.npy'))
            for bev_file in bev_files:
                frame_id = bev_file.stem.replace('_bev', '')  # e.g., '000100'

                # Required files
                gt_bev_file = bev_file
                single_voxels_file = voxels_dir / f'{frame_id}_voxels.npy'

                # Scene label: try _gt_scene.npy first (synthetic, already in learning space),
                # then .label (real data, needs remapping via LEARNING_MAP)
                gt_scene_npy = voxels_dir / f'{frame_id}_gt_scene.npy'
                is_npy_label = False
                if gt_scene_npy.exists():
                    scene_label_file = gt_scene_npy
                    is_npy_label = True
                elif use_rectified_labels:
                    rectified_file = ssc_voxels_dir / f'{frame_id}_rectified.label'
                    if rectified_file.exists():
                        scene_label_file = rectified_file
                        rectified_count += 1
                    else:
                        scene_label_file = ssc_voxels_dir / f'{frame_id}.label'
                else:
                    scene_label_file = ssc_voxels_dir / f'{frame_id}.label'

                if not single_voxels_file.exists() or not scene_label_file.exists():
                    continue

                # Invalid mask for official eval (packed bits)
                invalid_file = ssc_voxels_dir / f'{frame_id}.invalid'

                sample = {
                    'seq': seq,
                    'frame': frame_id,
                    'gt_bev': str(gt_bev_file),
                    'single_voxels': str(single_voxels_file),
                    'scene_label': str(scene_label_file),
                    'is_npy_label': is_npy_label,
                    'invalid_file': str(invalid_file) if invalid_file.exists() else None,
                }

                # Teacher mode: need multi-frame preprocessed (skip for synthetic)
                # ``force_single_frame_lidar`` suppresses multi-frame loading even in
                # teacher mode: the __getitem__ path then falls back to single_voxels.
                if mode == 'teacher':
                    if force_single_frame_lidar:
                        pass  # leave sample['multi_frame'] unset → single-frame fallback
                    else:
                        multi_frame_file = self.multi_frame_root / seq / f'{frame_id}.npz'
                        if multi_frame_file.exists():
                            sample['multi_frame'] = str(multi_frame_file)
                        elif scpnet_pred_dir is not None:
                            pass  # Cold diffusion doesn't need multi-frame
                        else:
                            continue  # Skip if multi-frame not available

                # Intermediate mode: GT BEV + Single-frame for STUDENT, Multi-frame for TEACHER
                # SCPNet DSKD: Teacher uses multi-frame (dense), Student uses single-frame (sparse)
                # We need BOTH for proper knowledge distillation
                if mode == 'intermediate':
                    multi_frame_file = self.multi_frame_root / seq / f'{frame_id}.npz'
                    if multi_frame_file.exists():
                        sample['multi_frame'] = str(multi_frame_file)
                    else:
                        continue  # Skip if multi-frame not available for DSKD

                # Student mode: need predicted BEV + multi-frame for teacher DSKD
                if mode == 'student':
                    pred_bev_file = self.pred_bev_root / seq / f'{frame_id}_bev.npy'
                    if pred_bev_file.exists():
                        sample['pred_bev'] = str(pred_bev_file)
                    else:
                        continue  # Skip if pred BEV not available
                    # Load multi-frame for teacher (trained with MF=49%, SF=~43%)
                    multi_frame_file = self.multi_frame_root / seq / f'{frame_id}.npz'
                    if multi_frame_file.exists():
                        sample['multi_frame'] = str(multi_frame_file)

                # LSK3DNet 3D features (for S5/S6)
                if self.lsk3d_dir is not None:
                    lsk3d_file = self.lsk3d_dir / seq / f'{frame_id}_lsk3d_3d.npz'
                    if lsk3d_file.exists():
                        sample['lsk3d'] = str(lsk3d_file)
                    else:
                        continue  # Skip if LSK3DNet features not available

                # S43: Geometric features (height/normals/intensity/TSDF)
                if self.geom_dir is not None:
                    geom_file = self.geom_dir / seq / f'{frame_id}_geom.npz'
                    if geom_file.exists():
                        sample['geom'] = str(geom_file)
                    else:
                        continue  # Skip if geometric features not available

                # 3D predictions for cold diffusion conditioning
                # Try TALoS first (better quality), fall back to SCPNet
                pred_found = False
                if self.talos_pred_dir is not None:
                    talos_file = self.talos_pred_dir / seq / f'{frame_id}_pred.npy'
                    if talos_file.exists():
                        sample['scpnet_pred'] = str(talos_file)
                        pred_found = True
                if not pred_found and self.scpnet_pred_dir is not None:
                    scpnet_file = self.scpnet_pred_dir / seq / f'{frame_id}_pred.npy'
                    if scpnet_file.exists():
                        sample['scpnet_pred'] = str(scpnet_file)
                        pred_found = True
                if not pred_found and (self.scpnet_pred_dir is not None or self.talos_pred_dir is not None):
                    continue  # Skip if no prediction available

                self.samples.append(sample)

        label_type = f"rectified ({rectified_count})" if use_rectified_labels else "original"
        mode_desc = mode
        if mode == 'student' and gt_bev_prob > 0:
            mode_desc = f"student (BEV mixing p={gt_bev_prob})"
        print(f"S3DSKDDataset ({mode_desc} mode, {label_type} labels): {len(self.samples)} samples from {len(sequences)} sequences")

    @property
    def gt_bev_prob(self):
        """Read from shared memory so DataLoader workers see main-process updates."""
        return self._gt_bev_prob.value

    @gt_bev_prob.setter
    def gt_bev_prob(self, value):
        """Write to shared memory so DataLoader workers see the update."""
        self._gt_bev_prob.value = value

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load GT 3D scene
        if sample.get('is_npy_label', False):
            gt_scene = np.load(sample['scene_label']).astype(np.int64)  # Already in learning space
        else:
            gt_scene = self._load_label(sample['scene_label'])  # [256, 256, 32]

        # Track whether we used GT BEV (for student mode with mixing)
        used_gt_bev = False
        # TSDF BEV: computed when no_tsdf_sparse=True, None otherwise
        tsdf_bev = None

        # Load BEV based on mode
        if hasattr(self, 'bev_cold_dir') and self.bev_cold_dir is not None:
            # Use S2D2 refined BEV (pre-generated by BEV-A)
            bev_cold_file = self.bev_cold_dir / sample['seq'] / f'{sample["frame"]}_bev.npy'
            if bev_cold_file.exists():
                bev = np.load(str(bev_cold_file)).astype(np.int64)
            else:
                bev = np.load(sample['gt_bev']).astype(np.int64)  # Fallback to GT
            used_gt_bev = False
        elif self.mode == 'teacher' or self.mode == 'intermediate':
            # Both teacher and intermediate modes use GT BEV
            bev = np.load(sample['gt_bev']).astype(np.int64)  # GT BEV
            used_gt_bev = True
        else:
            # Student mode: use Pred BEV, but with optional mixing
            if self.gt_bev_prob > 0 and random.random() < self.gt_bev_prob:
                # BEV mixing: use GT BEV with probability gt_bev_prob
                bev = np.load(sample['gt_bev']).astype(np.int64)
                used_gt_bev = True
            else:
                bev = np.load(sample['pred_bev']).astype(np.int64)  # Pred BEV

        # Load LiDAR voxels
        # SCPNet DSKD: Teacher uses multi-frame (dense), Student uses single-frame (sparse)
        lidar_multi = None  # Multi-frame for teacher (in INTERMEDIATE mode)

        # LSK3DNet 3D features: replace binary LiDAR with 20ch soft logits
        if self.lsk3d_dir is not None and 'lsk3d' in sample:
            data = np.load(sample['lsk3d'])
            coords = data['coords'].astype(np.int64)  # [N, 3] uint8
            probs = data['probs'].astype(np.float32)   # [N, 20] float16->float32

            # Allocate full tensor up front to avoid concatenation overhead
            has_geom = self.geom_dir is not None and 'geom' in sample
            if has_geom:
                n_ch = 29 if self.no_tsdf_sparse else 30
            else:
                n_ch = 20
            lidar = np.zeros((n_ch, self.resolution, self.resolution, 32), dtype=np.float32)

            valid_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < self.resolution) &
                (coords[:, 1] >= 0) & (coords[:, 1] < self.resolution) &
                (coords[:, 2] >= 0) & (coords[:, 2] < 32)
            )
            coords = coords[valid_mask]
            probs = probs[valid_mask]
            lidar[:20, coords[:, 0], coords[:, 1], coords[:, 2]] = probs.T
            # Mark as multi-channel (skip unsqueeze later)
            lidar_is_multichannel = True

            # S43: Fill geometric features in-place (channels 20-29)
            if has_geom:
                geom = np.load(sample['geom'])
                gc = geom['coords'].astype(np.int64)     # [N_occ, 3]
                # Channels 20-24: height features [N_occ, 5]
                hf = geom['height_feats'].astype(np.float32)
                # Channels 25-27: surface normals [N_occ, 3]
                nm = geom['normals'].astype(np.float32)
                # Channel 28: intensity [N_occ, 1]
                it = geom['intensity'].astype(np.float32)
                # Channel 29: TSDF values at original+generated coords
                tc = geom['tsdf_coords'].astype(np.int64)  # [M, 3]
                tv = geom['tsdf_values'].astype(np.float32)  # [M, 1]

                # Fill channels 20-24 (height) at original voxel coords
                gv = ((gc[:, 0] >= 0) & (gc[:, 0] < self.resolution) &
                      (gc[:, 1] >= 0) & (gc[:, 1] < self.resolution) &
                      (gc[:, 2] >= 0) & (gc[:, 2] < 32))
                gc_v = gc[gv]
                lidar[20:25, gc_v[:, 0], gc_v[:, 1], gc_v[:, 2]] = hf[gv].T
                # Fill channels 25-27 (normals)
                lidar[25:28, gc_v[:, 0], gc_v[:, 1], gc_v[:, 2]] = nm[gv].T
                # Fill channel 28 (intensity)
                lidar[28, gc_v[:, 0], gc_v[:, 1], gc_v[:, 2]] = it[gv, 0]

                # TSDF handling: either in sparse tensor (30ch) or BEV projection (29ch)
                tv_mask = ((tc[:, 0] >= 0) & (tc[:, 0] < self.resolution) &
                           (tc[:, 1] >= 0) & (tc[:, 1] < self.resolution) &
                           (tc[:, 2] >= 0) & (tc[:, 2] < 32))
                tc_v = tc[tv_mask]
                tv_v = tv[tv_mask, 0]

                if self.no_tsdf_sparse:
                    # S46: Compute TSDF BEV features (4ch column statistics)
                    # Instead of putting TSDF in sparse tensor (causes 4.3x voxel expansion),
                    # compute per-column statistics that capture the spatial field structure.
                    tsdf_vol = np.zeros((self.resolution, self.resolution, 32), dtype=np.float32)
                    tsdf_mask = np.zeros((self.resolution, self.resolution, 32), dtype=bool)
                    tsdf_vol[tc_v[:, 0], tc_v[:, 1], tc_v[:, 2]] = tv_v
                    tsdf_mask[tc_v[:, 0], tc_v[:, 1], tc_v[:, 2]] = True

                    col_has_tsdf = tsdf_mask.any(axis=2)  # [H, W]
                    tsdf_bev = np.zeros((4, self.resolution, self.resolution), dtype=np.float32)
                    # Vectorized column stats
                    tsdf_min = np.where(tsdf_mask, tsdf_vol, 999.0).min(axis=2)
                    tsdf_max = np.where(tsdf_mask, tsdf_vol, -999.0).max(axis=2)
                    tsdf_sum = (tsdf_vol * tsdf_mask).sum(axis=2)
                    tsdf_count = tsdf_mask.sum(axis=2).clip(min=1)
                    tsdf_mean = tsdf_sum / tsdf_count
                    surface_density = ((np.abs(tsdf_vol) < 0.1) & tsdf_mask).sum(axis=2) / 32.0
                    tsdf_bev[0] = np.where(col_has_tsdf, tsdf_min, 0.0)
                    tsdf_bev[1] = np.where(col_has_tsdf, tsdf_max, 0.0)
                    tsdf_bev[2] = np.where(col_has_tsdf, tsdf_mean, 0.0)
                    tsdf_bev[3] = surface_density
                else:
                    # Original S43: TSDF in channel 29 of sparse tensor (causes expansion)
                    lidar[29, tc_v[:, 0], tc_v[:, 1], tc_v[:, 2]] = tv_v

        elif self.mode == 'teacher':
            if 'multi_frame' in sample:
                # Teacher mode: use multi-frame
                data = np.load(sample['multi_frame'])
                coords = data['coords'].astype(np.int64)  # [N, 3]

                # Convert sparse to dense
                lidar = np.zeros((self.resolution, self.resolution, 32), dtype=np.float32)
                valid_mask = (
                    (coords[:, 0] >= 0) & (coords[:, 0] < self.resolution) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < self.resolution) &
                    (coords[:, 2] >= 0) & (coords[:, 2] < 32)
                )
                coords = coords[valid_mask]
                lidar[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            else:
                # Fallback to single-frame (synthetic data, no multi-frame available)
                lidar = np.load(sample['single_voxels']).astype(np.float32)
                lidar = (lidar > 0).astype(np.float32)
            lidar_is_multichannel = False

        elif self.mode == 'intermediate':
            # Intermediate mode: BOTH single-frame (student) AND multi-frame (teacher)
            # Student uses single-frame
            lidar = np.load(sample['single_voxels']).astype(np.float32)
            lidar = (lidar > 0).astype(np.float32)
            lidar_is_multichannel = False

            # Teacher uses multi-frame (for DSKD)
            data = np.load(sample['multi_frame'])
            coords = data['coords'].astype(np.int64)
            lidar_multi = np.zeros((self.resolution, self.resolution, 32), dtype=np.float32)
            valid_mask = (
                (coords[:, 0] >= 0) & (coords[:, 0] < self.resolution) &
                (coords[:, 1] >= 0) & (coords[:, 1] < self.resolution) &
                (coords[:, 2] >= 0) & (coords[:, 2] < 32)
            )
            coords = coords[valid_mask]
            lidar_multi[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

        else:
            # Student mode: Single-frame LiDAR for student + multi-frame for teacher
            lidar = np.load(sample['single_voxels']).astype(np.float32)
            lidar = (lidar > 0).astype(np.float32)
            lidar_is_multichannel = False

            # Load multi-frame for teacher if available
            if 'multi_frame' in sample:
                data = np.load(sample['multi_frame'])
                coords = data['coords'].astype(np.int64)
                lidar_multi = np.zeros((self.resolution, self.resolution, 32), dtype=np.float32)
                valid_mask = (
                    (coords[:, 0] >= 0) & (coords[:, 0] < self.resolution) &
                    (coords[:, 1] >= 0) & (coords[:, 1] < self.resolution) &
                    (coords[:, 2] >= 0) & (coords[:, 2] < 32)
                )
                coords = coords[valid_mask]
                lidar_multi[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

        # For student mode, load GT BEV BEFORE augmentation so it gets the same transforms
        gt_bev = None
        if self.mode == 'student':
            gt_bev = np.load(sample['gt_bev']).astype(np.int64)

        # Load SCPNet predictions BEFORE augmentation so they get the same transforms
        scpnet_pred = None
        if self.scpnet_pred_dir is not None and 'scpnet_pred' in sample:
            scpnet_pred = np.load(sample['scpnet_pred']).astype(np.int64)  # [256, 256, 32]

        # Apply augmentation (apply same transform to all spatial data)
        if self.augment:
            if lidar_multi is not None:
                aug_result = self._augment_with_multi(
                    lidar, gt_scene, bev, lidar_multi, gt_bev=gt_bev,
                    lidar_is_multichannel=lidar_is_multichannel,
                    tsdf_bev=tsdf_bev, scpnet_pred=scpnet_pred)
                lidar, gt_scene, bev, lidar_multi, gt_bev = aug_result[:5]
                idx = 5
                if tsdf_bev is not None:
                    tsdf_bev = aug_result[idx]; idx += 1
                if scpnet_pred is not None:
                    scpnet_pred = aug_result[idx]; idx += 1
            else:
                aug_result = self._augment(
                    lidar, gt_scene, bev, gt_bev=gt_bev,
                    lidar_is_multichannel=lidar_is_multichannel,
                    tsdf_bev=tsdf_bev, scpnet_pred=scpnet_pred)
                lidar, gt_scene, bev, gt_bev = aug_result[:4]
                idx = 4
                if tsdf_bev is not None:
                    tsdf_bev = aug_result[idx]; idx += 1
                if scpnet_pred is not None:
                    scpnet_pred = aug_result[idx]; idx += 1

        # Convert to tensors
        if lidar_is_multichannel:
            lidar = torch.from_numpy(lidar).float()  # [20, 256, 256, 32] already has channel dim
        else:
            lidar = torch.from_numpy(lidar).float().unsqueeze(0)  # [1, 256, 256, 32]
        gt_scene = torch.from_numpy(gt_scene).long()  # [256, 256, 32]
        bev = torch.from_numpy(bev).long()  # [256, 256]

        result = {
            'lidar': lidar,
            'gt_scene': gt_scene,
            'bev': bev,
            'seq': sample['seq'],
            'frame': sample['frame']
        }

        # Provide multi-frame lidar for teacher DSKD (intermediate and student modes)
        if lidar_multi is not None:
            result['lidar_multi'] = torch.from_numpy(lidar_multi).float().unsqueeze(0)

        # For student mode, also provide GT BEV for teacher KD
        if gt_bev is not None:
            result['gt_bev'] = torch.from_numpy(gt_bev).long()
            result['used_gt_bev'] = used_gt_bev

        # S46: TSDF BEV features (4ch column stats)
        if tsdf_bev is not None:
            result['tsdf_bev'] = torch.from_numpy(tsdf_bev.copy()).float()  # [4, H, W]

        # SCPNet 3D predictions for refinement conditioning (already loaded and augmented above)
        if scpnet_pred is not None:
            result['scpnet_pred'] = torch.from_numpy(scpnet_pred.copy()).long()  # [256, 256, 32]

        # Invalid mask for official-style eval (packed bits → [256,256,32])
        if sample.get('invalid_file') is not None:
            inv_packed = np.fromfile(sample['invalid_file'], dtype=np.uint8)
            inv_mask = np.unpackbits(inv_packed).reshape(256, 256, 32)
            result['invalid_mask'] = torch.from_numpy(inv_mask).bool()
        else:
            # No invalid file (e.g. synthetic data) — all voxels valid
            result['invalid_mask'] = torch.zeros(256, 256, 32, dtype=torch.bool)

        # Raw LiDAR point cloud for Cylinder3D feature extraction
        if self.load_raw_lidar:
            pt_fea, grid_ind = self._load_raw_lidar(sample['seq'], sample['frame'])
            result['pt_fea'] = torch.from_numpy(pt_fea).float()       # [M, 7]
            result['grid_ind'] = torch.from_numpy(grid_ind).long()     # [M, 3]

        return result

    def _load_label(self, path: str) -> np.ndarray:
        """Load and remap scene labels."""
        # Load raw labels
        labels = np.fromfile(path, dtype=np.uint16)
        labels = labels.reshape(256, 256, 32)

        # Remap to learning IDs
        labels = self._remap_labels(labels)

        return labels.astype(np.int64)

    def _remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """Remap raw labels to learning IDs (0-19)."""
        # SemanticKITTI learning map
        LEARNING_MAP = {
            0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
            30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
            51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
            99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
        }

        remapped = np.zeros_like(labels)
        for k, v in LEARNING_MAP.items():
            remapped[labels == k] = v

        return remapped

    def _load_raw_lidar(self, seq: str, frame: str) -> tuple[np.ndarray, np.ndarray]:
        """Load raw .bin pointcloud and compute 7D features + grid indices.

        Uses SCPNet's voxelization: compute cylinder coords, per-point
        features (dx, dy, dz, x, y, z, intensity) for Cylinder3D.

        Returns:
            pt_fea: [M, 7] per-point features
            grid_ind: [M, 3] voxel grid indices
        """
        pc_path = self.ssc_root / 'sequences' / seq / 'velodyne' / f'{frame}.bin'
        raw_data = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)

        xyz = raw_data[:, :3]
        sig = raw_data[:, 3]

        grid_size = np.array(self.scpnet_grid_size)
        max_vol = np.array(self.scpnet_max_vol, dtype=np.float64)
        min_vol = np.array(self.scpnet_min_vol, dtype=np.float64)

        # SCPNet x-axis centering
        x_bias = (max_vol[0] - min_vol[0]) / 2
        min_bound = min_vol.copy()
        max_bound = max_vol.copy()
        min_bound[0] -= x_bias
        max_bound[0] -= x_bias
        xyz_shifted = xyz.copy()
        xyz_shifted[:, 0] -= x_bias

        # Clip to volume bounds
        valid = np.ones(len(xyz_shifted), dtype=bool)
        for ci in range(3):
            valid &= (xyz_shifted[:, ci] >= min_bound[ci]) & (xyz_shifted[:, ci] <= max_bound[ci])
        xyz_shifted = xyz_shifted[valid]
        sig = sig[valid]

        # Compute grid indices
        crop_range = max_bound - min_bound
        intervals = crop_range / (grid_size - 1)
        grid_ind = np.floor(
            (np.clip(xyz_shifted, min_bound, max_bound) - min_bound) / intervals
        ).astype(np.int64)

        # 7D features: (dx, dy, dz, x, y, z, intensity)
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_shifted - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_shifted), axis=1)  # [M, 6]
        pt_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)  # [M, 7]

        return pt_fea.astype(np.float32), grid_ind

    def _transform_normals(self, lidar, flip_x=False, flip_y=False, rot_k=0):
        """Transform normal vectors (channels 25-27) under spatial augmentation.

        Normal vectors are directional and must be transformed when
        spatial data is flipped or rotated. Only applies when lidar has
        geom features (29+ channels).

        Args:
            lidar: [C, H, W, D] multi-channel tensor with C >= 29
            flip_x: whether X axis was flipped
            flip_y: whether Y axis was flipped
            rot_k: number of 90-degree rotations applied (0-3)
        """
        if lidar.shape[0] < 29:
            return lidar

        # Channel 25 = nx, Channel 26 = ny, Channel 27 = nz
        if flip_x:
            lidar[25] = -lidar[25]  # negate nx
        if flip_y:
            lidar[26] = -lidar[26]  # negate ny

        if rot_k > 0:
            nx = lidar[25].copy()
            ny = lidar[26].copy()
            # Rotate (nx, ny) as 2D vector by k*90 degrees CCW
            # np.rot90(arr, k=1, axes=(1,2)) is CCW: (x,y) → (-y, x)
            # Normal vectors must rotate the same way:
            # k=1 CCW: (nx,ny) -> (-ny, nx)
            # k=2 180: (nx,ny) -> (-nx, -ny)
            # k=3 CW:  (nx,ny) -> (ny, -nx)
            if rot_k == 1:
                lidar[25] = -ny
                lidar[26] = nx
            elif rot_k == 2:
                lidar[25] = -nx
                lidar[26] = -ny
            elif rot_k == 3:
                lidar[25] = ny
                lidar[26] = -nx

        return lidar

    def _augment(
        self,
        lidar: np.ndarray,
        gt_scene: np.ndarray,
        bev: np.ndarray,
        gt_bev: np.ndarray = None,
        lidar_is_multichannel: bool = False,
        tsdf_bev: np.ndarray = None,
        scpnet_pred: np.ndarray = None,
    ) -> tuple[np.ndarray, ...]:
        """Apply data augmentation.

        For single-channel lidar [H,W,D]: spatial axes are (0,1).
        For multi-channel lidar [C,H,W,D]: spatial axes are (1,2).
        tsdf_bev [4,H,W]: spatial axes are (1,2).
        """
        # Spatial axes for lidar depend on whether it has a channel dim
        lx, ly = (1, 2) if lidar_is_multichannel else (0, 1)
        # Geom features present if 29ch (no_tsdf_sparse) or 30ch (with TSDF)
        has_geom = lidar_is_multichannel and lidar.shape[0] >= 29

        # Random flip X
        flip_x = random.random() > 0.5
        if flip_x:
            lidar = np.flip(lidar, axis=lx).copy()
            gt_scene = np.flip(gt_scene, axis=0).copy()
            bev = np.flip(bev, axis=0).copy()
            if gt_bev is not None:
                gt_bev = np.flip(gt_bev, axis=0).copy()
            if tsdf_bev is not None:
                tsdf_bev = np.flip(tsdf_bev, axis=1).copy()
            if scpnet_pred is not None:
                scpnet_pred = np.flip(scpnet_pred, axis=0).copy()

        # Random flip Y
        flip_y = random.random() > 0.5
        if flip_y:
            lidar = np.flip(lidar, axis=ly).copy()
            gt_scene = np.flip(gt_scene, axis=1).copy()
            bev = np.flip(bev, axis=1).copy()
            if gt_bev is not None:
                gt_bev = np.flip(gt_bev, axis=1).copy()
            if tsdf_bev is not None:
                tsdf_bev = np.flip(tsdf_bev, axis=2).copy()
            if scpnet_pred is not None:
                scpnet_pred = np.flip(scpnet_pred, axis=1).copy()

        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lidar = np.rot90(lidar, k, axes=(lx, ly)).copy()
            gt_scene = np.rot90(gt_scene, k, axes=(0, 1)).copy()
            bev = np.rot90(bev, k=k).copy()
            if gt_bev is not None:
                gt_bev = np.rot90(gt_bev, k=k).copy()
            if tsdf_bev is not None:
                tsdf_bev = np.rot90(tsdf_bev, k, axes=(1, 2)).copy()
            if scpnet_pred is not None:
                scpnet_pred = np.rot90(scpnet_pred, k, axes=(0, 1)).copy()

        # Transform normal vectors after spatial augmentation
        if has_geom:
            lidar = self._transform_normals(lidar, flip_x, flip_y, k)

        result = [lidar, gt_scene, bev, gt_bev]
        if tsdf_bev is not None:
            result.append(tsdf_bev)
        if scpnet_pred is not None:
            result.append(scpnet_pred)
        return tuple(result)

    def _augment_with_multi(
        self,
        lidar: np.ndarray,
        gt_scene: np.ndarray,
        bev: np.ndarray,
        lidar_multi: np.ndarray,
        gt_bev: np.ndarray = None,
        lidar_is_multichannel: bool = False,
        tsdf_bev: np.ndarray = None,
        scpnet_pred: np.ndarray = None,
    ) -> tuple[np.ndarray, ...]:
        """Apply data augmentation to both lidar arrays (for DSKD).

        SCPNet DSKD requires same augmentation applied to both teacher and student inputs.
        """
        lx, ly = (1, 2) if lidar_is_multichannel else (0, 1)
        has_geom = lidar_is_multichannel and lidar.shape[0] >= 29

        # Random flip X
        flip_x = random.random() > 0.5
        if flip_x:
            lidar = np.flip(lidar, axis=lx).copy()
            lidar_multi = np.flip(lidar_multi, axis=0).copy()
            gt_scene = np.flip(gt_scene, axis=0).copy()
            bev = np.flip(bev, axis=0).copy()
            if gt_bev is not None:
                gt_bev = np.flip(gt_bev, axis=0).copy()
            if tsdf_bev is not None:
                tsdf_bev = np.flip(tsdf_bev, axis=1).copy()
            if scpnet_pred is not None:
                scpnet_pred = np.flip(scpnet_pred, axis=0).copy()

        # Random flip Y
        flip_y = random.random() > 0.5
        if flip_y:
            lidar = np.flip(lidar, axis=ly).copy()
            lidar_multi = np.flip(lidar_multi, axis=1).copy()
            gt_scene = np.flip(gt_scene, axis=1).copy()
            bev = np.flip(bev, axis=1).copy()
            if gt_bev is not None:
                gt_bev = np.flip(gt_bev, axis=1).copy()
            if tsdf_bev is not None:
                tsdf_bev = np.flip(tsdf_bev, axis=2).copy()
            if scpnet_pred is not None:
                scpnet_pred = np.flip(scpnet_pred, axis=1).copy()

        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            lidar = np.rot90(lidar, k, axes=(lx, ly)).copy()
            lidar_multi = np.rot90(lidar_multi, k, axes=(0, 1)).copy()
            gt_scene = np.rot90(gt_scene, k, axes=(0, 1)).copy()
            bev = np.rot90(bev, k=k).copy()
            if gt_bev is not None:
                gt_bev = np.rot90(gt_bev, k=k).copy()
            if tsdf_bev is not None:
                tsdf_bev = np.rot90(tsdf_bev, k, axes=(1, 2)).copy()
            if scpnet_pred is not None:
                scpnet_pred = np.rot90(scpnet_pred, k, axes=(0, 1)).copy()

        # Transform normal vectors after spatial augmentation
        if has_geom:
            lidar = self._transform_normals(lidar, flip_x, flip_y, k)

        result = [lidar, gt_scene, bev, lidar_multi, gt_bev]
        if tsdf_bev is not None:
            result.append(tsdf_bev)
        if scpnet_pred is not None:
            result.append(scpnet_pred)
        return tuple(result)


def scpnet_collate_fn(batch):
    """Custom collate function that handles variable-length pt_fea/grid_ind.

    pt_fea and grid_ind have different lengths per sample (different number
    of LiDAR points). We keep them as lists instead of stacking into tensors.
    """
    result = {}
    keys = batch[0].keys()
    for key in keys:
        if key in ('pt_fea', 'grid_ind'):
            # Variable-length: keep as list of tensors
            result[key] = [sample[key] for sample in batch]
        elif key in ('seq', 'frame', 'used_gt_bev'):
            # Non-tensor metadata
            result[key] = [sample[key] for sample in batch]
        else:
            # Standard tensors: stack
            result[key] = torch.stack([sample[key] for sample in batch])
    return result


def create_s3_dataloader(
    data_root: str,
    sequences: list[str],
    mode: str = 'teacher',
    pred_bev_dir: str | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    gt_bev_prob: float = 0.0,
    use_rectified_labels: bool = False,
    lsk3d_dir: str | None = None,
    geom_dir: str | None = None,
    no_tsdf_sparse: bool = False,
    scpnet_pred_dir: str | None = None,
    talos_pred_dir: str | None = None,
    bev_cold_dir: str | None = None,
    load_raw_lidar: bool = False,
) -> DataLoader:
    """Create DataLoader for S3 DSKD training.

    Args:
        data_root: Root directory containing datasets
        sequences: List of sequence IDs
        mode: 'teacher', 'intermediate', or 'student'
        pred_bev_dir: Directory containing predicted BEV maps (required for student mode)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        augment: Whether to apply data augmentation
        gt_bev_prob: BEV mixing probability for student mode
        use_rectified_labels: Use SCPNet-style rectified labels
        lsk3d_dir: LSK3DNet 3D features directory
        geom_dir: S43 geometric features directory
        load_raw_lidar: Load raw .bin for Cylinder3D features
    """

    dataset = S3DSKDDataset(
        data_root=data_root,
        sequences=sequences,
        mode=mode,
        pred_bev_dir=pred_bev_dir,
        augment=augment,
        gt_bev_prob=gt_bev_prob,
        use_rectified_labels=use_rectified_labels,
        lsk3d_dir=lsk3d_dir,
        geom_dir=geom_dir,
        no_tsdf_sparse=no_tsdf_sparse,
        scpnet_pred_dir=scpnet_pred_dir,
        talos_pred_dir=talos_pred_dir,
        bev_cold_dir=bev_cold_dir,
        load_raw_lidar=load_raw_lidar,
    )

    collate = scpnet_collate_fn if load_raw_lidar else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )


if __name__ == '__main__':
    # Test the dataset
    print("Testing S3DSKDDataset...")

    # Test teacher mode
    print("\n=== Teacher Mode ===")
    teacher_dataset = S3DSKDDataset(
        data_root='<repo>/datasets',
        sequences=['00'],
        mode='teacher',
        augment=True
    )
    print(f"Teacher samples: {len(teacher_dataset)}")

    if len(teacher_dataset) > 0:
        sample = teacher_dataset[0]
        print(f"LiDAR shape: {sample['lidar'].shape}")
        print(f"GT scene shape: {sample['gt_scene'].shape}")
        print(f"BEV shape: {sample['bev'].shape}")
        print(f"LiDAR occupancy: {(sample['lidar'] > 0).sum().item() / sample['lidar'].numel() * 100:.2f}%")
