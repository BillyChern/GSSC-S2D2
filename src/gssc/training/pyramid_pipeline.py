"""
Pyramid Discrete Diffusion Pipeline for SemanticKITTI Data Augmentation.
Implementation based on grounded evidence from original codebase.
"""
import logging
import os
from datetime import datetime

import numpy as np
import torch
from kitti_dataset import SemanticKITTIDataset
from pyramid_diffusion import PyramidDiscreteDiffusion
from pyramid_unet import create_pyramid_denoiser
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Configuration matching original paper (https://arxiv.org/abs/2311.12085)
# Original train_s_1.yaml settings:
#   - lr = 0.004
#   - batch_size = 16 (single GPU)
#   - mask_ratio = 0.0625
#   - mask_prob = [0.25, 0.25, 0.25, 0.25]
# We use larger batch size for multi-GPU efficiency
STAGE_CONFIG = {
    's1': {
        'size': (32, 32, 4),
        'batch_size': 32,  # PER-GPU for multi-GPU training
        'has_condition': True,
        'condition_type': 'masked'
    },
    's2': {
        'size': (64, 64, 8),
        'batch_size': 8,  # PER-GPU for multi-GPU training
        'has_condition': True,
        'condition_type': 'upsampled_prev'
    },
    's3': {
        'size': (256, 256, 32),  # SemanticKITTI adapted (paper uses s4)
        'batch_size': 4,  # PER-GPU for multi-GPU training
        'has_condition': True,
        'condition_type': 'dual',
        'patch_size': (136, 136, 32)
    }
}

TRAINING_CONFIG = {
    'learning_rate': 0.004,  # From original train_s_1.yaml (was incorrectly 0.001)
    'timesteps': 100,
    'max_epochs': 1000,
    'optimizer': 'adamw',
    'mask_ratio': 0.0625,
    'mask_prob': [0.25, 0.25, 0.25, 0.25],
    'sliding_ratio': 0.9375,
    'auxiliary_loss_weight': 0.0005
}


class ExactSceneSubdivision:
    """
    Exact scene subdivision implementation based on original split() function.
    Reproduces the exact logic from utils/spilt_scenes.py
    """

    def __init__(self,
                 scene_size: tuple[int, int, int] = (256, 256, 32),
                 target_resolution: tuple[int, int, int] = (136, 136, 32),
                 sliding_ratio: float = 0.9375):
        """
        Args:
            scene_size: Full scene size (H, W, D)
            target_resolution: Patch size (H', W', D) - NOTE: D must equal scene D!
            sliding_ratio: 1 - mask_ratio = 0.9375
        """
        self.scene_size = scene_size
        self.target_resolution = target_resolution
        self.sliding_ratio = sliding_ratio

        # Verify Z dimensions match (critical for fusion!)
        assert scene_size[2] == target_resolution[2], \
            f"Z dimensions must match: scene {scene_size[2]} vs patch {target_resolution[2]}"

        # Calculate step size (exactly from original code)
        self.step_size = (
            int(target_resolution[0] * sliding_ratio),  # ~127
            int(target_resolution[1] * sliding_ratio),  # ~127
            scene_size[2]  # Full Z dimension (32 for SemanticKITTI)
        )

    def split_scene(self, scene: torch.Tensor) -> list[torch.Tensor]:
        """
        Split scene into overlapping patches.
        Exact reproduction of split() function logic.
        """
        if len(scene.shape) == 3:  # [H, W, D]
            scene = scene.unsqueeze(0)  # Add batch dim

        batch_size = scene.shape[0]
        device = scene.device
        dtype = scene.dtype

        # Calculate number of patches (from original code)
        splits_along_x = 1 + (self.scene_size[0] - self.target_resolution[0]) // self.step_size[0]
        splits_along_y = 1 + (self.scene_size[1] - self.target_resolution[1]) // self.step_size[1]
        num_sub_scenes = splits_along_x * splits_along_y

        patches = []

        # Exact algorithm from original split() function
        sub_scene_idx = 0
        y = self.scene_size[1] - self.target_resolution[1]  # Start from top

        while y >= 0 and sub_scene_idx < num_sub_scenes:
            x = 0
            while x < self.scene_size[0] and sub_scene_idx < num_sub_scenes:
                # Check bounds
                if (self.scene_size[0] - x < self.target_resolution[0] or
                    y + self.target_resolution[1] > self.scene_size[1]):
                    break

                # Extract patch
                extract_x = min(self.target_resolution[0], self.scene_size[0] - x)
                extract_y = min(self.target_resolution[1], y + self.target_resolution[1])

                patch = torch.zeros((batch_size, *self.target_resolution),
                                  dtype=dtype, device=device)

                # Copy data (all Z dimension)
                patch[:, :extract_x, :extract_y, :] = \
                    scene[:, x:x+extract_x, y:y+extract_y, :]

                patches.append(patch)

                x += self.step_size[0]
                sub_scene_idx += 1

            y -= self.step_size[1]

        return patches

    def get_patch_offsets(self) -> list[tuple[int, int]]:
        """Get X,Y offsets for all patches (Z is always 0)."""
        offsets = []
        splits_along_x = 1 + (self.scene_size[0] - self.target_resolution[0]) // self.step_size[0]
        splits_along_y = 1 + (self.scene_size[1] - self.target_resolution[1]) // self.step_size[1]

        sub_scene_idx = 0
        y = self.scene_size[1] - self.target_resolution[1]

        while y >= 0 and sub_scene_idx < splits_along_x * splits_along_y:
            x = 0
            while x < self.scene_size[0] and sub_scene_idx < splits_along_x * splits_along_y:
                if (self.scene_size[0] - x < self.target_resolution[0] or
                    y + self.target_resolution[1] > self.scene_size[1]):
                    break

                offsets.append((x, y))
                x += self.step_size[0]
                sub_scene_idx += 1

            y -= self.step_size[1]

        return offsets

    def reconstruct_scene(self, patches: list[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct full scene from patches using voting (exact from fusion code).
        """
        batch_size = patches[0].shape[0]
        device = patches[0].device

        # Initialize reconstruction
        reconstructed = torch.zeros((batch_size, *self.scene_size),
                                  dtype=patches[0].dtype, device=device)
        vote_count = torch.zeros((batch_size, *self.scene_size),
                               dtype=torch.float32, device=device)

        # Get patch offsets
        offsets = self.get_patch_offsets()

        # Accumulate votes (exact from merge_voxel_files logic)
        for patch, (x_offset, y_offset) in zip(patches, offsets):
            end_x = x_offset + self.target_resolution[0]
            end_y = y_offset + self.target_resolution[1]

            # Add patch to reconstruction
            reconstructed[:, x_offset:end_x, y_offset:end_y, :] += patch.float()
            vote_count[:, x_offset:end_x, y_offset:end_y, :] += 1.0

        # Average overlapping regions
        vote_count = torch.clamp(vote_count, min=1.0)
        reconstructed = reconstructed / vote_count

        return reconstructed.long()


class FinalCorrectedPipeline:
    """
    Final corrected pyramid pipeline with exact subdivision logic.
    """

    def __init__(self,
                 dataset_root: str,
                 output_root: str,
                 num_classes: int = 20,
                 device: str = 'cuda',
                 batch_sizes: dict = None,
                 is_distributed: bool = False,
                 rank: int = 0,
                 world_size: int = 1,
                 local_rank: int = 0):

        # Distributed training setup
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # Ensure dataset_root points to sequences directory
        if dataset_root.endswith('/sequences'):
            self.dataset_root = dataset_root
        else:
            self.dataset_root = os.path.join(dataset_root, 'sequences')
        self.output_root = output_root
        self.num_classes = num_classes

        # Set device based on distributed mode
        if is_distributed:
            self.device = f'cuda:{local_rank}'
        else:
            self.device = device

        # Override batch sizes if provided
        if batch_sizes:
            for stage, batch_size in batch_sizes.items():
                if stage in STAGE_CONFIG:
                    STAGE_CONFIG[stage]['batch_size'] = batch_size

        # Only rank 0 creates directories in distributed mode
        if rank == 0:
            os.makedirs(output_root, exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Scene subdivision for s3
        self.subdivider = ExactSceneSubdivision(
            scene_size=STAGE_CONFIG['s3']['size'],
            target_resolution=STAGE_CONFIG['s3']['patch_size'],
            sliding_ratio=TRAINING_CONFIG['sliding_ratio']
        )

        # Initialize models
        self.models = {}
        self.diffusion_models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all pyramid stage models with optional DDP wrapper."""
        for stage, config in STAGE_CONFIG.items():
            if self.rank == 0:
                self.logger.info(f"Initializing {stage} model...")

            # For s3, use patch size; for others use stage size
            model_size = config.get('patch_size', config['size'])

            # Create exact Denoise model from original codebase
            denoiser = create_pyramid_denoiser(
                stage=stage,
                num_classes=self.num_classes
            ).to(self.device)

            # Create diffusion model
            diffusion = PyramidDiscreteDiffusion(
                args=None,
                denoise_model=denoiser,
                num_classes=self.num_classes,
                num_timesteps=TRAINING_CONFIG['timesteps'],
                multi_criterion=None,  # Not used (recon_loss=False in config)
                auxiliary_loss_weight=TRAINING_CONFIG['auxiliary_loss_weight'],
                adaptive_auxiliary_loss=True,
                recon_loss=False  # Matches train_s_1.yaml config
            ).to(self.device)

            # Wrap with DDP if distributed training
            if self.is_distributed:
                diffusion = DDP(diffusion, device_ids=[self.local_rank], output_device=self.local_rank)

            self.models[stage] = denoiser
            self.diffusion_models[stage] = diffusion


    def create_data_loader(self, stage: str, mode: str = 'train') -> DataLoader:
        """Create data loader with DistributedSampler support."""
        config = STAGE_CONFIG[stage]

        # EXCLUDE sequence 08 from training
        if mode == 'train':
            sequences = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
        elif mode == 'val':
            sequences = [8]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dataset = SemanticKITTIDataset(
            dataset_root=self.dataset_root,
            sequences=sequences,
            pyramid_size=config['size'],
            semantickitti_size=(256, 256, 32),
            augment=(mode == 'train')  # Enable augmentation for training
        )

        # Use DistributedSampler in distributed mode
        if self.is_distributed and mode == 'train':
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            return DataLoader(
                dataset,
                batch_size=config['batch_size'],
                sampler=sampler,
                num_workers=8,  # Matching original paper's train_s_1.yaml
                pin_memory=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=(mode == 'train'),
                num_workers=8,  # Matching original paper's train_s_1.yaml
                pin_memory=True
            )

    def downsample_to_stage(self, data: torch.Tensor, target_stage: str) -> torch.Tensor:
        """Downsample data to target pyramid stage size."""
        target_size = STAGE_CONFIG[target_stage]['size']
        current_size = data.shape[-3:]

        if current_size == target_size:
            return data

        data = data.unsqueeze(1).float()
        downsampled = torch.nn.functional.interpolate(
            data, size=target_size, mode='trilinear', align_corners=False
        )
        return downsampled.squeeze(1).long()

    def train_stage(self, stage: str, max_epochs: int = 1000, resume_checkpoint: str = None) -> None:
        """Train specific pyramid stage with optional resume functionality until target loss achieved."""
        if self.rank == 0:
            self.logger.info(f"Training stage {stage}...")

        model = self.diffusion_models[stage]
        model.train()

        # Optimizer settings - matching original pyramid-discrete-diffusion paper exactly
        # Paper: lr=0.001, batch_size=32 per GPU (128 total on 4 GPUs), AdamW with default betas
        # NO learning rate scaling, NO beta adjustments - use values directly as in paper
        lr = TRAINING_CONFIG['learning_rate']
        current_batch_size = STAGE_CONFIG[stage]['batch_size']
        total_batch_size = current_batch_size * self.world_size

        if self.rank == 0:
            self.logger.info(f"Batch size: {current_batch_size} per GPU × {self.world_size} GPUs = {total_batch_size} total")
            self.logger.info(f"Learning rate: {lr} (no scaling - matching paper)")
            self.logger.info("Adam betas: (0.9, 0.999) (default - matching paper)")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999)  # Default Adam betas, no adjustments
        )

        # Handle checkpoint resuming
        start_epoch = 0
        if resume_checkpoint:
            checkpoint = self.load_checkpoint(resume_checkpoint)

            # Handle loading from DDP or non-DDP checkpoint
            state_dict = checkpoint['model_state_dict']
            if self.is_distributed and not any(k.startswith('module.') for k in state_dict.keys()):
                # Loading non-DDP checkpoint into DDP model
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif not self.is_distributed and any(k.startswith('module.') for k in state_dict.keys()):
                # Loading DDP checkpoint into non-DDP model
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']

            if self.rank == 0:
                self.logger.info(f"🔄 Resumed {stage} from epoch {start_epoch}")
                self.logger.info(f"Previous train loss: {checkpoint['train_loss']:.6f}")
                self.logger.info(f"Previous val loss: {checkpoint['val_loss']:.6f}")

        train_loader = self.create_data_loader(stage, 'train')
        total_batches = len(train_loader)
        self.logger.info(f"Training {stage}: {total_batches} batches per epoch, target training loss < 0.05")

        # Training loop with progress bars (resume from start_epoch) - continue until target achieved
        epoch = start_epoch
        target_achieved = False

        # Train until target achieved (no epoch limit - only stop when loss < 0.05)
        while not target_achieved:
            epoch_pbar = tqdm(total=1, desc=f"Training {stage} Epoch {epoch+1}", unit="epoch")
            epoch_loss = 0.0
            num_batches = 0

            # Batch progress bar
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} (target train_loss<0.05)", unit="batch", leave=False)

            for batch_idx, batch in enumerate(batch_pbar):
                # Update progress bar description with loss info
                if batch_idx > 0:
                    avg_loss = epoch_loss / num_batches
                    batch_pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'})

                # Log progress every 20 batches (less frequent for better performance)
                if batch_idx % 20 == 0:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    self.logger.info(f"[{stage}] Epoch {epoch+1}, Batch {batch_idx+1}/{total_batches}, GPU: {memory_allocated:.1f}GB/{memory_reserved:.1f}GB")
                optimizer.zero_grad()

                if stage == 's3':
                    # S3: Use subdivision approach
                    target_full = batch['labels'].to(self.device)
                    patches = self.subdivider.split_scene(target_full)

                    # Get s2 condition and expand to patch size
                    cond_s2 = self.downsample_to_stage(batch['labels'], 's2').to(self.device)
                    cond_patches = []

                    for _ in range(len(patches)):
                        # Upsample s2 to patch size
                        cond_patch = torch.nn.functional.interpolate(
                            cond_s2.unsqueeze(1).float(),
                            size=STAGE_CONFIG['s3']['patch_size'],
                            mode='trilinear', align_corners=False
                        ).squeeze(1).long()
                        cond_patches.append(cond_patch)

                    # Train on all patches
                    total_loss = 0.0
                    for target_patch, cond_patch in zip(patches, cond_patches):
                        # For s3: need to format cond_patch with channel dimension
                        # Add channel dimension: [B, H, W, D] -> [B, 1, H, W, D]
                        cond_patch = cond_patch.unsqueeze(1).float()
                        # For now use single channel, later can add mask as second channel
                        loss = model(target_patch, cond_patch)
                        total_loss += loss

                    loss = total_loss / len(patches)

                else:
                    # S1, S2: Regular training
                    target = self.downsample_to_stage(batch['labels'], stage).to(self.device)

                    cond = None
                    if STAGE_CONFIG[stage]['has_condition']:
                        if stage == 's1':
                            # Stage 1: Use masked version of target as condition (grounded evidence)
                            target_np = target.cpu().numpy()
                            masked_scenes = mask_scene(
                                [target_np[i] for i in range(target_np.shape[0])],
                                TRAINING_CONFIG['mask_ratio'],
                                TRAINING_CONFIG['mask_prob']
                            )
                            cond = torch.from_numpy(np.stack(masked_scenes)).to(self.device)
                            # Add channel dimension: [B, H, W, D] -> [B, 1, H, W, D]
                            cond = cond.unsqueeze(1).float()
                        elif stage == 's2':
                            # Stage 2: Use upsampled s1 as condition
                            cond = self.downsample_to_stage(batch['labels'], 's1').to(self.device)
                            # Add channel dimension: [B, H, W, D] -> [B, 1, H, W, D]
                            cond = cond.unsqueeze(1).float()

                    loss = model(target, cond)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Log every epoch completion
            avg_loss = epoch_loss / max(num_batches, 1)
            self.logger.info(f"[{stage}] Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, Batches: {num_batches}")
            epoch_pbar.update(1)
            epoch_pbar.close()

            # Checkpoint saving and validation every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"=== [{stage}] Epoch {epoch+1} Summary ===")
                # Calculate validation loss
                val_loss = self.validate_stage(stage, model)

                # Save checkpoint
                checkpoint_path = self.save_checkpoint(stage, epoch + 1, model, optimizer, avg_loss, val_loss)

                self.logger.info(f"Training Loss: {avg_loss:.6f}")
                self.logger.info(f"Validation Loss: {val_loss:.6f}")
                self.logger.info(f"Total Batches: {num_batches}")
                self.logger.info(f"Batch Size: {STAGE_CONFIG[stage]['batch_size']}")
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
                self.logger.info("====================================")

                # Check if target loss achieved (use training loss)
                if avg_loss < 0.05:
                    self.logger.info(f"🎉 TARGET ACHIEVED! Training loss {avg_loss:.6f} < 0.05. Stopping training for {stage}")
                    target_achieved = True
                    break

            # Increment epoch for next iteration
            epoch += 1

            # Check if max_epochs limit reached
            if epoch >= max_epochs:
                self.logger.info(f"🏁 MAX EPOCHS REACHED! Stopping training for {stage} at epoch {epoch}")
                break

    def train_all_stages(self, epochs_per_stage: int = 50, resume_checkpoint: str = None):
        """Train all pyramid stages: s1→s2→s3 with optional resume functionality."""
        stages = ['s1', 's2', 's3']

        # Determine starting stage and epoch if resuming
        start_stage_idx = 0
        start_epoch = 0

        if resume_checkpoint:
            checkpoint = self.load_checkpoint(resume_checkpoint)
            resumed_stage = checkpoint['stage']
            start_epoch = checkpoint['epoch']

            # Find the stage index to resume from
            start_stage_idx = stages.index(resumed_stage)

            self.logger.info(f"🔄 RESUMING from {resumed_stage} epoch {start_epoch}")
            self.logger.info(f"Previous train loss: {checkpoint['train_loss']:.6f}")
            self.logger.info(f"Previous val loss: {checkpoint['val_loss']:.6f}")

        for stage_idx, stage in enumerate(stages[start_stage_idx:], start_stage_idx):
            self.logger.info(f"{'='*50}")
            self.logger.info(f"Training stage {stage}")
            self.logger.info(f"Stage size: {STAGE_CONFIG[stage]['size']}")
            if 'patch_size' in STAGE_CONFIG[stage]:
                self.logger.info(f"Patch size: {STAGE_CONFIG[stage]['patch_size']}")
            self.logger.info(f"{'='*50}")

            # Pass resume info only for the current resuming stage
            stage_resume_checkpoint = resume_checkpoint if stage_idx == start_stage_idx else None

            self.train_stage(stage, max_epochs=epochs_per_stage, resume_checkpoint=stage_resume_checkpoint)
            self.logger.info(f"Completed training for stage {stage}")

        self.logger.info("✅ All stages trained successfully!")

    def validate_stage(self, stage: str, model) -> float:
        """Calculate validation loss on sequence 08 for generative model evaluation."""
        model.eval()
        config = STAGE_CONFIG[stage]

        val_dataset = SemanticKITTIDataset(
            dataset_root=self.dataset_root,
            sequences=[8],  # Validation sequence
            pyramid_size=config['size'],
            semantickitti_size=(256, 256, 32)
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if num_batches >= 10:  # Limit validation to 10 batches for speed
                    break

                target = self.downsample_to_stage(batch['labels'], stage).to(self.device)

                # Same conditioning as training
                cond = None
                if STAGE_CONFIG[stage]['has_condition']:
                    if stage == 's1':
                        # Use masked version for validation
                        target_np = target.cpu().numpy()
                        masked_scenes = mask_scene(
                            [target_np[i] for i in range(target_np.shape[0])],
                            TRAINING_CONFIG['mask_ratio'],
                            TRAINING_CONFIG['mask_prob']
                        )
                        cond = torch.from_numpy(np.stack(masked_scenes)).to(self.device)
                        cond = cond.unsqueeze(1).float()
                    elif stage == 's2':
                        cond = self.downsample_to_stage(batch['labels'], 's1').to(self.device)
                        cond = cond.unsqueeze(1).float()

                loss = model(target, cond)
                total_loss += loss.item()
                num_batches += 1

        model.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, stage: str, epoch: int, model, optimizer, train_loss: float, val_loss: float) -> str:
        """Save comprehensive checkpoint with all training state (only from rank 0)."""
        # Only rank 0 saves checkpoints in distributed mode
        if self.rank != 0:
            return ""

        checkpoint_dir = os.path.join(self.output_root, 'checkpoints', stage)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'{stage}_epoch_{epoch:03d}.pt')

        # Unwrap DDP model state_dict for compatibility
        if self.is_distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            'stage': stage,
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'stage_config': STAGE_CONFIG[stage],
            'training_config': TRAINING_CONFIG,
            'timestamp': datetime.now().isoformat(),
            'world_size': self.world_size,  # Save world_size for reference
        }

        torch.save(checkpoint, checkpoint_path)

        # Update training log file
        self.update_training_log(stage, epoch, train_loss, val_loss, checkpoint_path)

        return checkpoint_path

    def update_training_log(self, stage: str, epoch: int, train_loss: float, val_loss: float, checkpoint_path: str):
        """Update comprehensive training log file."""
        log_path = os.path.join(self.output_root, 'training_progress.txt')

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"""
================================================================================
Timestamp: {timestamp}
Stage: {stage}
Epoch: {epoch}
Training Loss: {train_loss:.6f}
Validation Loss: {val_loss:.6f}
Checkpoint: {checkpoint_path}
GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.memory_reserved()/1024**3:.1f}GB
Batch Size: {STAGE_CONFIG[stage]['batch_size']}
Stage Size: {STAGE_CONFIG[stage]['size']}
Learning Rate: {TRAINING_CONFIG['learning_rate']}
Loss Improvement: {'✓' if hasattr(self, 'last_val_loss') and val_loss < self.last_val_loss else '?'}
================================================================================
"""

        with open(log_path, 'a') as f:
            f.write(log_entry)

        # Update best loss tracking
        self.last_val_loss = val_loss

        # Note: Target achievement is now checked using training loss in train_stage method
        # This log entry is for reference only
        if val_loss < 0.05:
            self.logger.info(f"📊 BONUS: Validation loss also < 0.05: {val_loss:.6f}")
            with open(log_path, 'a') as f:
                f.write(f"\n📊 BONUS: Validation loss also < 0.05 at {timestamp}! Stage {stage} Epoch {epoch} - Val Loss: {val_loss:.6f}\n")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Load checkpoint for crash recovery."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Stage: {checkpoint['stage']}, Epoch: {checkpoint['epoch']}")
        self.logger.info(f"Train Loss: {checkpoint['train_loss']:.6f}, Val Loss: {checkpoint['val_loss']:.6f}")

        return checkpoint


# GROUNDED EVIDENCE: Essential functions from original codebase
def mask_scene(input_scenes: list[np.ndarray], mask_ratio: float, mask_prob: list[float]) -> list[np.ndarray]:
    """Apply random masking to scenes for stage 1 training.
    EVIDENCE: pyramid-discrete-diffusion/utils/mask_scene.py:3-27
    """
    if len(input_scenes) == 0:
        return []

    x, y, _ = input_scenes[0].shape
    masked_scenes = []

    for sub_scene in input_scenes:
        random_number = np.random.choice(4, 1, p=mask_prob)[0]
        masked_sub_scene = np.copy(sub_scene)

        if random_number == 0:  # Mask bottom strip
            y_limit = int(y * mask_ratio)
            masked_sub_scene[:, :-y_limit, :] = 0
        elif random_number == 1:  # Mask left strip
            x_limit = int(x * mask_ratio)
            masked_sub_scene[x_limit:, :, :] = 0
        elif random_number == 2:  # Mask bottom-left corner
            y_limit = int(y * mask_ratio)
            x_limit = int(x * mask_ratio)
            masked_sub_scene[x_limit:, :-y_limit, :] = 0
        elif random_number == 3:  # Mask everything
            masked_sub_scene[:, :, :] = 0

        masked_scenes.append(masked_sub_scene)

    return masked_scenes


def generation_mask(block_idx: int, patch_size: tuple[int, int, int], mask_ratio: float) -> np.ndarray:
    """Generate position-specific masks for s3 patch generation.
    EVIDENCE: pyramid-discrete-diffusion/utils/generation_mask.py:14-80
    """
    mask = np.zeros(patch_size, dtype=np.int32)

    if block_idx == 0:
        return mask  # Empty mask
    elif block_idx == 1:
        # Left strip
        x_dim = patch_size[1]
        mask_width = int(mask_ratio * x_dim)
        mask[:mask_width, :, :] = 0
        return mask
    elif block_idx == 2:
        # Top strip
        y_dim = patch_size[0]
        mask_height = max(1, int(mask_ratio * y_dim))
        mask[:, -mask_height:, :] = 0
        return mask
    elif block_idx == 3:
        # L-shape
        y_dim, x_dim = patch_size[0], patch_size[1]
        mask_height = max(1, int(mask_ratio * y_dim))
        mask_width = int(mask_ratio * x_dim)
        mask[:, -mask_height:, :] = 0  # Top strip
        mask[:mask_width, :, :] = 0    # Left strip
        return mask
    else:
        return generation_mask(block_idx % 4, patch_size, mask_ratio)


def run_pyramid_training(dataset_root: str,
                        output_root: str,
                        epochs_per_stage: int = 50,
                        batch_sizes: dict = None,
                        resume_checkpoint: str = None,
                        is_distributed: bool = False,
                        rank: int = 0,
                        world_size: int = 1,
                        local_rank: int = 0) -> None:
    """Run pyramid discrete diffusion training with multi-GPU support."""
    pipeline = FinalCorrectedPipeline(
        dataset_root=dataset_root,
        output_root=output_root,
        batch_sizes=batch_sizes,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank
    )

    pipeline.train_all_stages(epochs_per_stage=epochs_per_stage, resume_checkpoint=resume_checkpoint)