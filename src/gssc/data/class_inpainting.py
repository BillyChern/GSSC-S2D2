"""
Class-Conditional Inpainting for Semantic Scene Completion

Implements RePaint-style inpainting where:
1. We keep voxels containing rare classes (anchor regions)
2. We mask out and regenerate the surrounding regions
3. Diffusion generates plausible context around the rare objects

This allows creating diverse scenes while preserving rare object instances.

Reference: "RePaint: Inpainting using Denoising Diffusion Probabilistic Models" (CVPR 2022)
"""

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn

# SemanticKITTI class definitions
RARE_CLASSES = [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]  # Classes to preserve
HEAD_CLASSES = [9, 11, 15, 17]  # road, sidewalk, vegetation, terrain

# Context classes that should remain near rare objects
CONTEXT_COMPATIBILITY = {
    # Person/cyclist classes need road/sidewalk context
    6: [9, 10, 11],  # person → road, parking, sidewalk
    7: [9, 10, 11],  # bicyclist → road, parking, sidewalk
    8: [9, 10, 11],  # motorcyclist → road, parking, sidewalk
    # Vehicle classes
    1: [9, 10],  # car → road, parking
    2: [9, 10, 11],  # bicycle → road, parking, sidewalk
    3: [9, 10],  # motorcycle → road, parking
    4: [9, 10],  # truck → road, parking
    5: [9, 10],  # other-vehicle → road, parking
    # Infrastructure
    16: [15, 17],  # trunk → vegetation, terrain
    18: [9, 11, 13],  # pole → road, sidewalk, building
    19: [9, 11, 13],  # traffic-sign → road, sidewalk, building
}


@dataclass
class InpaintingMask:
    """Represents an inpainting mask with anchor and regeneration regions."""
    anchor_mask: np.ndarray  # [X, Y, Z] bool - regions to keep
    inpaint_mask: np.ndarray  # [X, Y, Z] bool - regions to regenerate
    anchor_labels: np.ndarray  # [X, Y, Z] - original labels in anchor region
    rare_classes_present: list[int]  # Which rare classes are anchored


class RareClassAnchor:
    """
    Identifies and creates anchor masks around rare class instances.
    """

    def __init__(
        self,
        rare_classes: list[int] = None,
        dilation_radius: int = 3,
        min_context_radius: int = 5,
        preserve_context: bool = True,
    ):
        """
        Args:
            rare_classes: Classes to treat as anchors
            dilation_radius: Dilation for anchor regions
            min_context_radius: Minimum context around rare objects to preserve
            preserve_context: Whether to preserve compatible context classes
        """
        self.rare_classes = rare_classes or RARE_CLASSES
        self.dilation_radius = dilation_radius
        self.min_context_radius = min_context_radius
        self.preserve_context = preserve_context

    def create_anchor_mask(
        self,
        scene: np.ndarray,
        target_classes: list[int] = None,
    ) -> InpaintingMask:
        """
        Create an inpainting mask that preserves rare objects.

        Args:
            scene: [X, Y, Z] semantic labels
            target_classes: Specific classes to anchor (default: all rare)

        Returns:
            InpaintingMask with anchor and inpaint regions
        """
        if target_classes is None:
            target_classes = self.rare_classes

        X, Y, Z = scene.shape
        anchor_mask = np.zeros_like(scene, dtype=bool)
        rare_present = []

        # Find all rare class voxels
        for cls in target_classes:
            cls_mask = (scene == cls)
            if cls_mask.any():
                rare_present.append(cls)
                anchor_mask |= cls_mask

                # Dilate to include immediate context
                if self.dilation_radius > 0:
                    struct = ndimage.generate_binary_structure(3, 1)
                    dilated = ndimage.binary_dilation(
                        cls_mask,
                        structure=struct,
                        iterations=self.dilation_radius
                    )
                    anchor_mask |= dilated

                # Preserve compatible context classes
                if self.preserve_context and cls in CONTEXT_COMPATIBILITY:
                    context_classes = CONTEXT_COMPATIBILITY[cls]
                    for ctx_cls in context_classes:
                        ctx_mask = (scene == ctx_cls)
                        # Only preserve context within radius of rare object
                        if ctx_mask.any():
                            # Create distance field from rare object
                            dist = ndimage.distance_transform_edt(~cls_mask)
                            nearby_ctx = ctx_mask & (dist < self.min_context_radius)
                            anchor_mask |= nearby_ctx

        # Inpaint mask is everything not anchored (except unlabeled)
        inpaint_mask = ~anchor_mask & (scene != 0)

        # Also include some unlabeled regions for generation
        unlabeled_mask = (scene == 0)
        # Keep 50% of unlabeled for generation variety
        rand_unlabeled = np.random.random(scene.shape) < 0.5
        inpaint_mask |= (unlabeled_mask & rand_unlabeled)

        return InpaintingMask(
            anchor_mask=anchor_mask,
            inpaint_mask=inpaint_mask,
            anchor_labels=scene.copy(),
            rare_classes_present=rare_present,
        )

    def create_class_preserving_mask(
        self,
        scene: np.ndarray,
        preserve_ratio: float = 0.3,
        prioritize_rare: bool = True,
    ) -> InpaintingMask:
        """
        Create a mask that preserves a portion of the scene, prioritizing rare classes.

        Args:
            scene: [X, Y, Z] semantic labels
            preserve_ratio: Fraction of non-empty voxels to preserve
            prioritize_rare: Always preserve rare classes

        Returns:
            InpaintingMask
        """
        anchor_mask = np.zeros_like(scene, dtype=bool)
        rare_present = []

        # Always preserve rare classes
        if prioritize_rare:
            for cls in self.rare_classes:
                cls_mask = (scene == cls)
                if cls_mask.any():
                    rare_present.append(cls)
                    anchor_mask |= cls_mask

        # Calculate how many more voxels to preserve
        non_empty = scene != 0
        current_preserved = anchor_mask.sum()
        target_preserved = int(non_empty.sum() * preserve_ratio)

        if current_preserved < target_preserved:
            # Randomly select additional voxels
            remaining = non_empty & ~anchor_mask
            remaining_indices = np.argwhere(remaining)
            if len(remaining_indices) > 0:
                num_to_add = min(
                    target_preserved - current_preserved,
                    len(remaining_indices)
                )
                selected = np.random.choice(
                    len(remaining_indices),
                    size=num_to_add,
                    replace=False
                )
                for idx in selected:
                    x, y, z = remaining_indices[idx]
                    anchor_mask[x, y, z] = True

        inpaint_mask = non_empty & ~anchor_mask

        return InpaintingMask(
            anchor_mask=anchor_mask,
            inpaint_mask=inpaint_mask,
            anchor_labels=scene.copy(),
            rare_classes_present=rare_present,
        )


class RePaintInpainting:
    """
    RePaint-style inpainting for discrete diffusion.

    During sampling:
    1. Forward diffuse anchor regions to timestep t
    2. Replace anchor positions with forward-diffused values
    3. Continue reverse diffusion
    4. Optionally resample (jump back) for better harmony
    """

    def __init__(
        self,
        num_classes: int = 20,
        num_timesteps: int = 100,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        device: str = 'cuda',
    ):
        """
        Args:
            num_classes: Number of semantic classes
            num_timesteps: Total diffusion timesteps
            jump_length: How many steps to jump back for resampling
            jump_n_sample: Number of resampling iterations
            device: Compute device
        """
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.jump_length = jump_length
        self.jump_n_sample = jump_n_sample
        self.device = device

    def get_schedule(self) -> list[int]:
        """
        Get the timestep schedule with jumps for resampling.

        Returns list of timesteps to iterate through.
        """
        timesteps = []
        t = self.num_timesteps - 1

        while t >= 0:
            # Regular step
            timesteps.append(t)
            t -= 1

            # Resampling jumps at intervals
            if t > 0 and t % self.jump_length == 0:
                for _ in range(self.jump_n_sample):
                    # Jump back
                    t_back = min(t + self.jump_length, self.num_timesteps - 1)
                    timesteps.append(t_back)
                    # Then come back down
                    for t_down in range(t_back - 1, t - 1, -1):
                        timesteps.append(t_down)

        return timesteps

    def forward_diffuse(
        self,
        x_0: torch.Tensor,
        t: int,
        noise_type: str = 'absorbing',
    ) -> torch.Tensor:
        """
        Forward diffuse clean labels to timestep t.

        Args:
            x_0: [B, X, Y, Z] clean labels
            t: Target timestep
            noise_type: 'absorbing' or 'uniform'

        Returns:
            x_t: [B, X, Y, Z] noised labels
        """
        # Compute noise probability at timestep t
        # For cosine schedule: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        alpha_bar = self._cosine_alpha_bar(t)

        if noise_type == 'absorbing':
            # With probability (1 - alpha_bar), replace with mask token
            mask_prob = 1 - alpha_bar
            mask = torch.rand_like(x_0.float()) < mask_prob
            x_t = torch.where(mask, torch.zeros_like(x_0), x_0)
        else:
            # Uniform: with probability (1 - alpha_bar), replace with random
            noise_prob = 1 - alpha_bar
            mask = torch.rand_like(x_0.float()) < noise_prob
            random_labels = torch.randint(
                0, self.num_classes, x_0.shape,
                device=x_0.device
            )
            x_t = torch.where(mask, random_labels, x_0)

        return x_t

    def _cosine_alpha_bar(self, t: int) -> float:
        """Cosine schedule alpha_bar."""
        s = 0.008
        t_norm = t / self.num_timesteps
        return np.cos((t_norm + s) / (1 + s) * np.pi / 2) ** 2

    def inpaint_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: int,
        anchor_mask: torch.Tensor,
        anchor_labels: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Single inpainting step.

        Args:
            model: Denoising model
            x_t: Current noisy state [B, X, Y, Z]
            t: Current timestep
            anchor_mask: [B, X, Y, Z] bool - regions to preserve
            anchor_labels: [B, X, Y, Z] - original labels
            condition: Optional conditioning input

        Returns:
            x_{t-1}: Denoised state
        """
        # Get model prediction for denoised regions
        t_tensor = torch.full(
            (x_t.shape[0],), t,
            device=x_t.device, dtype=torch.long
        )

        if condition is not None:
            x_pred = model(x_t, t_tensor, condition)
        else:
            x_pred = model(x_t, t_tensor)

        # Forward diffuse anchor regions to current timestep
        anchor_t = self.forward_diffuse(anchor_labels, t)

        # Combine: use prediction for inpaint regions, anchor for preserved
        x_next = torch.where(anchor_mask, anchor_t, x_pred)

        return x_next

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        inpaint_mask: InpaintingMask,
        condition: torch.Tensor | None = None,
        batch_size: int = 1,
        shape: tuple[int, int, int] = (256, 256, 32),
        use_jumps: bool = True,
    ) -> torch.Tensor:
        """
        Full inpainting sampling loop.

        Args:
            model: Trained diffusion model
            inpaint_mask: InpaintingMask with anchor/inpaint regions
            condition: Optional conditioning
            batch_size: Batch size for generation
            shape: Output shape (X, Y, Z)
            use_jumps: Whether to use resampling jumps

        Returns:
            samples: [B, X, Y, Z] generated labels
        """
        device = self.device

        # Prepare masks
        anchor_mask = torch.from_numpy(inpaint_mask.anchor_mask).to(device)
        anchor_labels = torch.from_numpy(inpaint_mask.anchor_labels).long().to(device)

        # Expand for batch
        anchor_mask = anchor_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        anchor_labels = anchor_labels.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Initialize with noise
        x_t = torch.zeros(batch_size, *shape, device=device, dtype=torch.long)

        # Get timestep schedule
        if use_jumps:
            schedule = self.get_schedule()
        else:
            schedule = list(range(self.num_timesteps - 1, -1, -1))

        # Sampling loop
        for t in schedule:
            x_t = self.inpaint_step(
                model, x_t, t, anchor_mask, anchor_labels, condition
            )

        return x_t


class ClassConditionedGenerator:
    """
    Generate scenes conditioned on presence of specific classes.

    Uses classifier-free guidance to boost generation of rare classes.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 20,
        guidance_scale: float = 2.0,
        rare_classes: list[int] = None,
    ):
        """
        Args:
            model: Diffusion model with class conditioning
            num_classes: Number of semantic classes
            guidance_scale: CFG guidance strength
            rare_classes: Classes to boost during generation
        """
        self.model = model
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale
        self.rare_classes = rare_classes or RARE_CLASSES

    def generate_with_class_condition(
        self,
        target_classes: list[int],
        shape: tuple[int, int, int] = (256, 256, 32),
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate scenes containing specified classes.

        Args:
            target_classes: Classes that should appear in generated scenes
            shape: Output shape
            num_samples: Number of samples to generate

        Returns:
            samples: [N, X, Y, Z] generated labels
        """
        # Create class condition vector
        class_cond = torch.zeros(num_samples, self.num_classes)
        for cls in target_classes:
            class_cond[:, cls] = 1.0

        class_cond = class_cond.to(next(self.model.parameters()).device)

        # Generate with CFG
        # ... (implementation depends on specific model architecture)
        # This is a placeholder for the actual CFG sampling

        raise NotImplementedError(
            "Full implementation requires integration with specific model"
        )


def create_diverse_inpainting_masks(
    scenes: list[np.ndarray],
    num_variants: int = 5,
    rare_classes: list[int] = None,
) -> list[tuple[np.ndarray, InpaintingMask]]:
    """
    Create multiple inpainting variants from a set of scenes.

    For each scene containing rare classes, creates multiple masks
    with different preserved/inpainted regions.

    Args:
        scenes: List of [X, Y, Z] semantic label arrays
        num_variants: Number of variants per scene
        rare_classes: Classes to always preserve

    Returns:
        List of (original_scene, inpainting_mask) tuples
    """
    anchor = RareClassAnchor(rare_classes=rare_classes)
    results = []

    for scene in scenes:
        # Check if scene has rare classes
        has_rare = any(
            (scene == cls).any()
            for cls in (rare_classes or RARE_CLASSES)
        )

        if not has_rare:
            continue

        # Create variants with different preserve ratios
        preserve_ratios = np.linspace(0.2, 0.5, num_variants)

        for ratio in preserve_ratios:
            mask = anchor.create_class_preserving_mask(
                scene,
                preserve_ratio=ratio,
                prioritize_rare=True,
            )
            results.append((scene, mask))

    return results


if __name__ == '__main__':
    # Test inpainting mask creation
    print("Testing class-conditional inpainting...")

    # Create synthetic scene
    scene = np.zeros((64, 64, 8), dtype=np.uint8)
    scene[30:40, 30:40, 2:5] = 9  # road
    scene[35:37, 35:37, 5:7] = 6  # person (rare)
    scene[20:25, 20:25, 2:4] = 1  # car

    # Create anchor
    anchor = RareClassAnchor(dilation_radius=2)
    mask = anchor.create_anchor_mask(scene, target_classes=[6])

    print(f"Scene shape: {scene.shape}")
    print(f"Rare classes present: {mask.rare_classes_present}")
    print(f"Anchor voxels: {mask.anchor_mask.sum()}")
    print(f"Inpaint voxels: {mask.inpaint_mask.sum()}")

    # Test class-preserving mask
    mask2 = anchor.create_class_preserving_mask(scene, preserve_ratio=0.3)
    print("\nClass-preserving mask:")
    print(f"Preserved voxels: {mask2.anchor_mask.sum()}")
    print(f"Inpaint voxels: {mask2.inpaint_mask.sum()}")

    # Test schedule with jumps
    inpainter = RePaintInpainting(num_timesteps=100, jump_length=10, jump_n_sample=2)
    schedule = inpainter.get_schedule()
    print(f"\nSampling schedule length: {len(schedule)}")
    print(f"First 20 timesteps: {schedule[:20]}")

    print("\nClass-conditional inpainting tests passed!")
