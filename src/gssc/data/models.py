"""
Adapted pyramid diffusion model for SemanticKITTI data augmentation.
Modified to work with 256x256x32 voxel grids and SemanticKITTI label space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# from einops import rearrange  # Not needed for current implementation


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride)


class AsymmetricResidualBlock(nn.Module):
    """3D residual block with time embedding for diffusion."""

    def __init__(self, in_filters, out_filters, time_filters=128):
        super(AsymmetricResidualBlock, self).__init__()

        # Adaptive group normalization
        groups = min(32, in_filters) if in_filters >= 32 else min(16, in_filters)
        out_groups = min(32, out_filters) if out_filters >= 32 else min(16, out_filters)

        self.norm_in = nn.GroupNorm(groups, in_filters)
        self.norm1 = nn.GroupNorm(out_groups, out_filters)
        self.norm2 = nn.GroupNorm(out_groups, out_filters)

        self.conv1 = conv3x3x3(in_filters, out_filters)
        self.conv2 = conv3x3x3(out_filters, out_filters)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_filters, out_filters)
        )

        # Skip connection
        self.skip = conv1x1x1(in_filters, out_filters) if in_filters != out_filters else nn.Identity()

    def forward(self, x, time_emb):
        skip = self.skip(x)

        x = self.norm_in(x)
        x = F.relu(x)
        x = self.conv1(x)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb[:, :, None, None, None]  # Broadcast to spatial dims
        x = x + time_emb

        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x + skip


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class KittiPyramidUNet(nn.Module):
    """
    3D U-Net for SemanticKITTI data augmentation using pyramid diffusion.
    Adapted for 256x256x32 voxel grids.
    """

    def __init__(self, num_classes=20, base_filters=32, time_dim=128):
        super(KittiPyramidUNet, self).__init__()

        self.num_classes = num_classes
        self.time_dim = time_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Input projection
        self.input_conv = conv3x3x3(num_classes, base_filters)

        # Encoder
        self.enc1 = AsymmetricResidualBlock(base_filters, base_filters * 2, time_dim)
        self.enc2 = AsymmetricResidualBlock(base_filters * 2, base_filters * 4, time_dim)
        self.enc3 = AsymmetricResidualBlock(base_filters * 4, base_filters * 8, time_dim)

        # Bottleneck
        self.bottleneck = AsymmetricResidualBlock(base_filters * 8, base_filters * 8, time_dim)

        # Decoder
        self.dec3 = AsymmetricResidualBlock(base_filters * 16, base_filters * 4, time_dim)
        self.dec2 = AsymmetricResidualBlock(base_filters * 8, base_filters * 2, time_dim)
        self.dec1 = AsymmetricResidualBlock(base_filters * 4, base_filters, time_dim)

        # Output projection
        self.output_conv = conv1x1x1(base_filters, num_classes)

        # Downsampling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, timesteps):
        """
        Args:
            x: Input tensor [B, C, H, W, D] where C=num_classes (one-hot)
            timesteps: Diffusion timesteps [B]

        Returns:
            Denoised output [B, C, H, W, D]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Input projection
        x1 = self.input_conv(x)

        # Encoder path
        x2 = self.pool(self.enc1(x1, time_emb))
        x3 = self.pool(self.enc2(x2, time_emb))
        x4 = self.pool(self.enc3(x3, time_emb))

        # Bottleneck
        x = self.bottleneck(x4, time_emb)

        # Decoder path with skip connections
        x = self.upsample(x)
        if x.shape[2:] != x3.shape[2:]:
            # Handle size mismatch due to pooling
            x = F.interpolate(x, size=x3.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x, time_emb)

        x = self.upsample(x)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, time_emb)

        x = self.upsample(x)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, time_emb)

        # Output projection
        x = self.output_conv(x)

        return x


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule for diffusion."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    return np.sqrt(alphas)


class KittiDiffusionModel(nn.Module):
    """Complete diffusion model for SemanticKITTI data augmentation."""

    def __init__(self, num_classes=20, num_timesteps=1000, base_filters=32):
        super().__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # U-Net denoising model
        self.denoise_model = KittiPyramidUNet(num_classes, base_filters)

        # Diffusion schedule
        alphas = cosine_beta_schedule(num_timesteps)
        alphas = torch.tensor(alphas.astype('float32'))

        log_alpha = torch.log(alphas)
        log_cumprod_alpha = torch.cumsum(log_alpha, dim=0)

        self.register_buffer('log_alpha', log_alpha)
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha)
        self.register_buffer('log_1_min_alpha', torch.log(1 - alphas + 1e-40))
        self.register_buffer('log_1_min_cumprod_alpha', torch.log(1 - torch.exp(log_cumprod_alpha) + 1e-40))

    def q_posterior(self, log_x_start, log_x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        # This implements the discrete diffusion posterior
        # Simplified version - full implementation would handle multinomial diffusion properly
        return log_x_start  # Placeholder

    def p_sample(self, log_x, t, condition=None):
        """Sample x_{t-1} from x_t."""
        # Predict x_0
        log_x_recon = self.denoise_model(log_x, t)

        if t[0] == 0:
            return log_x_recon
        else:
            # Sample from posterior
            return self.q_posterior(log_x_recon, log_x, t)

    def forward(self, x, t):
        """Forward pass for training."""
        return self.denoise_model(x, t)