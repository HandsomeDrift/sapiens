"""Skeleton Diffusion Model — Diffusion-Driven Developmental Prior (DDP).

A lightweight DDPM that learns the distribution of anatomically valid
infant poses. Operates on normalized skeleton coordinates (K×2), not images.

Training: standalone script using only keypoint annotations (no images needed).
Inference: frozen, provides SDS gradient signal during Hyper-Sapiens training.

This is analogous to a Score-Based Generative Model — it learns the score
function ∇_x log p(x) of the infant pose distribution, which is then used
via Score Distillation Sampling (SDS) to regularize pose predictions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) timestep indices in [0, T).
        Returns:
            emb: (B, dim) time embeddings.
        """
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """MLP residual block with time conditioning."""

    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.fc1(h)
        h = h + self.time_proj(t_emb)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.fc2(h)
        return x + h


class SkeletonDDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model for skeleton keypoints.

    Learns to denoise corrupted skeleton coordinates, thereby learning
    the score function of the infant pose distribution.

    Architecture: flatten(K×2) → Linear → ResidualBlock × N → Linear → K×2

    Args:
        num_keypoints: Number of keypoints (e.g., 17 for COCO).
        hidden_dim: Hidden dimension of MLP blocks.
        num_blocks: Number of residual blocks.
        time_dim: Dimension of time embedding.
        num_timesteps: Total diffusion timesteps T.
        beta_start: Starting noise schedule value.
        beta_end: Ending noise schedule value.
    """

    def __init__(self,
                 num_keypoints: int = 17,
                 hidden_dim: int = 256,
                 num_blocks: int = 4,
                 time_dim: int = 128,
                 num_timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.input_dim = num_keypoints * 2
        self.num_timesteps = num_timesteps

        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))

        # Network
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, self.input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise ε given noisy skeleton x_t and timestep t.

        Args:
            x: (B, K*2) noisy skeleton coordinates (flattened).
            t: (B,) timestep indices.
        Returns:
            eps_pred: (B, K*2) predicted noise.
        """
        t_emb = self.time_mlp(self.time_embed(t))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.output_proj(h)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: add noise to clean skeleton.

        Args:
            x_0: (B, K*2) clean skeleton.
            t: (B,) timestep.
            noise: (B, K*2) optional pre-sampled noise.
        Returns:
            x_t: (B, K*2) noisy skeleton.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute DDPM training loss (simple MSE on predicted noise).

        Args:
            x_0: (B, K*2) clean skeleton coordinates.
        Returns:
            loss: scalar MSE loss.
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.forward(x_t, t)
        return F.mse_loss(eps_pred, noise)

    @staticmethod
    def normalize_skeleton(coords: torch.Tensor) -> torch.Tensor:
        """Normalize skeleton: center on hip midpoint, scale to unit."""
        # coords: (B, K, 2) — assume COCO17 where hips are indices 11, 12
        hip_center = (coords[:, 11] + coords[:, 12]) / 2.0  # (B, 2)
        centered = coords - hip_center.unsqueeze(1)
        scale = centered.abs().max(dim=1)[0].max(dim=1)[0].clamp_min(1e-6)  # (B,)
        return centered / scale.unsqueeze(1).unsqueeze(2)
