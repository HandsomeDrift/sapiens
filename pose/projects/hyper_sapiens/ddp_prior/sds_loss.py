"""Score Distillation Sampling (SDS) loss for pose estimation.

Adapts the SDS technique from DreamFusion to skeleton-level optimization.
A frozen DDP model provides the score function ∇_x log p(x), which pushes
predicted poses toward the learned "valid infant pose" distribution.

The SDS gradient:
    ∇_x L_SDS = w(t) * (ε_pred - ε)

where ε is the added noise and ε_pred is the DDP's prediction. This gradient
points toward higher-probability regions of the pose distribution.
"""

import torch
import torch.nn as nn

from .ddp_model import SkeletonDDPM


class SDSLoss(nn.Module):
    """Score Distillation Sampling loss using a frozen DDP prior.

    Args:
        ddp_model: Pre-trained SkeletonDDPM (will be frozen).
        t_range: (t_min, t_max) timestep range for sampling.
            Smaller t → less noise → finer corrections.
            Larger t → more noise → stronger regularization.
        weight: Loss weight multiplier.
    """

    def __init__(self, ddp_model: SkeletonDDPM,
                 t_range: tuple = (20, 500),
                 weight: float = 0.01):
        super().__init__()
        self.ddp = ddp_model
        self.ddp.eval()
        for p in self.ddp.parameters():
            p.requires_grad_(False)

        self.t_min, self.t_max = t_range
        self.weight = weight

    def forward(self, coords: torch.Tensor) -> dict:
        """Compute SDS loss on predicted pose coordinates.

        All computation is done in float32 for numerical stability.

        Args:
            coords: (B, K, 2) predicted keypoint coordinates.
                Must have requires_grad=True (or be connected to
                parameters that do) for gradients to flow back.
        Returns:
            Dict with 'loss' and logging metrics.
        """
        B, K, _ = coords.shape

        # Normalize to DDP's coordinate space
        coords_f32 = coords.float()
        x_0 = SkeletonDDPM.normalize_skeleton(coords_f32)  # (B, K, 2)
        x_0_flat = x_0.reshape(B, -1)  # (B, K*2)

        # Sample random timestep
        t = torch.randint(
            self.t_min, min(self.t_max, self.ddp.num_timesteps),
            (B,), device=coords.device,
        )

        # Add noise
        noise = torch.randn_like(x_0_flat)
        x_t = self.ddp.q_sample(x_0_flat.detach(), t, noise)

        # Predict noise with frozen DDP
        with torch.no_grad():
            eps_pred = self.ddp(x_t, t)

        # SDS gradient: w(t) * (eps_pred - eps)
        # We implement this as an MSE-like loss whose gradient equals the SDS gradient
        # w(t) is implicitly handled by the noise schedule
        w = (1.0 - self.ddp.alphas_cumprod[t]).unsqueeze(-1)  # (B, 1)

        # The "target" for the loss is: x_0_flat - w * (eps_pred - noise)
        # This ensures ∇_{x_0} L = w * (eps_pred - noise) = SDS gradient
        target = (x_0_flat - w * (eps_pred - noise)).detach()
        loss = self.weight * ((x_0_flat - target) ** 2).mean()

        return {
            'loss': loss,
            'sds_grad_norm': (w * (eps_pred - noise)).norm().item(),
            'timestep_mean': t.float().mean().item(),
        }
