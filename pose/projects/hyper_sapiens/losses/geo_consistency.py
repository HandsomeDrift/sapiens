"""Cross-modal geometric consistency losses.

These losses enforce physical constraints between the outputs of the
multi-head decoder (Pose, Depth, Normal) without requiring ground-truth
labels — enabling self-supervised learning on unlabeled real data.

This is analogous to cross-modal alignment in multimodal models: different
modalities (skeleton, depth field, surface normals) must be geometrically
consistent in a shared 3D space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNormalConsistencyLoss(nn.Module):
    """Enforce consistency between predicted depth and normal maps.

    From differential geometry: the surface normal can be derived from the
    depth map's spatial gradient. The directly predicted normals (Normal Head)
    should agree with the depth-derived normals (Depth Head).

    L = || n_pred - n_derived(D) ||_1

    All computation in float32 for numerical stability.

    Args:
        weight: Loss weight.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    @staticmethod
    def depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
        """Derive surface normals from a depth map via finite differences.

        Args:
            depth: (B, 1, H, W) depth map.
        Returns:
            normals: (B, 3, H, W) derived normal vectors (unit length).
        """
        depth = depth.float()
        # Sobel-like finite differences
        dz_dx = depth[:, :, :, 2:] - depth[:, :, :, :-2]  # (B, 1, H, W-2)
        dz_dy = depth[:, :, 2:, :] - depth[:, :, :-2, :]  # (B, 1, H-2, W)

        # Crop to common size
        H = min(dz_dx.shape[2], dz_dy.shape[2])
        W = min(dz_dx.shape[3], dz_dy.shape[3])
        dz_dx = dz_dx[:, :, :H, :W]
        dz_dy = dz_dy[:, :, :H, :W]

        # Normal = (-dz/dx, -dz/dy, 1), then normalize
        ones = torch.ones_like(dz_dx)
        normals = torch.cat([-dz_dx, -dz_dy, ones], dim=1)  # (B, 3, H, W)
        return F.normalize(normals, p=2, dim=1)

    def forward(self, depth_pred: torch.Tensor,
                normal_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_pred: (B, 1, H, W) from Depth Head.
            normal_pred: (B, 3, H, W) from Normal Head.
        Returns:
            loss: scalar consistency loss.
        """
        n_derived = self.depth_to_normal(depth_pred)

        # Crop normal_pred to match derived size
        H, W = n_derived.shape[2], n_derived.shape[3]
        n_pred = normal_pred[:, :, 1:H+1, 1:W+1]

        # L1 loss between predicted and derived normals
        loss = (n_pred.float() - n_derived).abs().mean()
        return self.weight * loss


class BoneRatioConsistencyLoss(nn.Module):
    """Scale-invariant bone ratio consistency using depth information.

    Projects 2D keypoints into 2.5D space using the depth map, computes
    bone lengths, and enforces that bone length *ratios* match the
    population statistics from synthetic data.

    This avoids the absolute scale ambiguity of monocular depth while
    still providing 3D geometric constraints.

    Args:
        edges: List of (u, v) bone edge index pairs.
        ref_ratios: Reference bone length ratios from synthetic data.
        weight: Loss weight.
        depth_scale: Scaling factor for depth contribution in 2.5D distance.
    """

    def __init__(self, edges: list, ref_ratios: list,
                 weight: float = 0.1, depth_scale: float = 1.0):
        super().__init__()
        self.edges = edges
        self.register_buffer('ref_ratios',
                             torch.tensor(ref_ratios, dtype=torch.float32))
        self.weight = weight
        self.depth_scale = depth_scale

    @staticmethod
    def sample_depth_at_keypoints(depth_map: torch.Tensor,
                                  coords: torch.Tensor) -> torch.Tensor:
        """Sample depth values at keypoint locations via bilinear interpolation.

        Args:
            depth_map: (B, 1, H, W) depth prediction.
            coords: (B, K, 2) keypoint pixel coordinates (x, y).
        Returns:
            depths: (B, K) depth values at keypoint locations.
        """
        B, _, H, W = depth_map.shape
        # Normalize coords to [-1, 1] for grid_sample
        grid_x = 2.0 * coords[..., 0] / (W - 1) - 1.0
        grid_y = 2.0 * coords[..., 1] / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (B, K, 2)
        grid = grid.unsqueeze(2)  # (B, K, 1, 2) for grid_sample

        sampled = F.grid_sample(
            depth_map.float(), grid.float(),
            mode='bilinear', padding_mode='border', align_corners=True,
        )  # (B, 1, K, 1)
        return sampled.squeeze(1).squeeze(-1)  # (B, K)

    def forward(self, coords: torch.Tensor,
                depth_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, K, 2) keypoint pixel coordinates.
            depth_map: (B, 1, H, W) depth prediction.
        Returns:
            loss: scalar bone ratio consistency loss.
        """
        coords = coords.float()
        depths = self.sample_depth_at_keypoints(depth_map, coords)  # (B, K)

        # Compute 2.5D bone lengths
        bone_lengths = []
        for (u, v) in self.edges:
            dx = coords[:, u, 0] - coords[:, v, 0]
            dy = coords[:, u, 1] - coords[:, v, 1]
            dz = (depths[:, u] - depths[:, v]) * self.depth_scale
            length = (dx**2 + dy**2 + dz**2).sqrt()  # (B,)
            bone_lengths.append(length)
        L = torch.stack(bone_lengths, dim=-1)  # (B, E)

        # Compute ratios relative to mean bone length (scale-invariant)
        L_mean = L.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        L_ratio = L / L_mean  # (B, E)

        # Compare with reference ratios
        ref = self.ref_ratios.to(L.device, L.dtype).unsqueeze(0)
        loss = (L_ratio - ref).pow(2).mean()
        return self.weight * loss
