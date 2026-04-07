"""Hyper-Sapiens main wrapper: multi-head decoder with LoRA backbone.

Orchestrates:
  - Shared LoRA-adapted Sapiens backbone
  - Pose Head (HeatmapHead from mmpose)
  - Depth Head (relative depth prediction)
  - Normal Head (surface normal prediction)
  - GRPO pose optimization (RL-based alignment)
  - DDP prior (SDS loss for pose regularization)
  - Geometric consistency (depth-normal cross-modal constraint)

Training stages:
  Stage 1: Supervised on synthetic (all heads trained with GT)
  Stage 2: Mixed synthetic+real (DDP + geo consistency on real)
  Stage 3: Real domain refinement (DDP weight annealing)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel

from .depth_head import DepthHead
from .normal_head import NormalHead


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Convert heatmaps to pixel coordinates via soft-argmax.

    Args:
        heatmaps: (B, K, H, W) predicted heatmaps.
    Returns:
        coords: (B, K, 2) coordinates in pixel space (x, y).
    """
    B, K, H, W = heatmaps.shape
    h = heatmaps.view(B, K, -1)
    h = F.softmax(h, dim=-1).view(B, K, H, W)

    ys = torch.arange(H, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 1, W)
    x = (h * xs).sum(dim=(2, 3))
    y = (h * ys).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)  # (B, K, 2)


class HyperSapiensWrapper(BaseModel):
    """Multi-head wrapper for Hyper-Sapiens training.

    This module manages the shared backbone and three prediction heads,
    along with the auxiliary loss modules (GRPO, DDP, geo consistency).

    Args:
        pose_estimator: Config dict for the base TopdownPoseEstimator
            (contains backbone + pose head). Built via mmpose registry.
        depth_head_cfg: Config dict for DepthHead.
        normal_head_cfg: Config dict for NormalHead.
        grpo_cfg: Config dict for GRPOPoseLoss (optional, Stage 2/3).
        sds_cfg: Config dict for SDSLoss (optional, Stage 2/3).
        geo_cfg: Config dict for geometric consistency losses (optional).
        stage: Current training stage (1, 2, or 3).
    """

    def __init__(self,
                 pose_estimator: dict,
                 depth_head_cfg: Optional[dict] = None,
                 normal_head_cfg: Optional[dict] = None,
                 grpo_cfg: Optional[dict] = None,
                 sds_cfg: Optional[dict] = None,
                 geo_cfg: Optional[dict] = None,
                 stage: int = 1):
        super().__init__()

        # Build pose estimator (backbone + pose head)
        from mmpose.registry import MODELS as POSE_MODELS
        self.pose_estimator = POSE_MODELS.build(pose_estimator)

        # Extract backbone reference (handles LoRAModel wrapping)
        self.backbone = self.pose_estimator.backbone

        # Build auxiliary heads
        if depth_head_cfg:
            in_ch = self._get_backbone_channels()
            self.depth_head = DepthHead(in_channels=in_ch, **depth_head_cfg)
        else:
            self.depth_head = None

        if normal_head_cfg:
            in_ch = self._get_backbone_channels()
            self.normal_head = NormalHead(in_channels=in_ch, **normal_head_cfg)
        else:
            self.normal_head = None

        # GRPO and DDP are set up externally and attached
        self.grpo_loss = None
        self.sds_loss = None
        self.geo_losses = nn.ModuleDict()

        self.stage = stage

    def _get_backbone_channels(self) -> int:
        """Infer backbone output channels from architecture."""
        # Try to get embed_dims from the backbone (handles LoRA wrapping)
        backbone = self.backbone
        if hasattr(backbone, 'module'):
            backbone = backbone.module
        if hasattr(backbone, 'embed_dims'):
            return backbone.embed_dims
        return 1024  # fallback for sapiens_1b

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run backbone to get shared feature map.

        Args:
            images: (B, 3, H, W) input images.
        Returns:
            features: (B, C, H', W') feature map.
        """
        feats = self.backbone(images)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        return feats

    def forward_train(self, images: torch.Tensor,
                      data_samples: list,
                      depth_gt: Optional[torch.Tensor] = None,
                      normal_gt: Optional[torch.Tensor] = None,
                      is_labeled: bool = True) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            images: (B, 3, H, W) input images.
            data_samples: List of PoseDataSample with GT annotations.
            depth_gt: (B, 1, H', W') depth GT (Stage 1, synthetic only).
            normal_gt: (B, 3, H', W') normal GT (Stage 1, synthetic only).
            is_labeled: Whether this batch has pose GT labels.

        Returns:
            losses: Dict of loss tensors.
        """
        losses = {}

        # 1. Shared backbone features
        features = self.extract_features(images)

        # 2. Pose head losses (supervised, if labeled)
        if is_labeled:
            # Use the pose estimator's own loss computation
            pose_losses = self.pose_estimator.loss(images, data_samples)
            losses.update(pose_losses)

        # 3. Depth head (Stage 1: supervised; Stage 2/3: no GT)
        if self.depth_head is not None:
            depth_pred = self.depth_head(features)
            if depth_gt is not None:
                losses['loss_depth'] = self.depth_head.loss(depth_pred, depth_gt)

        # 4. Normal head (Stage 1: supervised; Stage 2/3: no GT)
        if self.normal_head is not None:
            normal_pred = self.normal_head(features)
            if normal_gt is not None:
                losses['loss_normal'] = self.normal_head.loss(normal_pred, normal_gt)

        # 5. Geometric consistency (Stage 2/3, no GT needed)
        if self.depth_head is not None and self.normal_head is not None:
            if 'depth_normal_consistency' in self.geo_losses:
                losses['loss_geo_dn'] = self.geo_losses['depth_normal_consistency'](
                    depth_pred, normal_pred
                )

            if 'bone_ratio' in self.geo_losses:
                # Need pose coordinates
                with torch.no_grad():
                    heatmaps = self.pose_estimator.head.forward(features)
                    if isinstance(heatmaps, (list, tuple)):
                        heatmaps = heatmaps[0]
                coords = soft_argmax_2d(heatmaps)
                losses['loss_geo_bone'] = self.geo_losses['bone_ratio'](
                    coords, depth_pred
                )

        # 6. GRPO loss (Stage 2/3, no GT needed)
        if self.grpo_loss is not None and self.stage >= 2:
            heatmaps = self.pose_estimator.head.forward(features)
            if isinstance(heatmaps, (list, tuple)):
                heatmaps = heatmaps[0]
            grpo_result = self.grpo_loss(heatmaps)
            losses['loss_grpo'] = grpo_result['loss']

        # 7. SDS loss (Stage 2/3, no GT needed)
        if self.sds_loss is not None and self.stage >= 2:
            heatmaps = self.pose_estimator.head.forward(features)
            if isinstance(heatmaps, (list, tuple)):
                heatmaps = heatmaps[0]
            coords = soft_argmax_2d(heatmaps)
            sds_result = self.sds_loss(coords)
            losses['loss_sds'] = sds_result['loss']

        return losses

    def forward_test(self, images: torch.Tensor,
                     data_samples: list) -> list:
        """Test forward — delegates to pose estimator."""
        return self.pose_estimator.predict(images, data_samples)

    def forward(self, inputs, data_samples=None, mode='loss', **kwargs):
        """MMEngine-compatible forward dispatch."""
        if mode == 'loss':
            return self.forward_train(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.forward_test(inputs, data_samples)
        else:
            raise ValueError(f'Unsupported mode: {mode}')
