"""Lightweight depth estimation head for Hyper-Sapiens multi-head decoder.

Shares the deconv+conv pattern with HeatmapHead / VitDepthHead but outputs
a single-channel relative depth map. Used in Stage 1 (supervised on synthetic
data with depth GT) and Stage 2/3 (geometric consistency constraint).

Part of the cross-modal alignment architecture: the shared backbone must
produce features useful for Pose, Depth, and Normal simultaneously.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_upsample_layer


class DepthHead(nn.Module):
    """Single-channel relative depth prediction head.

    Architecture: DeconvLayers → ConvLayers → 1×1 Conv → sigmoid

    Args:
        in_channels: Number of input feature channels from backbone.
        deconv_out_channels: Output channels for each deconv layer.
        deconv_kernel_sizes: Kernel size for each deconv layer.
        conv_out_channels: Output channels for intermediate conv layers.
        conv_kernel_sizes: Kernel sizes for intermediate conv layers.
    """

    def __init__(self,
                 in_channels: int = 1024,
                 deconv_out_channels: Sequence[int] = (256, 256),
                 deconv_kernel_sizes: Sequence[int] = (4, 4),
                 conv_out_channels: Optional[Sequence[int]] = (256,),
                 conv_kernel_sizes: Optional[Sequence[int]] = (3,)):
        super().__init__()

        # Deconv layers (upsampling)
        layers = []
        ch = in_channels
        for out_ch, ks in zip(deconv_out_channels, deconv_kernel_sizes):
            padding = (ks - 1) // 2
            output_padding = padding  # ensures exact 2x upsampling for ks=4
            layers.append(nn.ConvTranspose2d(
                ch, out_ch, kernel_size=ks, stride=2,
                padding=padding, output_padding=output_padding, bias=False,
            ))
            layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.SiLU(inplace=True))
            ch = out_ch
        self.deconv_layers = nn.Sequential(*layers)

        # Conv layers (refinement)
        conv_layers = []
        if conv_out_channels:
            for out_ch, ks in zip(conv_out_channels, conv_kernel_sizes):
                padding = (ks - 1) // 2
                conv_layers.append(nn.Conv2d(ch, out_ch, ks, padding=padding, bias=False))
                conv_layers.append(nn.InstanceNorm2d(out_ch))
                conv_layers.append(nn.SiLU(inplace=True))
                ch = out_ch
        self.conv_layers = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        # Final 1×1 conv → 1 channel depth
        self.final_layer = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) backbone feature map.
        Returns:
            depth: (B, 1, H', W') relative depth map in [0, 1].
        """
        x = self.deconv_layers(features)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return torch.sigmoid(x)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scale-invariant depth loss (SILog).

        Args:
            pred: (B, 1, H, W) predicted depth.
            target: (B, 1, H, W) ground-truth depth.
            mask: (B, 1, H, W) valid pixel mask (optional).
        """
        eps = 1e-6
        log_diff = torch.log(pred.clamp_min(eps)) - torch.log(target.clamp_min(eps))

        if mask is not None:
            log_diff = log_diff * mask
            n = mask.sum().clamp_min(1)
            si_loss = (log_diff ** 2).sum() / n - \
                      0.5 * (log_diff.sum() / n) ** 2
        else:
            n = log_diff.numel()
            si_loss = (log_diff ** 2).mean() - 0.5 * log_diff.mean() ** 2

        return si_loss
