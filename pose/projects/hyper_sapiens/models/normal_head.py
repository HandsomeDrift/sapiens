"""Surface normal prediction head for Hyper-Sapiens multi-head decoder.

Outputs a 3-channel unit normal vector map. Used alongside DepthHead to
provide cross-modal geometric consistency constraints — the depth-derived
normals should agree with directly predicted normals.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalHead(nn.Module):
    """3-channel surface normal prediction head.

    Architecture: DeconvLayers → ConvLayers → 1×1 Conv → L2 normalize

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

        layers = []
        ch = in_channels
        for out_ch, ks in zip(deconv_out_channels, deconv_kernel_sizes):
            padding = (ks - 1) // 2
            output_padding = padding
            layers.append(nn.ConvTranspose2d(
                ch, out_ch, kernel_size=ks, stride=2,
                padding=padding, output_padding=output_padding, bias=False,
            ))
            layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.SiLU(inplace=True))
            ch = out_ch
        self.deconv_layers = nn.Sequential(*layers)

        conv_layers = []
        if conv_out_channels:
            for out_ch, ks in zip(conv_out_channels, conv_kernel_sizes):
                padding = (ks - 1) // 2
                conv_layers.append(nn.Conv2d(ch, out_ch, ks, padding=padding, bias=False))
                conv_layers.append(nn.InstanceNorm2d(out_ch))
                conv_layers.append(nn.SiLU(inplace=True))
                ch = out_ch
        self.conv_layers = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        # Final 1×1 conv → 3 channels (nx, ny, nz)
        self.final_layer = nn.Conv2d(ch, 3, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) backbone feature map.
        Returns:
            normals: (B, 3, H', W') unit normal vectors.
        """
        x = self.deconv_layers(features)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        # L2 normalize to unit vectors
        return F.normalize(x, p=2, dim=1)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Angular (cosine) loss between predicted and target normals.

        Args:
            pred: (B, 3, H, W) predicted normals (unit vectors).
            target: (B, 3, H, W) GT normals (unit vectors).
            mask: (B, 1, H, W) valid pixel mask (optional).
        """
        # Cosine similarity per pixel
        cos_sim = (pred * target).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        loss = 1.0 - cos_sim  # angular loss: 0 when aligned, 2 when opposite

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        else:
            loss = loss.mean()

        return loss
