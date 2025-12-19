
from typing import Dict
import torch
import torch.nn.functional as F

__all__ = ['warp_heatmaps_affine']

def warp_heatmaps_affine(heatmaps: torch.Tensor, affine_matrices: torch.Tensor, out_size: tuple) -> torch.Tensor:
    """把热图按仿射矩阵映射到指定强增视图坐标系。
    heatmaps: (B,K,H,W)
    affine_matrices: (B,2,3) from teacher->student view
    out_size: (H_out, W_out)
    """
    B, K, H, W = heatmaps.shape
    H_out, W_out = out_size
    grid = F.affine_grid(affine_matrices, size=(B*K, 1, H_out, W_out), align_corners=False)
    x = heatmaps.view(B*K, 1, H, W)
    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return y.view(B, K, H_out, W_out)
