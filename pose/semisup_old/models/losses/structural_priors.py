
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'soft_argmax_2d', 'LaplacianTopoLoss', 'BoneLengthLoss', 'JointAngleLoss'
]

def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """对热图做 soft-argmax，返回 (B,K,2) 的坐标（x,y）。
    heatmaps: (B, K, H, W) 概率热图（未归一化也可，会内部 softmax）。
    """
    B, K, H, W = heatmaps.shape
    heatmaps = heatmaps.view(B, K, -1)
    heatmaps = F.softmax(heatmaps, dim=-1)
    heatmaps = heatmaps.view(B, K, H, W)
    # 网格
    ys = torch.linspace(-1, 1, H, device=heatmaps.device, dtype=heatmaps.dtype).view(1,1,H,1)
    xs = torch.linspace(-1, 1, W, device=heatmaps.device, dtype=heatmaps.dtype).view(1,1,1,W)
    ex = (heatmaps * xs).sum(dim=(-1, -2))  # (B,K)
    ey = (heatmaps * ys).sum(dim=(-1, -2))
    # 还原到像素坐标 [0,W-1],[0,H-1]
    x = (ex + 1) * (W - 1) / 2
    y = (ey + 1) * (H - 1) / 2
    return torch.stack([x, y], dim=-1)

class LaplacianTopoLoss(nn.Module):
    def __init__(self, edges: List[Tuple[int,int]], weight: float=0.05):
        super().__init__()
        self.edges = edges
        self.weight = weight

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,K,2)
        if len(self.edges) == 0:
            return coords.new_zeros([])
        # 图拉普拉斯：∑e ||p_u - p_v||^2 的近似（作为连贯性正则）
        diffs = []
        for u,v in self.edges:
            diffs.append((coords[:,u]-coords[:,v]).pow(2).sum(dim=-1))
        lap = torch.stack(diffs, dim=-1).mean()
        return self.weight * lap

class BoneLengthLoss(nn.Module):
    def __init__(self, edges: List[Tuple[int,int]], ref_lengths: List[float], weight: float=0.05, eps: float=1e-6):
        super().__init__()
        self.edges = edges
        self.ref = torch.tensor(ref_lengths).view(1, -1)  # 标注集稳健中值骨长
        self.weight = weight
        self.eps = eps

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,K,2)
        lens = []
        for u,v in self.edges:
            lens.append((coords[:,u]-coords[:,v]).norm(dim=-1))
        L = torch.stack(lens, dim=-1)  # (B,E)
        ref = self.ref.to(device=coords.device, dtype=coords.dtype)
        loss = ((L / (ref + self.eps) - 1.0).abs()).mean()
        return self.weight * loss

class JointAngleLoss(nn.Module):
    def __init__(self, triplets: List[Tuple[int,int,int]], weight: float=0.02, eps: float=1e-6):
        super().__init__()
        self.triplets = triplets
        self.weight = weight
        self.eps = eps

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,K,2)
        if len(self.triplets) == 0:
            return coords.new_zeros([])
        angles = []
        for a,b,c in self.triplets:
            ba = coords[:,a]-coords[:,b]
            bc = coords[:,c]-coords[:,b]
            cosang = (ba*bc).sum(-1) / (ba.norm(dim=-1)*bc.norm(dim=-1)+self.eps)
            # 合理角度靠近余弦参考（0~π -> cos in [-1,1]，用1-cos作平滑hinge）
            angles.append(1.0 - cosang)
        angle_loss = torch.stack(angles, dim=-1).mean()
        return self.weight * angle_loss
