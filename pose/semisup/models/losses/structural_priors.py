import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_argmax_2d(heatmaps: torch.Tensor):
    """把热图转为坐标（-1~1 规范化坐标系）"""
    B, K, H, W = heatmaps.shape
    h = heatmaps.view(B, K, -1)
    h = F.softmax(h, dim=-1).view(B, K, H, W)
    ys = torch.linspace(-1, 1, H, device=heatmaps.device, dtype=heatmaps.dtype).view(1,1,H,1)
    xs = torch.linspace(-1, 1, W, device=heatmaps.device, dtype=heatmaps.dtype).view(1,1,1,W)
    x = (h * xs).sum(dim=(2,3))
    y = (h * ys).sum(dim=(2,3))
    return torch.stack([x, y], dim=-1)  # (B,K,2)

class LaplacianTopoLoss(nn.Module):
    def __init__(self, edges, weight=0.05):
        super().__init__()
        self.edges = edges
        self.weight = weight

    def forward(self, coords, mask_edges: torch.Tensor = None):
        """
        coords:     (B,K,2)，建议在[-1,1]归一化坐标系
        mask_edges: (B,|E|) in {0,1} 或 [0,1]，可选
        """
        B = coords.shape[0]
        E = len(self.edges)
        diffs = []
        for (u, v) in self.edges:
            d = (coords[:, u] - coords[:, v]).abs().sum(dim=-1)  # (B,)
            diffs.append(d)
        D = torch.stack(diffs, dim=-1)  # (B,E)

        if mask_edges is not None:
            w = mask_edges.float()
            wsum = w.sum(dim=-1).clamp_min(1.0)  # (B,)
            loss = (D * w).sum(dim=-1) / wsum     # (B,)
            return self.weight * loss.mean()
        else:
            return self.weight * D.mean()


class BoneLengthLoss(nn.Module):
    def __init__(self, edges, ref_bone_lengths, weight=0.05, scale_invariant=True):
        super().__init__()
        self.edges = edges
        self.register_buffer('ref', torch.tensor(ref_bone_lengths, dtype=torch.float32), persistent=False)
        self.weight = weight
        self.scale_inv = bool(scale_invariant)
        # 预存“参考骨长”的无量纲版本（用于 scale_invariant）
        ref = self.ref.clamp_min(1e-6)
        self.register_buffer('ref_norm', ref / ref.mean(), persistent=False)

    def forward(self, coords, mask_edges: torch.Tensor = None):
        B = coords.shape[0]
        lens = []
        for (u, v) in self.edges:
            l = (coords[:, u] - coords[:, v]).pow(2).sum(dim=-1).sqrt()  # (B,)
            lens.append(l)
        L = torch.stack(lens, dim=-1)  # (B,E)

        if self.scale_inv:
            L_norm = L / (L.mean(dim=-1, keepdim=True).clamp_min(1e-6))   # (B,E)
            target = self.ref_norm.to(device=coords.device, dtype=coords.dtype).unsqueeze(0)
            err = (L_norm - target).abs()                                  # (B,E)
        else:
            ref = self.ref.to(device=coords.device, dtype=coords.dtype).unsqueeze(0)
            ratio = (L / ref.clamp_min(1e-6))
            err = (ratio - 1.0).abs()                                      # (B,E)

        if mask_edges is not None:
            w = mask_edges.float()
            wsum = w.sum(dim=-1).clamp_min(1.0)
            loss = (err * w).sum(dim=-1) / wsum
            return self.weight * loss.mean()
        else:
            return self.weight * err.mean()


class JointAngleLoss(nn.Module):
    def __init__(self, triplets, weight=0.02):
        super().__init__()
        self.triplets = triplets
        self.weight = weight

    def forward(self, coords, mask_triplets: torch.Tensor = None):
        eps = 1e-6
        vals = []
        for (i, j, k) in self.triplets:
            v1 = coords[:, i] - coords[:, j]
            v2 = coords[:, k] - coords[:, j]
            cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + eps)
            vals.append(cos.abs())   # |cos| 接近 1 不合理
        V = torch.stack(vals, dim=-1)  # (B, T)

        if mask_triplets is not None:
            w = mask_triplets.float()
            wsum = w.sum(dim=-1).clamp_min(1.0)
            loss = (V * w).sum(dim=-1) / wsum
            return self.weight * loss.mean()
        else:
            return self.weight * V.mean()

