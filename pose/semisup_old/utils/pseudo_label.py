
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

COCO_SIGMAS_17 = torch.tensor([
    0.26,0.25,0.25,0.35,0.35,0.79,0.79,0.72,0.72,0.62,0.62,1.07,1.07,0.87,0.87,0.89,0.89
])

__all__ = ['fuse_teachers', 'oks_from_heatmaps', 'dynamic_kpt_mask']

def fuse_teachers(hm_g: torch.Tensor, conf_g: torch.Tensor,
                  hm_a: torch.Tensor, conf_a: torch.Tensor, T: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """双教师加权融合；返回融合热图与融合置信。
    hm_*, conf_*: (B,K,H,W), (B,K)
    """
    eps = 1e-6
    w = conf_g / (conf_g + conf_a + eps)            # (B,K)
    hm = w.unsqueeze(-1).unsqueeze(-1) * hm_g + (1-w).unsqueeze(-1).unsqueeze(-1) * hm_a
    conf = torch.maximum(conf_g, conf_a)
    return hm, conf

def oks_from_heatmaps(hm: torch.Tensor, gt_hm: torch.Tensor, sigmas: torch.Tensor=COCO_SIGMAS_17, img_size=(1024,768)) -> torch.Tensor:
    """用热图间 L2 距离近似 OKS 的打分（仅用于实例级筛选的相对排序）。"""
    # 简化：用峰值坐标距离和热图相似度综合
    B,K,H,W = hm.shape
    pred = torch.stack(torch.meshgrid(
        torch.arange(H, device=hm.device), torch.arange(W, device=hm.device), indexing='ij'), dim=-1)
    pred = pred.view(1,1,H,W,2).float()
    w = F.softmax(hm.view(B,K,-1), dim=-1).view(B,K,H,W)
    mu = (w.unsqueeze(-1) * pred.view(1,1,H,W,2)).sum(dim=(2,3))  # (B,K,2)
    # 这里仅演示：与 gt_hm 的 soft-argmax 坐标距离 -> 近似 OKS
    w_gt = F.softmax(gt_hm.view(B,K,-1), dim=-1).view(B,K,H,W)
    mu_gt = (w_gt.unsqueeze(-1) * pred.view(1,1,H,W,2)).sum(dim=(2,3))
    d2 = ((mu-mu_gt).pow(2).sum(-1))  # (B,K)
    vars = (sigmas.to(hm.device) * 2)**2
    oks_k = torch.exp(-d2 / (2*vars.view(1,-1) + 1e-6))
    return oks_k.mean(dim=-1)  # (B,)

def dynamic_kpt_mask(conf: torch.Tensor, stats: Dict[str, torch.Tensor], beta: float=0.5) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """按关键点运行统计生成逐点动态阈值掩码。
    conf: (B,K)
    stats: {'mu': (K,), 'sigma': (K,), 'n': int}
    返回: mask (B,K) ∈ {0,1}, 更新后的 stats
    """
    K = conf.shape[1]
    if 'mu' not in stats:
        stats['mu'] = conf.mean(dim=0).detach()
        stats['sigma'] = conf.std(dim=0).detach()+1e-6
        stats['n'] = 1
    mu = stats['mu']
    sigma = stats['sigma']
    tau = mu - beta * sigma
    mask = (conf >= tau.view(1,-1)).float()
    # 更新运行统计（简单 EMA）
    m = 0.99
    stats['mu'] = m*mu + (1-m)*conf.mean(dim=0).detach()
    stats['sigma'] = m*sigma + (1-m)*conf.std(dim=0).detach()
    stats['n'] += 1
    return mask, stats
