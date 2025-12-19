import torch
import torch.nn.functional as F

@torch.no_grad()
def fuse_teachers(
    hm_g: torch.Tensor,
    conf_g: torch.Tensor,
    hm_a: torch.Tensor,
    conf_a: torch.Tensor,
    T: float = 1.0,
    spatial_tau: float = 0.5,
    sharpen: bool = True,
    conf_from: str = 'prob_max',
):
    """融合两位教师的热图，并可在空间维进行温度锐化（softmax over H×W）。

    Args:
        hm_g: (B, K, H, W) 几何教师的“logits”热图
        conf_g: (B, K)     几何教师逐关键点置信（可为峰值）
        hm_a: (B, K, H, W) 外观教师的“logits”热图
        conf_a: (B, K)     外观教师逐关键点置信
        T: 标量温度，用于计算 per-keypoint 融合权重 w 的对比度（越小对比越强）
        spatial_tau: 空间温度，<1 会锐化空间分布；=1 不变；>1 会变平滑
        sharpen: 是否对融合后的热图在空间维做 softmax(+temperature) 得到概率图
        conf_from: 置信估计方式:
            - 'prob_max': 用空间概率图的最大值作为置信（推荐）
            - 其他:       退化为 max(conf_g, conf_a)

    Returns:
        fused_map: (B, K, H, W)
            - 若 sharpen=True：返回“空间概率图”（softmax 后）
            - 若 sharpen=False：返回融合后的 logits
        fused_conf: (B, K) 融合后的逐关键点置信
    """
    eps = 1e-6

    # 1) 逐关键点融合权重 w ∈ [0,1]
    if T is not None and T > 0:
        w = torch.sigmoid((conf_g - conf_a) / T)
    else:
        w = conf_g / (conf_g + conf_a + eps)
    w = w.clamp_(0, 1)

    # 2) 在 logits 空间按 w 线性融合两位教师
    #    注意：这里保持“先融合，再空间 softmax”的顺序
    hm = w.unsqueeze(-1).unsqueeze(-1) * hm_g + (1.0 - w).unsqueeze(-1).unsqueeze(-1) * hm_a

    # 3) 空间 softmax + 温度化（得到概率图），并据此评估置信
    if sharpen:
        B, K, H, W = hm.shape
        tau = max(float(spatial_tau), 1e-6)
        # 对每个关键点通道在 H×W 上归一化
        # p = F.softmax(hm.reshape(B, K, -1) / tau, dim=-1).reshape(B, K, H, W)
        # ---- 强化 logits：每 (B,K) 做 z-score，然后再除以 tau ----

        # print('[logit dbg][geom] mean=%.6f std=%.6f max=%.6f' %
        #     (float(hm.mean()), float(hm.std()), float(hm.abs().max())))
        
        m = hm.flatten(2).mean(-1).view(B, K, 1, 1)
        s = hm.flatten(2).std(-1, unbiased=False).clamp_min(1e-6).view(B, K, 1, 1)
        logits = (hm - m) / s                        # 归一化到 ~N(0,1)

        # print('[tau dbg] spatial_tau =', float(tau))
        
        p = torch.softmax(logits.flatten(2) / tau, dim=-1).view(B, K, H, W)
        p = p.clamp_min(1e-12)
        p = p / (p.flatten(2).sum(-1).view(B, K, 1, 1) + 1e-12)


        if conf_from == 'prob_max':
            conf = p.flatten(2).max(dim=-1).values  # (B, K)
        else:
            conf = torch.maximum(conf_g, conf_a)

        S = p.flatten(2).sum(dim=-1).mean().item()
        Cmean = conf.mean().item(); Cmax = conf.max().item()
        # print(f'[fuse dbg] sum≈{S:.4f}, prob_max mean={Cmean:.5f}, max={Cmax:.5f}')
        # print('[fuse dbg] sum=', float(p.flatten(2).sum(-1).mean()),
        #     ', prob_max mean=', float(p.flatten(2).max(-1).values.mean()),
        #     ', max=', float(p.max()))


        return p, conf
    else:
        # 不做空间锐化时，保持 logits 返回；置信取两者较大
        conf = torch.maximum(conf_g, conf_a)
        return hm, conf


def dynamic_kpt_mask(
    conf: torch.Tensor,
    stats: dict,
    beta: float = 0.5,
    momentum: float = 0.1,
    min_keep_ratio: float = 0.0,
    percentile: float = None,
):
    """根据关键点运行统计生成逐点掩码。

    Args:
        conf: (B, K) 当前 batch 的逐关键点置信（建议来自概率图的空间最大值）
        stats: dict，包含 {'mu','sigma','n'} 这三项的历史统计（tensor）
        beta: 阈值强度，阈值 thr = mu + beta * sigma ；beta 越大筛得越严
        momentum: 统计更新动量（越小越稳）
        min_keep_ratio: 每个样本的最小保留比例（0~1），用于避免 mask 全 0
        percentile: 若给定，例如 0.5 表示至少保留每个关键点在本 batch 的 top-50%

    Returns:
        mask: (B, K) ∈ {0,1}
        new_stats: 更新后的 {'mu','sigma','n'}
    """
    device = conf.device
    mu = stats['mu'].to(device)
    sigma = stats['sigma'].to(device)
    n = int(stats['n'])

    # 1) 计算 batch 级统计
    batch_mu = conf.mean(dim=0)
    batch_sigma = conf.std(dim=0).clamp_min(1e-6)

    # 2) 指数滑动更新
    m = float(momentum)
    mu = (1 - m) * mu + m * batch_mu
    sigma = (1 - m) * sigma + m * batch_sigma
    n = n + conf.shape[0]

    # 3) 基于均值-方差的动态阈值
    thr = (mu + beta * sigma).clamp(0.0, 1.0)  # beta 越大越“苛刻”
    mask = (conf >= thr).float()  # (B, K)

    # 4) 可选：按百分位保证每个关键点的最低通过比例
    if percentile is not None:
        q = torch.quantile(conf, float(percentile), dim=0)  # (K,)
        mask = torch.maximum(mask, (conf >= q.unsqueeze(0)).float())

    # 5) 可选：按样本保证至少保留一定比例的关键点
    if min_keep_ratio > 0:
        B, K = conf.shape
        k_min = max(1, int(K * min_keep_ratio))
        topk_idx = conf.topk(k_min, dim=1).indices  # (B, k_min)
        base = torch.zeros_like(mask)
        base.scatter_(1, topk_idx, 1.0)
        mask = torch.maximum(mask, base)

    return mask, {'mu': mu.detach(), 'sigma': sigma.detach(), 'n': n}
