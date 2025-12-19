import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def warp_heatmaps_affine(heatmaps: torch.Tensor,
                         A: torch.Tensor,
                         out_size,
                         align_corners: bool = False) -> torch.Tensor:
    """把教师热图按仿射矩阵 A warp 到指定分辨率。
    Args:
        heatmaps: (B, K, H, W)  可能是 bf16/fp16/fp32
        A:        (B, 2, 3)     教师->目标视图 的仿射矩阵
        out_size: (H_out, W_out)
    Returns:
        warped:   (B, K, H_out, W_out)，dtype 与输入 heatmaps 保持一致
    """
    B, K, H, W = heatmaps.shape
    H_out, W_out = out_size
    dev = heatmaps.device
    orig_dtype = heatmaps.dtype

    assert heatmaps.dim() == 4 and heatmaps.shape[1] > 0, 'heatmaps must be (B,K,H,W)'
    assert A.shape[-2:] == (2,3) and A.shape[0] == heatmaps.shape[0], 'A shape must be (B,2,3)'


    # 计算部分统一用 float32，禁用 autocast，避免 bf16 的 inverse 限制
    with autocast(enabled=False):
        hm32 = heatmaps.to(torch.float32)
        A32  = A.to(device=dev, dtype=torch.float32)

        # 组装 3x3 仿射并求逆（输出->输入）
        eye = torch.eye(3, device=dev, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3).clone()
        eye[:, :2, :] = A32
        inv   = torch.linalg.inv(eye)      # float32 安全
        theta = inv[:, :2, :]              # (B, 2, 3)

        # ★ 关键：theta 扩展到 (B*K, 2, 3)，与展平后的输入 batch 对齐
        theta_rep = theta.repeat_interleave(K, dim=0)  # (B*K, 2, 3)

        # 构建采样网格并采样
        grid  = F.affine_grid(theta_rep, size=(B * K, 1, H_out, W_out),
                              align_corners=align_corners)
        hm_in = hm32.reshape(B * K, 1, H, W)
        # warped = F.grid_sample(hm_in, grid, mode='bilinear',
        #                        padding_mode='zeros', align_corners=align_corners)
        grid = grid.clamp_(-1, 1)  # 防越界
        warped = F.grid_sample(
            hm_in.float(), grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        # 数值保护 + 概率重归一化（逐关键点）
        warped = warped.clamp_min(1e-8)
        warped = warped / warped.sum(dim=(-1, -2), keepdim=True)
        warped = warped.view(B, K, H_out, W_out).to(orig_dtype)

    return warped
