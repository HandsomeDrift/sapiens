python - <<'PY'
import torch
from pose.semisup.utils.geometry import warp_heatmaps_affine

B,K,H,W = 2,17,192,256
hm  = torch.randn(B,K,H,W, device='cuda', dtype=torch.bfloat16)
A   = torch.tensor([[[1,0,0],[0,1,0]]]*B, device='cuda', dtype=torch.bfloat16)  # 恒等
out = warp_heatmaps_affine(hm, A, out_size=(H,W))
print(out.shape, out.dtype)   # 期望: torch.Size([2, 17, 192, 256]) torch.bfloat16
PY
