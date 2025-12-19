from typing import Dict, Any, List, Optional
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from mmengine.registry import MODELS

from ..utils.geometry import warp_heatmaps_affine
from ..utils.pseudo_label import fuse_teachers, dynamic_kpt_mask
from .losses.structural_priors import soft_argmax_2d, LaplacianTopoLoss, BoneLengthLoss, JointAngleLoss
import math

def _parse_dtype(name: str):
    name = str(name).lower()
    if name in ('fp16','half','float16'): return torch.float16
    if name in ('bf16','bfloat16'): return torch.bfloat16
    return torch.float32

def _rand_affine_matrix(B, max_deg=15, scale_range=(0.9,1.1), translate=0.05, device='cpu', dtype=torch.float32):
    # 返回每样本 2x3 仿射矩阵（像素坐标系下，用于 grid_sample 前的换算会做归一化）
    ang = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_deg / 180.0 * torch.pi
    sc  = torch.empty(B, device=device, dtype=dtype).uniform_(*scale_range)
    tx  = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * translate
    ty  = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * translate
    cos, sin = torch.cos(ang)*sc, torch.sin(ang)*sc
    A = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    A[:,0,0] =  cos; A[:,0,1] = -sin; A[:,0,2] = tx
    A[:,1,0] =  sin; A[:,1,1] =  cos; A[:,1,2] = ty
    return A

def _strong_augment(x: torch.Tensor):
    # x: (B,C,H,W), 仅用轻量安全的强增：随机擦除 + 轻微颜色扰动（避免破坏几何标签）
    B = x.size(0)
    # 颜色：亮度/对比度
    x = x + (torch.rand(B,1,1,1, device=x.device, dtype=x.dtype)*0.2 - 0.1)  # brightness
    x = x * (1.0 + (torch.rand(B,1,1,1, device=x.device, dtype=x.dtype)*0.3 - 0.15))  # contrast
    x = x.clamp(0.0, 1.0)
    # 随机擦除（Cutout）
    for i in range(B):
        if torch.rand(1, device=x.device) < 0.5:
            _, H, W = x.shape[1:]
            rh = int(H * torch.empty(1, device=x.device).uniform_(0.1, 0.25))
            rw = int(W * torch.empty(1, device=x.device).uniform_(0.1, 0.25))
            cy = torch.randint(rh//2, H - rh//2, (1,), device=x.device).item()
            cx = torch.randint(rw//2, W - rw//2, (1,), device=x.device).item()
            y1, y2 = max(0, cy - rh//2), min(H, cy + rh//2)
            x1, x2 = max(0, cx - rw//2), min(W, cx + rw//2)
            x[i, :, y1:y2, x1:x2] = 0.0
    return x

def _strong_augment01(x01: torch.Tensor):
    """在 [0,1] 像素域做轻量强增：小幅亮度/对比度 + cutout"""
    B, _, H, W = x01.shape
    # 亮度/对比度
    x01 = x01 + (torch.rand(B,1,1,1, device=x01.device, dtype=x01.dtype)*0.12 - 0.06)
    x01 = x01 * (1.0 + (torch.rand(B,1,1,1, device=x01.device, dtype=x01.dtype)*0.25 - 0.125))
    x01 = x01.clamp(0.0, 1.0)
    # Cutout（填充为 0=黑色，更符合像素域语义）
    for i in range(B):
        if torch.rand(1, device=x01.device) < 0.5:
            rh = int(H * torch.empty(1, device=x01.device).uniform_(0.1, 0.25))
            rw = int(W * torch.empty(1, device=x01.device).uniform_(0.1, 0.25))
            cy = torch.randint(rh//2, H - rh//2, (1,), device=x01.device).item()
            cx = torch.randint(rw//2, W - rw//2, (1,), device=x01.device).item()
            y1, y2 = max(0, cy - rh//2), min(H, cy + rh//2)
            x1, x2 = max(0, cx - rw//2), min(W, cx + rw//2)
            x01[i, :, y1:y2, x1:x2] = 0.0
    return x01


@MODELS.register_module()
class DualTeacherWrapper(BaseModel):
    """双教师(EMA) + 多强视图一致性 + 结构先验 的封装。
    - 适配 mmengine 标准接口：loss(self, inputs, data_samples)
    - 支持 unsup_only=True 仅无监督；False 时对 L 计算监督损失
    - 教师/学生 dtype 可分离（教师半精度省显存）
    """
    def __init__(self,
                 student: Dict[str, Any],
                 num_keypoints: int,
                 semi_cfg: Dict[str, Any]):
        super().__init__()
        self.semi_cfg = dict(semi_cfg) if semi_cfg is not None else {}
        self.num_kpt = num_keypoints

        # 构建 student / 两位 teacher（同结构）
        from mmengine.registry import MODELS as _M
        self.student = _M.build(student)
        self.teacher = _M.build(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()  # 教师常驻 eval()，更省/更稳

        self.data_preprocessor = self.student.data_preprocessor

        # dtype 策略（推荐：教师 FP32，学生按需 BF16/FP32）
        t_dtype = torch.float32  # 强制教师 FP32 更稳
        s_dtype = torch.float32
        self.teacher.to(dtype=torch.float32).eval()
        self.student.to(dtype=s_dtype)

        # EMA 参数
        self.momentum = float(self.semi_cfg.get('ema_momentum', 0.996))

        # 无监督权重与先验
        self.lambda_u   = float(self.semi_cfg.get('lambda_u', 1.0))
        self.lambda_topo= float(self.semi_cfg.get('lambda_topo', 0.05))
        self.lambda_bone= float(self.semi_cfg.get('lambda_bone', 0.05))
        self.lambda_angle=float(self.semi_cfg.get('lambda_angle', 0.02))
        self.temperature= float(self.semi_cfg.get('temperature', 1.0))
        self.beta       = float(self.semi_cfg.get('beta', 0.5))
        self.M          = int(self.semi_cfg.get('num_strong_views', 2))
        self.edges      = self.semi_cfg.get('edges', [])
        self.triplets   = self.semi_cfg.get('angle_triplets', [])
        self.ref_bone_lengths = self.semi_cfg.get('ref_bone_lengths', [1.0]*len(self.edges))
        self.ins_thresh = float(self.semi_cfg.get('instance_thresh', 0.70))
        self.weak_max_deg = float(self.semi_cfg.get('weak_max_deg', 10))
        self.strong_max_deg = float(self.semi_cfg.get('strong_max_deg', 25))
        self.cons_warmup_iters = semi_cfg.get('consistency_warmup_iters', 0)
        self.min_keep_ratio    = semi_cfg.get('min_keep_ratio', 0.0)
        self.debug_log_interval = semi_cfg.get('debug_log_interval', 50)

        self.lambda_u_ramp_iters   = int(self.semi_cfg.get('lambda_u_ramp_iters', -1))
        self.lambda_u_ramp_updates = int(self.semi_cfg.get('lambda_u_ramp_updates', -1))
        self.stats_warmup_iters    = int(self.semi_cfg.get('stats_warmup_iters', -1))
        self.stats_warmup_updates  = int(self.semi_cfg.get('stats_warmup_updates', -1))
        self.cons_warmup_updates   = int(self.semi_cfg.get('consistency_warmup_updates', -1))



        # 结构先验损失
        self.topo_loss = LaplacianTopoLoss(self.edges, self.lambda_topo)
        self.bone_loss = BoneLengthLoss(self.edges, self.ref_bone_lengths, self.lambda_bone)
        self.angle_loss= JointAngleLoss(self.triplets, self.lambda_angle)

        # 关键点置信统计
        self.register_buffer('kpt_mu', torch.zeros(self.num_kpt))
        self.register_buffer('kpt_sigma', torch.ones(self.num_kpt))
        self.register_buffer('kpt_n', torch.tensor(0))

    def init_weights(self):
        """仅初始化 student，然后把 teacher 严格拷成 student，并冻结为推理态。"""
        # 1) 先初始化 student（按其自身配置加载 ckpt/预训练）
        if hasattr(self.student, 'init_weights'):
            self.student.init_weights()

        # 2) 再把 teacher 拷成 student（不要再单独 init teacher）
        sd = self.student.state_dict()
        self.teacher.load_state_dict(sd, strict=True)

        # 3) 冻结 + eval（并确保 BN 统计一致）
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        # （buffers 已在 state_dict 里一起拷了，这里只是稳妥起见）
        for (nt, bt), (ns, bs) in zip(self.teacher.named_buffers(),
                                    self.student.named_buffers()):
            if bt.shape == bs.shape:
                bt.data.copy_(bs.data)

    # ========== 监督接口 ==========
    def extract_feat(self, batch_inputs):
        return self.student.extract_feat(batch_inputs)
    
    def _denorm01(self, x):
        """把 data_preprocessor 归一化过的张量还原到 [0,1] 像素域（RGB）"""
        dp = self.data_preprocessor
        device, dtype = x.device, x.dtype
        mean = torch.tensor(dp.mean, device=device, dtype=dtype).view(1, -1, 1, 1)  # 0-255
        std  = torch.tensor(dp.std,  device=device, dtype=dtype).view(1, -1, 1, 1)  # 0-255
        x255 = x * std + mean               # 回到 0-255
        return (x255 / 255.0).clamp(0.0, 1.0)

    def _renorm(self, x01):
        """把 [0,1] 像素域张量重新归一化成模型期望的 ((img-mean)/std)"""
        dp = self.data_preprocessor
        device, dtype = x01.device, x01.dtype
        mean = torch.tensor(dp.mean, device=device, dtype=dtype).view(1, -1, 1, 1)  # 0-255
        std  = torch.tensor(dp.std,  device=device, dtype=dtype).view(1, -1, 1, 1)  # 0-255
        x255 = (x01 * 255.0).clamp(0.0, 255.0)
        return (x255 - mean) / std



    # 替换原先的 forward（参数名是 batch_inputs 的那段）
    def forward(self, inputs, data_samples=None, mode='tensor'):
        """遵循 BaseModel 规范：根据 mode 路由到 tensor/predict/loss。"""
        if mode == 'loss':
            # 训练路径：交给 self.loss 计算（会同时支持无监督/半监督）
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # 推理路径：直接复用学生模型的 predict
            return self.student(inputs, data_samples, mode='predict')
        elif mode == 'tensor':
            # 张量路径：用于获取学生的热图张量
            return self.student(inputs, data_samples, mode='tensor')
        else:
            raise RuntimeError(f'Invalid mode "{mode}", expected one of ["tensor","predict","loss"].')
    
    # ----- 1) 在类里增加 -----
    def _get_global_step(self):
        # A. 先尝试 Runner（不同 mmengine 版本兼容）
        try:
            from mmengine.runner import Runner
            getter = getattr(Runner, 'get_current', None) or getattr(Runner, 'get_instance', None)
            if getter is not None:
                return int(getter().iter)
        except Exception:
            pass

        # B. 再尝试 MessageHub（注意用 get_info，而不是 get_scalar）
        try:
            from mmengine.logging import MessageHub
            mh = getattr(MessageHub, 'current', None)
            if mh is None:
                mh = getattr(MessageHub, 'get_current_instance')  # 旧版本
            hub = mh()
            val = hub.get_info('iter')
            if val is not None:
                return int(val)
        except Exception:
            pass

        # C. 最后兜底：用“模型本地计数”（持久 buffer，每次 forward 自增）
        import torch
        if not hasattr(self, '_local_step'):
            # 注册为 buffer，resume 时会随 checkpoint 恢复
            self.register_buffer('_local_step', torch.zeros((), dtype=torch.long), persistent=True)
        self._local_step += 1
        return int(self._local_step.item())
    
    def _get_steps(self):
        """返回 (cur_iter, update_step, accum_steps)
        - cur_iter: 按 batch 计的全局步 (0,1,2,...)
        - update_step: 按“参数更新”计的全局步（考虑梯度累积）
        - accum_steps: 梯度累积步数
        """
        # A) batch 步：沿用你写好的 _get_global_step
        cur_iter = int(self._get_global_step())

        # B) 推断/读取梯度累积步数
        accum_steps = int(self.semi_cfg.get('accum_steps', 0) or 0)
        if accum_steps <= 0:
            try:
                from mmengine.runner import Runner
                getter = getattr(Runner, 'get_current', None) or getattr(Runner, 'get_instance', None)
                ow = getter().optim_wrapper
                for name in ('accumulative_counts', 'accumulation_steps', 'update_interval'):
                    if hasattr(ow, name):
                        val = getattr(ow, name)
                        if isinstance(val, (list, tuple)):
                            val = val[0]
                        accum_steps = int(val)
                        break
            except Exception:
                pass
        if accum_steps <= 0:
            accum_steps = 1

        update_step = cur_iter // accum_steps
        return cur_iter, update_step, accum_steps




    # ========== 核心：训练损失 ==========
    def loss(self, inputs, data_samples=None, **kwargs):
        """mmengine 标准签名。
        支持两种 dataloader：
        - 监督 L：gt 在 data_samples 中（若未设置 unsup_only，则计算监督损失）
        - 无监督 U：仅有 inputs（本函数内构造 weak/strong 视图与一致性/先验损失）
        """
        from torch.cuda.amp import autocast
        import torch
        losses = {}

        # 1) 监督分支（若允许且存在标注）
        if (not self.semi_cfg.get('unsup_only', False)) and (data_samples is not None):
            sup_losses = self.student.loss(inputs, data_samples)
            # 只合并，不覆盖同名键
            for k, v in sup_losses.items():
                if k in losses:
                    losses[f'sup.{k}'] = v
                else:
                    losses[k] = v

        # 2) 无监督视图构建（weak_geom / weak_app / strong_views / affine_mats）
        U = self._build_unsup_views(inputs)

        # 3) 教师预测（弱增）—— 单教师、bf16 autocast，参数保持 fp32
        with torch.no_grad():
            use_cuda = inputs.is_cuda
            with autocast(enabled=use_cuda, dtype=torch.bfloat16):
            # with autocast(enabled=False):
                out_g = self.teacher(U['weak_geom'], data_samples=None, mode='tensor')  # (B,K,Hm,Wm)
                out_a = self.teacher(U['weak_app'],  data_samples=None, mode='tensor')  # (B,K,Hm,Wm)
            # 置信：取每关键点热图峰值
            conf_g = out_g.flatten(2).max(dim=-1).values   # (B,K)
            conf_a = out_a.flatten(2).max(dim=-1).values   # (B,K)
            # 融合双教师预测（几何/外观弱增）
            tea_hm, tea_conf = fuse_teachers(out_g, conf_g, out_a, conf_a,
                                 T=self.temperature,
                                 spatial_tau=getattr(self, 'teacher_tau', 0.5),
                                 sharpen=True, conf_from='prob_max')

            # 释放中间，降低峰值显存
            del out_g, out_a, conf_g, conf_a

        # 4) 一致性损失（多强视图）
        cons_losses = []
        stu_out_last = None

        # 4.1 动态掩码：支持热身与最小保留比例
        #    - 热身阶段：掩码全 1
        #    - 非热身：根据动态统计与 beta 生成掩码；再按 min_keep_ratio 做保底
        # 获取当前迭代（多重兜底）
        # cur_iter = 0
        # try:
        #     from mmengine.logging import MessageHub
        #     mh = MessageHub.get_current_instance()
        #     # 不同版本 API 兼容
        #     if hasattr(mh, 'get_scalar'):
        #         cur_iter = int(mh.get_scalar('iter').current())
        #     else:
        #         cur_iter = int(mh.get_info('iter', 0))
        # except Exception:
        #     cur_iter = int(getattr(self, 'iter', 0))
        # --- 获取全局步数（global step）---
        cur_iter, update_step, accum_steps = self._get_steps()


        # —— 统计/一致性热身（优先按“参数更新步”）——
        cons_warmup_updates = int(self.semi_cfg.get('consistency_warmup_updates', -1))
        stats_warmup_updates = int(self.semi_cfg.get('stats_warmup_updates', -1))
        cons_warmup_iters   = int(self.semi_cfg.get('consistency_warmup_iters', -1))
        stats_warmup_iters  = int(self.semi_cfg.get('stats_warmup_iters', -1))

        in_cons_warmup = (update_step < cons_warmup_updates) if cons_warmup_updates >= 0 else (
                        (cur_iter   < cons_warmup_iters)   if cons_warmup_iters   >= 0 else False)
        in_stats_warmup = (update_step < stats_warmup_updates) if stats_warmup_updates >= 0 else (
                        (cur_iter   < stats_warmup_iters)   if stats_warmup_iters  >= 0 else False)

        if in_cons_warmup:
            mask = tea_conf.new_ones(tea_conf.shape)  # (B,K)
            stats = {'mu': self.kpt_mu, 'sigma': self.kpt_sigma, 'n': int(self.kpt_n.item())}
        else:
            stats = {'mu': self.kpt_mu, 'sigma': self.kpt_sigma, 'n': int(self.kpt_n.item())}
            mask, stats = dynamic_kpt_mask(
                tea_conf, stats,
                beta=self.beta,
                momentum=float(self.semi_cfg.get('momentum', 0.9)),
                min_keep_ratio=float(self.semi_cfg.get('min_keep_ratio', 0.0)),
                percentile=self.semi_cfg.get('percentile', None),
            )


        # 最小保留比例（避免 mask 全 0）
        min_keep_ratio = getattr(self, 'min_keep_ratio', 0.0)
        if min_keep_ratio > 0:
            Bm, Km = mask.shape
            k_min = max(1, int(Km * min_keep_ratio))
            topk = torch.topk(tea_conf, k=k_min, dim=1).indices  # (B, k_min)
            base = torch.zeros_like(mask)
            base.scatter_(1, topk, 1.0)
            mask = torch.maximum(mask, base)

        # 回写统计缓存
        self.kpt_mu = stats['mu']
        self.kpt_sigma = stats['sigma']
        self.kpt_n = torch.tensor(stats['n'], device=self.kpt_mu.device)

        # 强视图循环
        for i in range(len(U['strong_views'])):
            x_i = U['strong_views'][i]      # (B,C,Hs,Ws)
            A_i = U['affine_mats'][i]       # (B,2,3)
            # 学生张量前向（AMP 由外层 Runner 管理）
            stu_out = self.student(x_i, data_samples=None, mode='tensor')  # (B,K,Hs,Ws)
            # 掩码广播到热图空间
            m = mask.unsqueeze(-1).unsqueeze(-1)  # (B,K,1,1)



            # # 教师热图 warp 到该强视图下（内部已做 fp32 逆与 batch 对齐）
            # tea_to_i = warp_heatmaps_affine(tea_hm, A_i, out_size=stu_out.shape[-2:])  # (B,K,Hs,Ws)
            # # MSE 一致性（掩码内）
            # tau = getattr(self, 'prob_temperature', 0.5)  # < 1 会锐化
            # B, K, Hs, Ws = stu_out.shape

            # # 学生: log-prob 作为 input
            # log_p_s = F.log_softmax(stu_out.view(B, K, -1) / tau, dim=-1)  # (B,K,Hs*Ws)

            # # 教师: prob 作为 target —— 注意：softmax 后再“严格归一化”，避免 warp 破坏和为1
            # p_t = F.softmax(tea_to_i.view(B, K, -1) / tau, dim=-1)        # (B,K,Hs*Ws)
            # p_t = p_t / (p_t.sum(dim=-1, keepdim=True) + 1e-12)           # ★ 再归一化
            # p_t = p_t.clamp_min(1e-12)                                    # ★ 防 0

            tea_to_i = warp_heatmaps_affine(tea_hm, A_i, out_size=stu_out.shape[-2:])  # (B,K,Hs,Ws)

            B, K, Hs, Ws = stu_out.shape
            tau = float(getattr(self, 'prob_temperature', 0.5))

            # 学生：log-prob
            log_p_s = F.log_softmax(stu_out.view(B, K, -1) / tau, dim=-1)

            # 教师：fuse_teachers 已返回“概率图”；只做归一化，千万不要再 softmax 或除以 tau
            p_t = tea_to_i.view(B, K, -1)
            p_t = p_t / (p_t.sum(dim=-1, keepdim=True) + 1e-12)
            p_t = p_t.clamp_min(1e-12)


            # KL(input=log_prob, target=prob) —— 非负
            kl_per = F.kl_div(log_p_s, p_t, reduction='none')             # (B,K,Hs*Ws)
            kl_per = kl_per.sum(dim=-1)                                    # (B,K) 按空间求和
            kl_per = kl_per / math.log(Hs * Ws + 1e-12)   # ★ 归一到 log(HW)，与分辨率无关


            # 掩码归一化后加权
            m = mask                                                       # (B,K)∈{0,1}
            m_norm = m / (m.sum(dim=1, keepdim=True) + 1e-6)               # 每样本归一
            cons = (kl_per * m_norm).sum(dim=1).mean()                     # 标量 ≥ 0

            cons_losses.append(cons)
            stu_out_last = stu_out  # 最后一个强视图供结构先验使用

        loss_cons = sum(cons_losses) / max(1, len(cons_losses))

        # 5) 结构先验（在最后一个强视图上）
        #    soft-argmax → 像素坐标（热图分辨率）→ 三类先验损失
        assert stu_out_last is not None, 'strong_views 为空，无法计算结构先验。'
        coords = soft_argmax_2d(stu_out_last.detach()).float()  # (B,K,2) in [-1,1]
        B = coords.size(0)

        # —— 由关节掩码生成边/三元组掩码 —— #
        edge_mask = None
        if len(self.edges) > 0:
            edge_mask = torch.stack([mask[:, u] * mask[:, v] for (u, v) in self.edges], dim=-1)  # (B,E)

        triplet_mask = None
        if len(self.triplets) > 0:
            triplet_mask = torch.stack([mask[:, i] * mask[:, j] * mask[:, k] for (i, j, k) in self.triplets], dim=-1)  # (B,T)

        # —— 在[-1,1]坐标上计算先验（结构类已支持掩码与尺度不变） —— #
        loss_topo  = self.topo_loss(coords, mask_edges=edge_mask)
        loss_bone  = self.bone_loss(coords, mask_edges=edge_mask)
        loss_angle = self.angle_loss(coords, mask_triplets=triplet_mask)


        # —— λ_u ramp-up（优先按“更新步”，无配置则恒为 1.0）——
        lambda_u_ramp_updates = int(self.semi_cfg.get('lambda_u_ramp_updates', -1))
        lambda_u_ramp_iters   = int(self.semi_cfg.get('lambda_u_ramp_iters', -1))
        if lambda_u_ramp_updates > 0:
            t = min(1.0, update_step / float(lambda_u_ramp_updates))
        elif lambda_u_ramp_iters > 0:
            t = min(1.0, cur_iter / float(lambda_u_ramp_iters))
        else:
            t = 1.0
        lambda_u_eff = float(self.lambda_u) * t


        # 6) 汇总无监督损失
        loss_unsup = lambda_u_eff * loss_cons + loss_topo + loss_bone + loss_angle
        losses.update({
            'loss_unsup': loss_unsup,
            'loss_cons': loss_cons.detach(),
            'loss_topo': loss_topo.detach(),
            'loss_bone': loss_bone.detach(),
            'loss_angle': loss_angle.detach(),
        })

        # 7) （可选）调试日志：掩码与置信
        dbg_int = getattr(self, 'debug_log_interval', 0)
        if dbg_int > 0 and (cur_iter % dbg_int == 0):
            try:
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                # logger.info(f'[semi] iter={cur_iter} mask_mean={mask.float().mean().item():.3f} '
                #             f'tea_conf(m/mx)={tea_conf.mean().item():.3f}/{tea_conf.max().item():.3f} '
                #             f'cons={loss_cons.item():.5f}')
                logger.info(f'[semi] iter={cur_iter} upd={update_step} mask_mean={mask.float().mean().item():.3f} '
                            f'tea_conf(m/mx)={tea_conf.mean().item():.5f}/{tea_conf.max().item():.5f} '
                            f'cons={loss_cons.item():.5f}')

            except Exception:
                pass

        return losses


    # ========== 训练时每步调用：更新教师 EMA ==========
    # def _update_ema(self):
    #     with torch.no_grad():
    #         ms, mt = self.student.state_dict(), self.teacher.state_dict()
    #         for k in mt.keys():
    #             if k in ms and mt[k].shape == ms[k].shape:
    #                 mt[k].copy_(mt[k] * self.momentum + ms[k].to(dtype=mt[k].dtype) * (1.0 - self.momentum))

    @torch.no_grad()
    def _update_ema(self):
        m = float(self.momentum)
        # 参数做 EMA
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            if p_t.shape == p_s.shape:
                p_t.data.mul_(m).add_(p_s.data.to(dtype=p_t.dtype), alpha=1.0 - m)
        # BN running_* 直接同步（不要 EMA）
        for (nt, bt), (ns, bs) in zip(self.teacher.named_buffers(), self.student.named_buffers()):
            if 'running_' in nt and bt.shape == bs.shape:
                bt.data.copy_(bs.data)

        # # ===== EMA UPDATE (after optimizer.step()) =====
        # with torch.no_grad():
        #     m = float(self.momentum)  # 你的配置里写的那个
        #     # dbg: 打印一次 final_layer 的范数（sentinel）
        #     try:
        #         name_t, Wt = next((n, p) for n, p in self.teacher.named_parameters() if 'final_layer.weight' in n)
        #         name_s, Ws = next((n, p) for n, p in self.student.named_parameters() if 'final_layer.weight' in n)
        #         print(f'[ema dbg] ||W_t||={float(Wt.norm()):.4f} ||W_s||={float(Ws.norm()):.4f} m={m}')
        #     except StopIteration:
        #         pass

        #     for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
        #         # 关键：方向不能反、并用 FP32 做加权
        #         p_t.data = p_t.data * m + p_s.data.float() * (1.0 - m)

        # # 老师要保持 eval()，防止 BN 跑动态统计
        # self.teacher.eval()
        # # ==============================================



    # def train_step(self, data, optim_wrapper):
    #     # 复用 BaseModel 的默认实现，但在每步后更新教师
    #     outputs = super().train_step(data, optim_wrapper)
    #     self._update_ema()
    #     return outputs

    def train_step(self, data, optim_wrapper):
        # 让 mmengine 完成一次标准的前/反传与（可能的）参数更新
        outputs = super().train_step(data, optim_wrapper)

        # —— 仅在“参数更新步”之后更新教师 EMA —— #
        did_step = False
        # 兼容不同 mmengine 版本：inner_count 在一次 step 后通常会被清零
        for name in ('inner_count', '_inner_count'):
            if hasattr(optim_wrapper, name):
                inner = int(getattr(optim_wrapper, name))
                # inner==0：刚完成一次 step（或没有累积机制）
                did_step = (inner == 0)
                break
        if not did_step:
            # 兜底：按 update_interval/accumulative_counts 判断
            accum = 1
            for key in ('accumulative_counts', 'accumulation_steps', 'update_interval'):
                if hasattr(optim_wrapper, key):
                    accum = getattr(optim_wrapper, key)
                    if isinstance(accum, (list, tuple)):
                        accum = int(accum[0])
                    else:
                        accum = int(accum)
                    break
            # 用全局步数兜底（见 _get_steps）
            cur_iter, _, _ = self._get_steps()
            did_step = ((cur_iter + 1) % max(1, accum) == 0)

        if did_step:
            self._update_ema()

        return outputs


    # ========== 在线构造 U 的弱/强视图 ==========
    def _build_unsup_views(self, inputs):
        """inputs: (B,C,H,W)，值域[0,1]（由 DataPreprocessor 处理）"""
        x = inputs
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # 先回到 [0,1] 像素域做增强
        x01 = self._denorm01(inputs)

        # 几何弱增（给 geom-teacher）
        A_g = _rand_affine_matrix(B, max_deg=self.weak_max_deg, scale_range=(0.95,1.05),
                                translate=0.02, device=device, dtype=x01.dtype)
        weak_geom01 = self._apply_affine(x01, A_g)        # 注意：_apply_affine 会改成 align_corners=False

        # 外观弱增（给 app-teacher）：只做轻微颜色扰动（像素域），不 clamp 到 0/1 之外
        noise_b = (torch.rand(B,1,1,1, device=device, dtype=x01.dtype)*0.08 - 0.04)   # 亮度±0.04
        noise_c = (torch.rand(B,1,1,1, device=device, dtype=x01.dtype)*0.20 - 0.10)   # 对比度±0.10
        weak_app01 = (x01 + noise_b) * (1.0 + noise_c)
        weak_app01 = weak_app01.clamp(0.0, 1.0)

        # 多强视图（学生）
        strong_views, affine_mats = [], []
        # 先把 A_g 扩成 3×3 并求逆，后面要用到
        eye = torch.eye(3, device=device, dtype=x01.dtype).unsqueeze(0).repeat(B,1,1)
        Ag33 = eye.clone(); Ag33[:, :2, :] = torch.cat(
            [A_g, torch.tensor([0,0,1], device=device, dtype=x01.dtype).view(1,1,3).repeat(B,1,1)],
            dim=1)[:, :2, :]
        Ag_inv = torch.inverse(Ag33)

        for _ in range(self.M):
            A = _rand_affine_matrix(B, max_deg=self.strong_max_deg, scale_range=(0.85,1.15),
                                    translate=0.06, device=device, dtype=x01.dtype)
            x_i01 = self._apply_affine(x01, A)
            x_i01 = _strong_augment01(x_i01)        # 新版强增强：像素域，不在归一化域里 clamp
            strong_views.append(self._renorm(x_i01)) # ← 回到归一化域再喂学生

            # 严格的 teacher→strong：A_strong @ inv(A_weak_geom)
            A33 = eye.clone(); A33[:, :2, :] = torch.cat(
                [A, torch.tensor([0,0,1], device=device, dtype=x01.dtype).view(1,1,3).repeat(B,1,1)],
                dim=1)[:, :2, :]
            A_ts = torch.bmm(A33, Ag_inv)[:, :2, :]
            affine_mats.append(A_ts)

        # 返回给 loss() 的 teacher 输入也要再归一化
        weak_geom = self._renorm(weak_geom01)
        weak_app  = self._renorm(weak_app01)

        return dict(weak_geom=weak_geom, weak_app=weak_app,
                    strong_views=strong_views, affine_mats=affine_mats)


    def _apply_affine(self, x, A):
        """把 2x3 像素仿射应用于张量图像 x（先归一化处理见 geometry.warp 逻辑）"""
        B, C, H, W = x.shape
        # 复用 warp_heatmaps_affine 的实现：把通道当关键点=1 处理
        # 这里简单重写一份 grid，不依赖 K 维
        import torch.nn.functional as F
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B,1,1)
        eye[:, :2, :] = torch.cat([A, torch.tensor([0,0,1], device=x.device, dtype=x.dtype).view(1,1,3).repeat(B,1,1)], dim=1)[:, :2, :]
        inv = torch.inverse(eye)[:, :2, :]
        sx = 2.0 / max(W - 1, 1)
        sy = 2.0 / max(H - 1, 1)
        theta = inv.clone()
        theta[:,0,:] *= sx
        theta[:,1,:] *= sy
        # grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=True)
        # return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

