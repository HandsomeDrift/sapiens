from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmengine.registry import MODELS  # 若你更偏好 mmpose 的注册树，可改为: from mmpose.registry import MODELS

from ..utils.geometry import warp_heatmaps_affine
from ..utils.pseudo_label import fuse_teachers, dynamic_kpt_mask
from .losses.structural_priors import soft_argmax_2d, LaplacianTopoLoss, BoneLengthLoss, JointAngleLoss


def _parse_dtype(name: str):
    """将字符串映射到 torch.dtype."""
    name = str(name).lower()
    if name in ('fp16', 'float16', 'half'):
        return torch.float16
    if name in ('bf16', 'bfloat16'):
        return torch.bfloat16
    # 默认使用 fp32
    return torch.float32


@MODELS.register_module()
class DualTeacherWrapper(BaseModel):
    """把现有 Sapiens 学生模型包装为：双教师(EMA) + 多强视图一致性 + 结构先验 的半/无监督学习器."""

    def __init__(self,
                 student: Dict[str, Any],
                 num_keypoints: int,
                 semi_cfg: Dict[str, Any]):
        super().__init__()
        # 保存 semi_cfg，避免后续访问 self.semi_cfg 报错
        self.semi_cfg = dict(semi_cfg) if semi_cfg is not None else {}

        # ===== 构建 student / 两位 teacher =====
        from mmengine.registry import MODELS as _M  # 如用 mmpose.registry，可改为 from mmpose.registry import MODELS as _M
        self.student = _M.build(student)
        self.geom_teacher = _M.build(student)
        self.app_teacher = _M.build(student)
        for p in list(self.geom_teacher.parameters()) + list(self.app_teacher.parameters()):
            p.requires_grad_(False)
        self.num_kpt = num_keypoints

        # ===== OOM 方案一：在 CPU 上先把教师参数转半精度（或 BF16），学生可选保持 FP32 =====
        t_dtype = _parse_dtype(self.semi_cfg.get('teacher_param_dtype', 'fp16'))   # 默认教师半精度
        s_dtype = _parse_dtype(self.semi_cfg.get('student_param_dtype', 'fp32'))   # 默认学生 FP32（数值更稳）
        # 这里转换发生在 CPU 上；随后 runner.model.to(cuda) 时会按该 dtype 拷到 GPU，显著降低参数显存
        self.geom_teacher.to(dtype=t_dtype)
        self.app_teacher.to(dtype=t_dtype)
        self.student.to(dtype=s_dtype)

        # ===== 损失与超参 =====
        self.lambda_u = self.semi_cfg.get('lambda_u', 1.0)
        self.lambda_topo = self.semi_cfg.get('lambda_topo', 0.05)
        self.lambda_bone = self.semi_cfg.get('lambda_bone', 0.05)
        self.lambda_angle = self.semi_cfg.get('lambda_angle', 0.02)
        self.temperature = self.semi_cfg.get('temperature', 1.0)
        self.beta = self.semi_cfg.get('beta', 0.5)
        self.M = self.semi_cfg.get('num_strong_views', 2)
        self.edges = self.semi_cfg.get('edges', [])
        self.triplets = self.semi_cfg.get('angle_triplets', [])
        self.ref_bone_lengths = self.semi_cfg.get('ref_bone_lengths', [1.0] * len(self.edges))
        self.ins_thresh = self.semi_cfg.get('instance_thresh', 0.70)

        # 结构先验损失模块
        self.topo_loss = LaplacianTopoLoss(self.edges, self.lambda_topo)
        self.bone_loss = BoneLengthLoss(self.edges, self.ref_bone_lengths, self.lambda_bone)
        self.angle_loss = JointAngleLoss(self.triplets, self.lambda_angle)

        # 关键点置信统计（运行时动态阈值所需）
        self.register_buffer('kpt_mu', torch.zeros(self.num_kpt))
        self.register_buffer('kpt_sigma', torch.ones(self.num_kpt))
        self.register_buffer('kpt_n', torch.tensor(0))

    def extract_feat(self, batch_inputs):
        # 委托给 student
        return self.student.extract_feat(batch_inputs)

    def forward(self, batch_inputs, data_samples=None, mode='tensor'):
        # 兼容 BaseModel，但具体训练逻辑在 loss() 里
        return self.student(batch_inputs, data_samples, mode)

    def loss(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # ========== 监督分支（可选，unsup_only=True 时跳过） ==========
        sup_losses: Dict[str, torch.Tensor] = {}
        L = batch_data.get('inputs')
        L_samples = batch_data.get('data_samples')
        if not self.semi_cfg.get('unsup_only', False) and (L is not None):
            sup_losses = self.student.loss(L, L_samples)
        else:
            sup_losses = {}

        # ========== 无监督分支 ==========
        U = batch_data.get('U', None)
        if U is None:
            return sup_losses

        # 教师预测（弱增）
        weak_g = U['weak_geom']   # (B, C, H, W)
        weak_a = U['weak_app']
        with torch.no_grad():
            out_g = self.geom_teacher(weak_g, mode='tensor')  # (B,K,Hm,Wm)
            out_a = self.app_teacher(weak_a, mode='tensor')
            # 置信：用每关键点热图峰值作为简化置信
            conf_g = out_g.flatten(2).max(dim=-1).values  # (B,K)
            conf_a = out_a.flatten(2).max(dim=-1).values
            tea_hm, tea_conf = fuse_teachers(out_g, conf_g, out_a, conf_a, T=self.temperature)

        # 多强视图一致性
        strong_list = U['strong_views']        # List[(B,C,Hs,Ws)]，长度 M
        T_mats = U['affine_mats']              # List[(B,2,3)] 教师->学生各强视图的仿射
        cons_losses: List[torch.Tensor] = []
        stu_out = None  # 用于结构先验坐标
        for i in range(len(strong_list)):
            stu_out = self.student(strong_list[i], mode='tensor')  # (B,K,Hs,Ws)
            tea_to_i = warp_heatmaps_affine(tea_hm, T_mats[i], out_size=stu_out.shape[-2:])
            # 动态逐点掩码
            stats = {'mu': self.kpt_mu, 'sigma': self.kpt_sigma, 'n': int(self.kpt_n.item())}
            mask, stats = dynamic_kpt_mask(tea_conf, stats, beta=self.beta)
            # 回写统计
            self.kpt_mu = stats['mu']
            self.kpt_sigma = stats['sigma']
            self.kpt_n = torch.tensor(stats['n'], device=self.kpt_mu.device)
            # 一致性 (按掩码)
            m = mask.unsqueeze(-1).unsqueeze(-1)
            cons = ((stu_out - tea_to_i) ** 2 * m).mean()
            cons_losses.append(cons)

        loss_cons = sum(cons_losses) / max(1, len(cons_losses))

        # 结构先验（在学生强视图 0/最后一次输出上计算）
        # 注意：这里只约束坐标，不把约束梯度回传到学生热图（detach）
        coords = soft_argmax_2d(stu_out.detach())  # (B,K,2)
        loss_topo = self.topo_loss(coords)
        loss_bone = self.bone_loss(coords)
        loss_angle = self.angle_loss(coords)

        # 汇总
        unsup = self.lambda_u * loss_cons + loss_topo + loss_bone + loss_angle
        sup_losses['loss_unsup'] = unsup
        sup_losses['loss_cons'] = loss_cons.detach()
        sup_losses['loss_topo'] = loss_topo.detach()
        sup_losses['loss_bone'] = loss_bone.detach()
        sup_losses['loss_angle'] = loss_angle.detach()
        return sup_losses
