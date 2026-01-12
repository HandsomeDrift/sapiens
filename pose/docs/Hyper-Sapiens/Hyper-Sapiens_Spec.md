# Hyper-Sapiens 规格说明（COCO17，扩散先验 + 几何一致性）

状态：草案  
负责人：Codex  
更新时间：2025-02-14  

## 1. 目标
- 基于新方案（DDP 扩散先验 + 几何一致性）制定可落地的实现规格。
- 统一关键点空间为 COCO17，兼容现有 Sapiens pose checkpoint。
- 以三阶段课程学习实现 Sim-to-Real 迁移：合成对齐 → 混合桥接 → 真实精调。
- 不复用 `pose/semisup` 与 `pose/semisup_old` 的实现。

## 2. 非目标
- 不重写 mmpose/mmseg 的核心模块。
- 不更改 COCO 标注格式。
- 不支持 bottom-up 模式。

## 3. 数据角色与分层
- SyRIP Syn：语义桥梁 + 精准监督（Pose/Depth/Normal）。
- SyRIP Real：纹理锚点 + 真实域监督/校正（Pose + DDP/几何约束）。
- MINI-RGBD：几何信号放大器（Depth/Normal 监督）。
- 私有数据：扩散先验（仅骨架序列/关键点），定义合法姿态流形。

## 4. 关键点空间
- 输出与训练统一为 COCO17。
- 25 -> 17 映射需补齐（KeypointConverter）。
- 评估只算 16 个公共点（非公共点 visibility=0 或指标子集化）。

## 5. 模型架构
### 5.1 主干
- Sapiens ViT backbone（当前默认 0.3B，方案建议 1B/2B）。
- 采用分层解冻：初期仅解冻最后 3-6 个 block + 位置编码。

### 5.2 多头解码器
- Pose Head：COCO17 热图回归（HeatmapHead/UDPHeatmap）。
- Depth Head：相对深度估计（VitDepthHead）。
- Normal Head：表面法线预测（VitNormalHead）。
- 三头共享主干，多任务训练增强结构表征。

## 6. 扩散先验（DDP Module）
- 模型：轻量 MLP 去噪网络 + 时间步嵌入（DDPM/Score-based）。
- 训练数据：私有骨架序列（2D 或 3D）。
- 归一化：以骨盆/躯干为中心，尺度归一化，消除平移尺度影响。
- 推理时冻结，用 SDS 损失约束真实样本预测。

## 7. 损失设计
- 合成监督：`L_pose_syn + lambda_d * L_depth_syn + lambda_n * L_normal_syn`
- 真实监督：`L_pose_real`（若真实样本有标注）
- DDP 先验：`L_ddp`（SDS 梯度）
- 几何一致性：
  - Depth-Normal Consistency：`L_geo_cons`
  - Bone Ratio Consistency：`L_bone`（尺度不变比例）

## 8. 三阶段训练流程
### Stage 1：合成热身 + 几何对齐
- 数据：SyRIP Syn + MINI-RGBD（100% 合成）。
- 损失：`L_pose_syn + L_depth_syn + L_normal_syn`

### Stage 2：混合桥接（DDP + 几何）
- 数据：混合批次（默认 1:1，例如 16 syn + 16 real）。
- 损失：`L_pose_syn + L_pose_real + alpha * L_ddp + beta * L_geo_cons`
- 推荐权重：`alpha=0.01, beta=0.1`（可调）。

### Stage 3：真实域精调
- 数据：80% Real + 20% Syn（Replay Buffer）。
- 逐步降低 `alpha`，保持几何一致性。
- 分布式训练建议启用 SyncBN + 梯度累积。

## 9. 分辨率与增强策略
- 输入分辨率按数据集与显存配置，可优先高分辨率（如 1024）。
- 合成数据使用纹理随机化（色彩抖动/噪声/模糊）。
- 避免破坏几何的增强（如弹性形变），除非同步处理深度/法线。

## 10. 实现布局建议
```
pose/hyper_sapiens/
  datasets/
    unlabeled_topdown.py
  models/
    hyper_pose_wrapper.py
    ema_utils.py
  losses/
    geo_losses.py
    ddp_prior.py
  utils/
    heatmap.py
    augment.py
pose/configs/hyper_sapiens/
  stage1_syn_warmup.py
  stage2_mixed_ddp.py
  stage3_real_refine.py
```

## 11. 评估
- 主指标：PCK@0.05（16 公共点）。
- 辅助指标：G-Error（骨长方差，可选）。

## 12. 待确认事项
- 25 -> 17 关键点映射表与 16 公共点索引。
- 真实数据是否提供 keypoints（决定是否启用 `L_pose_real`）。
- 主干规模选择（0.3B vs 1B/2B）。
- Depth/Normal 1B 权重路径。
- 各数据集实际输入分辨率与检测框 JSON 格式。
