# Geo-Sapiens 规格说明（COCO17，按数据集分辨率，Sapiens-0.3B）

状态：草案
负责人：Codex
更新时间：2025-02-14

## 1. 目标
- 在不使用 `pose/semisup` 或 `pose/semisup_old` 的前提下，给出可落地的
  Geo-Sapiens 实现规格。
- 以 COCO17 为统一关键点空间，用合成 + 真实数据并结合几何约束
  (Depth/Normal) 进行婴幼儿姿态估计适配。
- 保持训练流程与 mmengine/mmpose/seg 代码路径兼容。
- 输入分辨率按数据集与任务配置（W x H）。

## 2. 非目标
- 不重写 mmpose/mmseg 的核心模块。
- 不更改 COCO 标注格式。
- 不支持 bottom-up 训练模式。

## 3. 假设与约束
- 姿态关键点统一为 COCO17。
- 25 点数据在输入端转换为 COCO17。
- 评估仅在 16 个公共关键点上进行（剔除 1 个非公共点）。
- Depth/Normal 批评家使用 Sapiens-1B 权重（路径待补充）。
- Stage 1 使用合成标注 + 真实无标注混合训练，比例 2:1。
- 真实数据的检测框 JSON 由用户提供。

## 4. 关键点空间
### 4.1 输出空间
- 训练/推理全部使用 COCO17（K=17）。
- 热图输出维度：`(B, 17, Hm, Wm)`。

### 4.2 映射占位（25 -> 17）
- TODO：给出 25 点到 COCO17 的精确映射表。
- 非公共点通过 `visibility=0` 丢弃。
- 映射通过 `KeypointConverter` 在数据 pipeline 中完成。

### 4.3 评估子集（16 个公共点）
- 通过 visibility mask 屏蔽非公共点。
- PCK@0.05 仅在 16 个公共点上计算。
- 需要时实现子集指标（按索引裁剪预测与 GT）。

## 5. 数据
### 5.1 合成有标注
- MINI-RGBD（COCO17 格式）
- SyRIP-Synth（COCO17 格式）

### 5.2 真实无标注
- SyRIP-Real 图像 + 检测框 JSON（COCO bbox）。
- 无关键点标注，生成 17 点的全 0 可见度伪标注。

### 5.3 少样本真实（可选 Stage 2）
- SyRIP-Real 的少量标注子集（COCO17 格式）。

## 6. 数据流水线（Top-Down）
### 6.1 基础流水线（共享）
- LoadImage
- GetBBoxCenterScale
- TopdownAffine（input_size 随数据集配置）
- PhotometricDistortion
- Albumentation（CoarseDropout, Blur）
- GenerateTarget（UDPHeatmap 或 MSRAHeatmap）
- PackPoseInputs

### 6.2 Stage 0 增强
- ElasticScale：非均匀缩放，扩展婴幼儿比例分布。
- CutOcclude：粘贴纹理补丁模拟遮挡。

### 6.3 无标注样本处理
- 无标注样本返回 dummy keypoints，visibility=0。
- GenerateTarget 生成 0 权重监督目标，确保 `L_sup=0`。
- 通过 `keypoints_visible.sum==0` 识别无标注样本。

## 7. 模型组件
### 7.1 Pose Student
- `TopdownPoseEstimator` + Sapiens-0.3B ViT backbone。
- Head：`HeatmapHead`，out_channels=17。
- 输入尺寸：随数据集配置（W x H）。

### 7.2 Pose Teacher（EMA）
- Student 的 EMA 副本。
- 可用内部 EMA 或 mmengine 的 `EMAHook`。
- Teacher 固定 eval 模式，不反传梯度。

### 7.3 几何批评家（冻结）
- DepthEstimator + VitDepthHead（Sapiens-1B 权重）。
- NormalEstimator + VitNormalHead（Sapiens-1B 权重）。
- 输入尺寸与 Pose 对齐（按数据集配置）。
- 输出 resize 到输入尺寸以便采样关键点位置。

## 8. 损失
### 8.1 监督热图损失
- `L_sup = MSE(heatmap_pred, heatmap_gt)`。
- 由 `GenerateTarget` 提供权重（无标注为 0）。

### 8.2 一致性损失（无标注）
- Teacher 用 weak view，Student 用 strong view。
- 用仿射 grid 将 teacher 热图对齐到 strong view。
- 计算 KL 或 MSE（对归一化热图）。

### 8.3 深度-骨骼一致性（无标注）
- 对左右对称肢体：
  - 用预测关键点采样深度图。
  - 用 (u,v,depth) 计算 2.5D 骨长 D_ij。
  - 惩罚左右骨长不一致。

### 8.4 法线对齐损失（无标注）
- 在关键点处采样法线向量。
- 与关键点法线先验做余弦距离。

### 8.5 幼儿流形先验（可选）
- 用合成 COCO17 关键点训练小型 VAE。
- 以重构误差作为 `L_manifold`。

### 8.6 总损失
- `L_total = L_sup + lambda_cons * L_cons + lambda_geo * (L_geo_depth + L_geo_norm)
  + lambda_manifold * L_manifold`
- 默认权重：`lambda_cons=1.0`，`lambda_geo=0.1`，`lambda_manifold=0.05`。
- Stage 1 早期对无监督项进行 ramp-up。

## 9. 训练阶段
### Stage 0：监督热身
- 仅使用合成标注数据。
- 启用 ElasticScale + CutOcclude。
- 20 epochs，AdamW，backbone 低 LR。

### Stage 1：几何引导 UDA（混合）
- 合成标注 + 真实无标注混合训练。
- `MultiSourceSampler`，比例 2:1（合成:真实）。
- 启用 EMA teacher、一致性、Depth/Normal 损失。
- 30-40 epochs，无监督项 ramp-up。

### Stage 2：少样本真实微调
- 用少量真实标注微调。
- 低 LR，5-10 epochs。

### 可选：测试时自适应（TTA）
- 冻结 backbone，仅优化 head。
- 只用几何损失做少步更新。

## 10. 采样策略
- Wrapper：`CombinedDataset`
- Sampler：`MultiSourceSampler`
- `source_ratio=[2, 1]` 偏向合成稳定性。
- Batch size 要保证两路数据同批次出现。

## 11. 评估
- 主指标：PCK@0.05（16 公共点）。
- 辅助指标：G-Error（骨长方差）。
- 使用 `pose/tools/test.py`，必要时新增自定义 metric。

## 12. 实现目录布局
```
pose/geosapiens/
  datasets/
    unlabeled_topdown.py
  models/
    geo_pose_wrapper.py
    ema_utils.py
  losses/
    geo_losses.py
    manifold_vae.py
  utils/
    heatmap.py
    augment.py
pose/configs/geo_sapiens/
  stage0_sup.py
  stage1_uda.py
  stage2_fewshot.py
```

## 13. 待确认问题
- 25 -> 17 的精确映射表。
- Depth/Normal 1B 权重路径。
- 哪个 COCO17 关键点是非公共点（用于 16 点评估）。
- 真实数据检测框 JSON 的具体格式（默认 COCO）。
