# SPEC.md — NK-Sapiens (Stage0+Stage1) 可执行规格（mmengine/mmpose 版）

## 0. 范围与结论先行

### 0.1 目标
在现有 Sapiens pose（mmpose/mmengine、heatmap 监督）基础上，实现 NK-Sapiens 的 **Stage0（监督热身）+ Stage1（半监督对齐）**，并支持可复现实验与消融：
- Baseline（L_sup + L_cons）
- +DDP（L_dev）
- +CMGC（L_geo_depth + L_geo_normal）
- +DDP + CMGC
- Full（+Hybrid Teacher gating：EMA Teacher A + DDP Teacher B）

### 0.2 明确不做（本里程碑）
- Stage2（抓握 fine-grained / zoom-in crops）暂不实现，仅保留“未来可选”附录。
- 不在 20k 上微调 YOLOv8 检测器：先使用通用检测器生成 bbox JSON，并通过 config 的 `bbox_file` 接入。

### 0.3 代码库现实假设（必须对齐）
- 配置系统：mmengine `.py` config（非 YAML）。
- 训练入口：`python pose/tools/train.py <config.py>`
- 多卡入口：`bash pose/tools/dist_train.sh <config.py> <gpus>`
- 测试入口：`python pose/tools/test.py <config.py> --checkpoint <ckpt>`
- 无 dummy dataset：smoke 必须依赖真实数据，并在 dataset config 中用 `indices=[...]` 抽子集跑通闭环。
- pose 输出契约：`HeatmapHead` 输出 `B x K x H x W` heatmaps；`TopdownPoseEstimator(mode='loss')` 返回 `(losses, preds)` 且 `preds` 为热图；`mode='predict'` 输出 `PoseDataSample`。

---

## 1. 交付物（Deliverables）

### 1.1 新增/修改配置（必须）
新增目录（建议）：
- `pose/configs/nk_sapiens/infant/`

至少包含：
- `stage0_supervised.py`：从现有 Sapiens COCO config 继承并改成你的 infant 数据集/路径。
- `stage1_ssl_baseline.py`：Stage1，但只启用 L_sup + L_cons（先跑通 SSL 框架）。
- `stage1_ssl_full.py`：Stage1 完整 NK-Sapiens（DDP + CMGC + Hybrid gating），并可通过开关做消融。

### 1.2 新增代码模块（必须）
建议使用 OpenMMLab 的 project 方式，放在：
- `pose/projects/nk_sapiens/`

并在 config 使用 `custom_imports` 引入。

模块最小集合：
1) **SSL 数据与双视图**
- Dataset/Wrapper：支持 labeled + unlabeled（COCO-style）同时进入训练。
- Transform：生成 weak/strong 两个视图；必须支持 `num_strong_views=1`（默认 1）。

2) **Teacher 系统**
- EMA Teacher A：参数滑动平均更新。
- Hybrid gating：基于 conf + KPS 选择 A 的硬伪标签或启用 Teacher B（DDP 引导）。

3) **DDP（Diffusion Developmental Prior）**
- `ddp_prior/train_ddp_prior.py`：用 20k 标注 keypoints（coords）训练骨架扩散模型。
- `ddp_prior/ddp_prior.py`：推断接口（给定 student 预测的 coords 或 heatmap 解码 coords，输出去噪/修正结果或 soft guidance）。
- 约束：DDP 输出不得作为“硬伪标签”，只能作为结构正则或引导信号（L_dev）。

4) **CMGC（Cross-Modal Geometric Consistency）**
- `cmgc/geometric_teacher.py`：从 `seg` 分支加载冻结的 depth/normal 模型（或两模型），在 pose 训练 step 内前向得到：
  - depth：`B x 1 x H x W`
  - normal：`B x 3 x H x W`
- `cmgc/losses.py`：实现 `L_geo_depth` 与 `L_geo_normal`（见第 4 节定义）。

5) **统计工具（KPS / ROM / limb stats）**
- `tools/compute_kps_stats.py`：从 20k labeled 计算 limb 长度分布与 ROM 范围，输出 `stats/skeleton_stats.json`。

### 1.3 文档（必须）
- `docs/nk_sapiens/SPEC.md`（本文）
- `docs/nk_sapiens/PROTOCOL.md`
- `docs/nk_sapiens/AGENTS.md`

---

## 2. 数据契约（Data Contract）

### 2.1 数据切分（必须固定）
- `train_labeled`：20k（COCO keypoints）
- `train_unlabeled`：100k（COCO-style images + bbox；无 keypoints）
- `val`：从标注集中固定划分（强制固定 seed 或固定 id 列表）
- `test`：若有则独立；若没有，至少保证 val 不参与阈值/超参调优后再评估（保持可复现）

### 2.2 标注格式（必须 COCO-style）
- labeled：COCO keypoints JSON（含 keypoints, visibility 等）
- unlabeled：COCO-style JSON（至少 images；建议也有 annotations 的 bbox 或外部 bbox_file）
- top-down：通过 config 指定 `bbox_file`（检测结果 JSON）

### 2.3 关键点定义（必须从 dataset_info 读取）
- K、keypoint 名称、flip、skeleton edges、sigmas 等由 `pose/configs/_base_/datasets/*.py` 的 `dataset_info` 提供，并通过 `metainfo=dict(from_file=...)` 引用。
- CMGC 与 KPS 必须使用同一份 skeleton edges。

---

## 3. 模型与集成方式（Architecture Spec）

### 3.1 Student（pose）
- 仍使用 mmpose 的 `TopdownPoseEstimator + HeatmapHead` 结构（不重写 backbone/head）。
- Stage0：标准监督训练（heatmap MSE 由 head.loss 提供）。

### 3.2 Geometric Teacher（depth/normal，冻结，来自 seg）
由于 pose 分支默认单 head，depth/normal 在 seg 分支单独实现，因此本项目采用 “跨分支冻结教师”：
- 在 pose 训练 step 中，对同一 batch 的输入（与 pose 同尺度/同裁剪策略）调用几何教师前向，得到 depth/normal 预测。
- 教师参数 `eval()` 且 `requires_grad_(False)`。
- 教师的输入预处理必须与 pose 对齐（推荐使用与 pose 相同的 1024x768 级别输入；如你现有 config 是 1024x768，则保持一致）。

---

## 4. 损失函数（Loss Spec）

总损失（Stage1）：
\[
L_{\text{total}} =
L_{\text{sup}} +
\lambda_{\text{cons}} L_{\text{cons}} +
\lambda_{\text{dev}} L_{\text{dev}} +
\lambda_{\text{geoD}} L_{\text{geo\_depth}} +
\lambda_{\text{geoN}} L_{\text{geo\_normal}}
\]
所有 λ 均必须由 config 控制，并支持置 0 做消融。

### 4.1 L_sup（监督）
- 在 labeled 上：沿用 HeatmapHead 默认的 heatmap MSE（或 UDPHeatmap 的编码/解码配置）。

### 4.2 L_cons（一致性，半监督基本盘）
- 输入：unlabeled 的 weak/strong 两视图。
- teacher：EMA Teacher A 在 weak 上输出 heatmap（或 coords），student 在 strong 上输出。
- 形式：heatmap MSE 或在 heatmap 上的 KL（任选其一，但要在 config 中固定）。
- 伪标签过滤：最小版本可用 confidence 阈值（peak conf / keypoint_scores）；完整版本见 Hybrid gating（第 5 节）。

### 4.3 L_dev（DDP 扩散先验正则）
- 目标：把 student 预测拉回“幼儿合法姿态流形”，但不把 DDP 输出当硬标签。
- 实现形式（示例）：
  - 从 student heatmap 解码得到 coords \(y\)
  - DDP prior 对 \(y\) 做去噪得到 \( \hat{y} \)（或输出 score/残差）
  - 定义 \(L_{\text{dev}} = \|y - \hat{y}\|^2\)（或与 score matching 形式等价）
- 关键约束：
  - L_dev 的计算必须在 float32 下进行（见第 6 节数值硬约束）。

### 4.4 L_geo_depth（深度刚性约束）
- 输入：几何教师输出 depth \(D\) 与 pose coords（由 heatmap 解码）；
- 思路：对每条 limb (i,j)，结合 2D 像素距离与 depth 差，构造近似 3D 长度或“深度补偿长度”，与统计得到的平均 limb 长度对齐。
- 统计参数：来自 `stats/skeleton_stats.json`（第 5.2 节）。

### 4.5 L_geo_normal（法向一致性约束）
- 输入：几何教师输出 normal \(N\) 与 pose limb 向量；
- 思路：limb 向量与局部体表法向满足几何关系（例如近似正交/约束角度），违反则惩罚。
- 注：该项可先实现最小可用版本（例如 limb 向量与采样点 normal 的点积惩罚），后续再迭代更精细的采样策略。

---

## 5. Hybrid Teacher（EMA + DDP，动态选择）

### 5.1 Teacher A（EMA）
- 维护 teacher 权重为 student 的 EMA：
  - 每个 iter 更新一次（或每 N iter）
  - momentum 由 config 指定（如 0.999 或 warmup）

### 5.2 KPS（Kinematic Plausibility Score）
KPS 用于判断 Teacher A 输出是否“可信且合理”。KPS 由三部分组成（权重可配）：
- `score_conf`：由 Teacher A 的 keypoint_scores 或 heatmap peak 置信度得到
- `score_len`：limb 长度 z-score（基于 labeled 统计均值/方差）
- `score_rom`：关节角是否落在 ROM 合理区间（基于 labeled 统计分位数区间）

统计来源：
- 使用 `tools/compute_kps_stats.py` 在 20k labeled 上计算并固化为 `stats/skeleton_stats.json`。

### 5.3 Teacher B（DDP 引导）
- 当 Teacher A 置信度低或 KPS 不合理时，启用 Teacher B：
  - Teacher B 不是独立网络：它由 DDP prior 对当前预测做修正/去噪得到。
- Teacher B 的输出不得作为硬伪标签写入监督项；它只能进入 L_dev 或 soft guidance。

### 5.4 选择规则（必须实现且可统计）
对每个 unlabeled 样本（可先做 sample-level，后续可拓展到 keypoint-level）：
- 若 `conf >= thr_conf` 且 `kps >= thr_kps`：使用 Teacher A 的硬伪标签（进入 L_cons）
- 否则：启用 Teacher B（进入 L_dev / 或给 L_cons 提供 soft target）

必须在日志中记录：
- `num_use_teacherA`
- `num_use_teacherB`

---

## 6. 工程硬约束（必须遵守）

1) 显存：Stage1 默认 `num_strong_views=1`，否则不保证可跑。
2) AMP：允许前向 bfloat16/amp，但以下计算必须强制 float32：
   - CMGC（涉及坐标变换/开方/归一化等）
   - DDP prior 采样/去噪与 L_dev
   实现方式：在这些 loss 计算块内显式关闭 autocast 或 `.float()` 转换。
3) 依赖：不得替换仓库内嵌的定制 mmpretrain；必须沿用 repo 自带版本（否则 Sapiens 架构可能无法识别）。

---

## 7. 验收标准（Acceptance Criteria）

### 7.1 工程验收（必须）
- Stage0：
  - `python pose/tools/train.py pose/configs/nk_sapiens/infant/stage0_supervised.py` 能启动
  - 用 `indices` 抽样跑 50 iter：loss 非 NaN
- Stage1（baseline 先过，再 full）：
  - `python pose/tools/train.py pose/configs/nk_sapiens/infant/stage1_ssl_baseline.py` 能启动
  - `python pose/tools/train.py pose/configs/nk_sapiens/infant/stage1_ssl_full.py` 能启动
  - 50 iter smoke：loss 非 NaN，且日志里 teacherA/teacherB 选择统计可见
- 测试：
  - `python pose/tools/test.py <config> --checkpoint <ckpt>` 在 val 上输出指标

### 7.2 研究验收（必须可复现）
- 至少完成并记录：
  - Baseline（L_sup + L_cons）
  - Full（DDP + CMGC + Hybrid）
- 其余消融（+DDP、+CMGC、+DDP+CMGC）可分批，但必须通过 config 开关可直接复现。

---

## 8. 未来可选：Stage2（不在本里程碑）
若后续决定做 Stage2：
- 在 Stage1 收敛后：冻结 backbone，仅解冻 pose head + 最后两层 transformer block
- 使用更高分辨率的 zoom-in crops 针对手部关键点多轮微调
- 需额外定义：手部 ROI 生成策略、hand AP 指标与对照实验
