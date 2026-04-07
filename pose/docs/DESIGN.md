# Hyper-Sapiens 系统设计文档

> 本文档整合了项目历次设计迭代（SAGE-Pose → NK-Sapiens → Geo-Sapiens → Hyper-Sapiens），以 Hyper-Sapiens 为最终方向，作为后续开发的统一技术指导。

---

## 1. 项目背景与目标

### 1.1 临床需求

本项目服务于**婴幼儿脑瘫早期筛查**。临床"黄金标准"是**全身运动评估（GMA）**——通过观察 0–5 月龄婴儿的自发性"不安运动"（Fidgety Movements）来判断神经发育状态。这类运动幅度小、频率高、伴随细微旋转，现有的人工评估难以大规模普及。

本项目的目标是构建一个**自动化的婴幼儿 2D 姿态估计系统**，为 GMA 提供客观的量化分析基础。

### 1.2 核心技术挑战

1. **解剖学漂移**：婴儿头身比约 1:4（成人 1:7.5），肢体短小、关节被皮下脂肪掩盖，四肢柔韧性极高（可呈现脚趾触碰面部等极端姿态）。
2. **姿态流形偏移**：婴儿以仰卧/俯卧为主，缺乏直立姿态的重力约束，运动模式为无目的的全身运动，与成人动作库完全不同。
3. **数据极度匮乏**：公开真实婴儿标注数据仅约 700 张（SyRIP），合成数据存在严重的 Sim-to-Real 鸿沟。自采数据需经"模型预标注 → 人工审核"流程，产出速度有限。
4. **自遮挡与环境干扰**：婴儿蜷缩姿态导致严重自遮挡，且肢体与毯子/衣物纹理高度相似。

### 1.3 基础模型选择

采用 **Meta Sapiens**（ECCV 2024 Best Paper Candidate）作为基础模型：
- ViT 架构，参数量 0.3B–2B，在 3 亿张人类图像上 MAE 预训练
- 原生支持 1024×1024 高分辨率推理
- 多任务能力：Pose、Depth、Normal、Segmentation

Sapiens 的优势在于强大的通用人体特征提取能力和多任务协同潜力；劣势在于预训练数据以成人为主，直接应用于婴儿会产生显著退化。

---

## 2. 设计演变回顾

### 2.1 SAGE-Pose（V1，已实现）

**核心思路**：双教师 EMA + 多强视图一致性 + 结构先验

- 几何教师 + 外观教师，按扰动类型分工（实际实现为单教师）
- 学生对同一样本走多路强增强，对齐融合教师目标
- 骨架图正则：拓扑（Laplacian）+ 骨长比例 + 关节角约束

**已实现的代码**（`pose/semisup/`）：
- `DualTeacherWrapper`（单教师 EMA）+ 一致性损失（KL/JS）+ 动态关键点掩码
- 结构先验三件套（`LaplacianTopoLoss`、`BoneLengthLoss`、`JointAngleLoss`）
- 无标注数据加载（`UnlabeledCocoTopDownDataset`）+ 热图仿射变换

**局限**：纯 2D 像素级一致性，无几何深度感知，无法利用 Sapiens 多任务能力，面向通用场景而非婴儿 Sim-to-Real。

### 2.2 NK-Sapiens（V2，未实现）

**关键跳跃**：首次引入临床叙事、DDP、CMGC 概念

- 首次提出扩散发育先验（DDP）和跨模态几何一致性（CMGC）
- Hybrid Teacher（EMA + DDP 选择）+ KPS 评分
- 假设 20k 标注 + 100k 无标注（偏理想化）

**贡献**：概念框架的奠基。**局限**：数据假设脱离实际条件。

### 2.3 Geo-Sapiens（V3，未实现）

**关键转变**：从理想化 SSL 回归 Sim-to-Real 小样本现实

- 回到 SyRIP + MINI-RGBD 的实际数据条件
- Depth/Normal 头冻结为"几何批评家"，提出具体的 2.5D 骨长约束和法线-表面对齐公式
- 先验从扩散退回 VAE

**贡献**：现实约束的确立。**局限**：VAE 先验不如扩散模型。

### 2.4 Hyper-Sapiens（V4，当前方向）

**最终融合**：吸收所有前序版本的最佳设计

- 从 NK-Sapiens 拿回扩散先验，升级为 **SDS 损失**（有 MAP 贝叶斯理论支撑）
- 超越 Geo-Sapiens：Depth/Normal 头**可训练**（多任务联合学习）
- 系统化的**数据分层战略**
- 成熟的**三阶段课程学习**

---

## 3. Hyper-Sapiens 框架设计

### 3.1 核心理念

重新定义合成数据的角色——不仅是数据增强的手段，更是连接解剖学结构与视觉表征的"语义桥梁"。通过**扩散驱动先验（DDP）**约束预测落入合法婴儿姿态流形，通过**多头几何解码器**的联合训练强制 2D 关键点与潜在 3D 表面结构保持一致。

微调过程建模为**最大后验概率估计（MAP）**：

$$p(\mathbf{x} | I_{real}) \propto \underbrace{p(I_{real} | \mathbf{x})}_{\text{似然（监督损失）}} \cdot \underbrace{p(\mathbf{x})}_{\text{先验（DDP）}}$$

### 3.2 总体架构

```
输入图像
    │
    ▼
┌──────────────────────────────┐
│  Sapiens Backbone (1B/2B)    │  ← 共享编码器，LoRA 参数高效微调（rank=8, ~0.2% 可训参数）
│  ViT + MAE Pre-trained       │     或渐进解冻（底层冻结，顶部 3-6 层可训）
│  输入分辨率: 1024×1024       │
└──────────┬───────────────────┘
           │
     ┌─────┼──────────┐
     ▼     ▼          ▼
  Pose   Depth     Normal       ← 多头几何解码器（均为轻量反卷积头）
  Head   Head      Head
     │     │          │
     │     │          │
     ▼     ▼          ▼
  关键点  相对深度图  法线向量图
  热图    (1024²)    (1024²×3)
     │     │          │
     │     └────┬─────┘
     │          │
     ▼          ▼
  Pose       几何一致性约束
  Prediction  (深度-法线一致性 + 骨长比例一致性)
     │
     ▼
  DDP 先验约束 ←── 冻结的骨架扩散模型（独立预训练）
```

### 3.3 模块一：LoRA 参数高效微调

Sapiens-1B 有 ~1.5B 参数，全量微调顶部 6 层需训练 ~226M 参数（15%），在小样本下极易过拟合。采用 LoRA（Low-Rank Adaptation）对 ViT 的 attention 层进行低秩分解微调。

**目标层**：所有 Transformer Block 的 `attn.qkv`（Q/K/V 合并投影）和 `attn.proj`（输出投影）

**参数效率**（rank=8 示例）：
- 每层 LoRA 参数：~73K（qkv: 49K + proj: 24K）
- 全部 40 层总计：~2.95M 可训练参数（占总参数的 **0.2%**）
- 对比全量微调：参数量降低 **~50 倍**，显存占用降低 **~3 倍**

**实现**：仓库已有 `LoRAModel` 和 `LoRALinear`（`pretrain/mmpretrain/models/peft/lora.py`），通过 config 中 `backbone=dict(type='mmpretrain.LoRAModel', module=dict(...), alpha=16, rank=8, targets=[...])` 接入。需适配 `LayerDecayOptimWrapperConstructor` 以识别 LoRA 参数的命名路径。

**大模型关联**：与 LLM 领域的 LoRA 微调完全同构——同样的数学原理（$W' = W_0 + BA \cdot \alpha/r$）、同样的工程实践（冻结基座、仅训练低秩矩阵）、同样的参数规模（1B+）。

### 3.4 模块二：多头几何解码器

| 解码器 | 任务 | 输出 | 初始化 | 训练数据 | 作用 |
|--------|------|------|--------|---------|------|
| Pose Head | 关键点热图 | K 个高斯热图 | 随机/Sapiens-Pose | SyRIP Syn/Real + 自采 | 核心任务输出 |
| Depth Head | 稠密深度估计 | 相对深度图 | Sapiens-Depth | MINI-RGBD / SyRIP Syn | 几何一致性约束 |
| Normal Head | 表面法线预测 | 法线向量图 | Sapiens-Normal | MINI-RGBD / SyRIP Syn | 辅助遮挡与光照处理 |

**关键设计**：即使最终只需 Pose Head 输出，Depth/Normal Head 在训练阶段的联合学习能迫使共享编码器学习更本质的 3D 结构特征。在阶段一（合成数据）上三头全监督训练；在阶段二/三（真实数据无深度标注）上，Depth/Normal 头通过几何一致性损失相互约束。

**大模型关联**：共享编码器被迫产出对三种模态（骨架、深度场、表面法线）均有效的特征表示，这与 CLIP/BLIP 的跨模态对齐原理一致——不同模态的信息在统一特征空间中对齐。

### 3.5 模块三：扩散驱动先验（DDP）

**目标**：学习合法婴儿姿态的概率分布 $p_{prior}(\mathbf{x})$，作为动态正则化项。

**架构**：轻量级残差 MLP + 时间步嵌入的去噪网络（DDPM），输入为归一化骨架坐标（K×2）。以臀部为中心、尺度归一化，确保平移和尺度不变性。

**训练**：仅使用私有数据 / 标注数据中的骨架坐标序列，独立于主训练流程，训练成本极低。

**使用**（主训练时冻结，通过 SDS 损失接入）：
1. Pose Head 预测姿态 $\hat{\mathbf{x}}$
2. 添加随机噪声 $\epsilon \sim \mathcal{N}(0, \sigma_t^2 I)$，得到加噪姿态 $\mathbf{x}_t$
3. 冻结的 DDP 模型预测噪声 $\hat{\epsilon} = \epsilon_\theta(\mathbf{x}_t, t)$
4. 计算 SDS 损失梯度：

$$\nabla_{\hat{\mathbf{x}}} \mathcal{L}_{DDP} = w(t)(\hat{\epsilon} - \epsilon)$$

**物理含义**：将 Pose Head 的预测推向扩散模型学到的"高概率密度区域"。当模型因遮挡或模糊预测出反关节或怪异姿态时，DDP 产生惩罚梯度，强制预测回归合法姿态流形。

**大模型关联**：DDP 本质上是一个**奖励模型（Reward Model）**——它评估预测姿态的"合理性"并提供梯度信号，与 RLHF 中奖励模型的角色同构。SDS 损失源自 DreamFusion，是 Score-based 生成式 AI 的核心技术。

### 3.6 模块四：几何一致性约束

在真实图像（无深度标注）上，通过 Depth/Normal Head 的输出之间的物理关系构建无监督约束。

**深度-法线一致性损失**：

$$\mathcal{L}_{geo\_cons} = \| \mathbf{n}_{pred} - \mathbf{n}_{derived}(D_{pred}) \|_1$$

$\mathbf{n}_{pred}$ 为 Normal Head 直接预测的法线，$\mathbf{n}_{derived}(D_{pred})$ 为对 Depth Head 预测的深度图微分计算得到的法线。两者应一致。

**尺度不变骨长比例一致性**：

$$\mathcal{L}_{bone\_ratio} = \sum_{b \in Bones} \left\| \frac{L_b}{L_{ref}} - r_{syn} \right\|^2$$

利用 Depth Head 将 2D 关键点反投影回 3D 空间，计算骨骼长度比例，与合成数据统计的标准比例 $r_{syn}$ 对齐。规避了单目深度的尺度模糊性。

### 3.7 模块五：VLM 姿态质量评估（VLM-as-Reward-Model）

利用视觉语言模型（如 Qwen2-VL）作为姿态预测的质量评估器，为伪标签提供多模态反馈信号。

**架构**：离线评估模块（不在训练循环内），输入为"原始图像 + 预测骨架叠加图"，输出为结构化质量评分。

**流程**：
1. 渲染预测关键点为骨架叠加图
2. 构造结构化 prompt（评估解剖合理性、关节可见性、整体质量）
3. 调用 Qwen2-VL 推理，解析 JSON 评分
4. 标量化为 [0,1] 质量分数，存为伪标签元数据
5. 训练时按质量分数加权伪标签样本

**应用场景**：
- 阶段二/三的伪标签过滤：仅保留 VLM 评分高于阈值的伪标签
- 自采数据的标注质量审计：批量检查人工标注的一致性

**大模型关联**：这是 **VLM-in-the-Loop** 的工程落地——将多模态大模型集成到视觉任务的数据流水线中，作为质量控制的"评审专家"。其角色等同于 RLHF 管线中的 Reward Model，从多模态视角（图像+骨架）评估输出质量。

---

## 4. 数据分层战略

每类数据有明确的战略角色，而非简单混合训练。

### 4.1 角色分配

| 数据 | 角色 | 战略价值 |
|------|------|---------|
| **SyRIP 合成** (1,000) | 语义桥梁 | 标签精确（数学绝对准确）；遮挡透明化（可获得被遮挡关键点真值） |
| **SyRIP 真实** (~700) | 纹理锚点 | 校正模型对真实光照/材质的响应；域适应终点 |
| **MINI-RGBD** (12,000) | 几何放大器 | 提供精确深度/法线 GT，训练辅助几何头；扩展姿态变化和视角多样性 |
| **私有/自采数据** | 流形守护者 | 构建 DDP 扩散先验，定义"什么是合法的婴儿姿态" |

### 4.2 可用数据集详情

| 数据集 | 类型 | 规模 | 格式状态 |
|--------|------|------|---------|
| COCO 2017 | 成人标注 | 118K train + 5K val | ✅ 就绪 |
| SyRIP | 婴儿合成+真实 | 1,000 合成 + ~700 真实 | ✅ 已转 COCO17（`syrip_coco17/`） |
| MINI-RGBD | 婴儿合成(RGB-D) | 12×1000 帧 | ✅ 已转 COCO17（`mini_rgbd_coco17/`） |
| 自采(stand) | 真实预标注 | 73 受试者, 45,752 帧 | ⚠️ 需 labelme→COCO 转换 |
| 自采(cruising) | 真实人工审核 | 30 受试者, 7,914 帧 | ⚠️ 需 labelme→COCO 转换 |

### 4.3 数据转换流水线

自采数据：视频 6fps 抽帧 → PNG → 模型预标注（`*_labelme_init_coco17.json`）→ 人工审核修正 → `labelme_to_coco_pose17.py` 转 COCO 格式 → 检测器生成 `person_detection_results/*.json`

---

## 5. 训练策略：三阶段课程学习

### 5.1 阶段一：合成数据热身与几何对齐

- **数据**：100% SyRIP Synthetic + MINI-RGBD
- **目标**：将 Sapiens 特征提取器适配到婴儿解剖结构，训练辅助几何头
- **损失**：$\mathcal{L}_{Stage1} = \mathcal{L}_{pose}^{syn} + \lambda_d \mathcal{L}_{depth}^{syn} + \lambda_n \mathcal{L}_{normal}^{syn}$
- **策略**：三头全监督训练（合成数据有完美 GT）；渐进解冻 ViT 顶层 3–6 层
- **产出**：模型理解婴儿骨骼拓扑和基本 3D 几何，但对真实纹理一无所知

### 5.2 阶段二：桥接与约束迁移

- **数据**：混合批次（每 batch 含合成 + 真实图像）
- **目标**：保持几何结构知识的同时，适应真实纹理分布
- **损失**：

$$\mathcal{L}_{Stage2} = \underbrace{\mathcal{L}_{pose}^{syn} + \mathcal{L}_{pose}^{real}}_{\text{监督项}} + \alpha \underbrace{\mathcal{L}_{DDP}(\hat{\mathbf{x}}_{real})}_{\text{流形约束}} + \beta \underbrace{\mathcal{L}_{geo\_cons}(\hat{D}_{real}, \hat{N}_{real})}_{\text{几何约束}}$$

- **策略**：
  - 真实数据：开启 DDP 先验 + 几何一致性（无深度标注，靠跨模态一致性自监督）
  - 合成数据：继续强监督，防止遗忘婴儿解剖结构（抗灾难性遗忘）
- **建议权重**：$\alpha=0.01$, $\beta=0.1$

### 5.3 阶段三：真实域精调

- **数据**：80% SyRIP Real + 20% SyRIP Synthetic（重放缓冲）
- **目标**：最大化真实场景精度，处理长尾分布
- **策略**：
  - 逐渐降低 DDP 权重 $\alpha$（退火），允许模型学习偏离先验的罕见真实姿态
  - 保持几何一致性，确保物理合理性
  - 可引入自采数据中人工审核部分，进一步增加真实域多样性

### 5.4 消融矩阵

所有消融通过 config 开关控制（λ=0 即可关闭）：

| 编号 | 配置 | 验证目标 |
|------|------|---------|
| A1 | 仅阶段一（合成监督） | Baseline：合成域性能上限 |
| A2 | + 阶段二（无 DDP） | 几何一致性单独贡献 |
| A3 | + 阶段二（无几何约束） | DDP 单独贡献 |
| A4 | + 阶段二（DDP + 几何） | 两者交互效果 |
| A5 | + 阶段三（完整流程） | Full Hyper-Sapiens |

---

## 6. 损失函数体系

### 6.1 全阶段损失汇总

| 损失 | 数据来源 | 描述 | 阶段 |
|------|---------|------|------|
| $\mathcal{L}_{pose}$ | 合成/真实标注 | 热图 MSE（Pose Head） | 1, 2, 3 |
| $\mathcal{L}_{depth}$ | 合成 (MINI-RGBD) | 深度估计监督 | 1 |
| $\mathcal{L}_{normal}$ | 合成 (MINI-RGBD) | 法线预测监督 | 1 |
| $\mathcal{L}_{DDP}$ | 真实（无标注） | SDS 扩散先验正则 | 2, 3 |
| $\mathcal{L}_{geo\_cons}$ | 真实（无标注） | 深度-法线跨模态一致性 | 2, 3 |
| $\mathcal{L}_{bone\_ratio}$ | 真实（无标注） | 尺度不变骨长比例约束 | 2, 3 |

### 6.2 从 SAGE-Pose 可选继承的结构先验

作为几何一致性的补充，在训练初期可启用：

| 损失 | 描述 | 建议权重 |
|------|------|---------|
| $\mathcal{L}_{topo}$ | 图 Laplacian 拓扑约束 | 0.02 |
| $\mathcal{L}_{bone}$ | 2D 骨长比例约束 | 0.02 |
| $\mathcal{L}_{angle}$ | 关节角合理性约束 | 0.01 |

---

## 7. 工程约束

### 7.1 显存管理

- 骨干：Sapiens-1B（首选）或 Sapiens-0.6B（资源受限时）
- 输入分辨率：1024×1024（必须，婴儿手指/脚趾等小目标依赖高分辨率）
- 激活检查点：`with_cp=True`（必须）
- 混合精度：前向 bfloat16，几何/扩散计算强制 float32
- 梯度累积：有效 batch size ≥ 64
- 渐进式分辨率：训练初期可用较低分辨率，后期提升

### 7.2 数值精度

以下计算**必须**在 float32 下进行（显式关闭 autocast 或 `.float()` 转换）：
- 几何一致性损失（坐标变换、开方、归一化）
- DDP 扩散采样与 SDS 梯度
- 仿射矩阵求逆（`warp_heatmaps_affine`）

### 7.3 依赖与环境

- 必须使用仓库内嵌的定制 `mmpretrain`（含 Sapiens ViT 定义）
- PYTHONPATH 必须优先指向仓库：
  ```bash
  export PYTHONPATH=$SAPIENS_ROOT:$SAPIENS_ROOT/pretrain:$SAPIENS_ROOT/pose:$PYTHONPATH
  ```
- Python 3.10, PyTorch 2.2.2 + CUDA 12.1, mmengine 0.10.7, mmcv 2.1.0

---

## 8. 代码组织（规划）

### 8.1 现有代码（可复用）

```
pose/semisup/
├── models/dual_teacher_wrapper.py     # EMA 框架 → 可作为阶段二/三的 Teacher 基础
├── models/losses/structural_priors.py # 拓扑/骨长/关节角 → 保留作为补充约束
├── data/unlabeled_coco_topdown.py     # 无标注数据加载 → 直接复用
├── utils/geometry.py                  # 热图仿射变换 → 直接复用
└── utils/pseudo_label.py              # 教师融合 + 动态掩码 → 可复用
```

### 8.2 已实现模块

```
pose/projects/hyper_sapiens/
├── __init__.py
├── engine/
│   └── lora_layer_decay_constructor.py    # ✅ LoRA 优化器构造器（适配 layer decay + LoRA 参数分组）
├── rl/                                    # ✅ GRPO 姿态群组优化（对接 DeepSeek-R1 GRPO）
│   ├── reward.py                          #    复合解剖学奖励函数（骨长 + 关节角 + 对称性）
│   └── grpo_pose.py                       #    热图采样 + group advantage + 策略梯度损失
├── agents/                                # ✅ 多 Agent 标注质量管线（工具调用 + 协作决策）
│   ├── tools.py                           #    PoseDetectionTool / VLMAssessmentTool / StructuralValidationTool
│   ├── orchestrator.py                    #    QualityOrchestrator（三 Agent 协调 + 决策聚合）
│   └── run_quality_pipeline.py            #    CLI 入口
├── models/                                # ✅ 多头跨模态解码器
│   ├── hyper_wrapper.py                   #    主 wrapper: LoRA backbone + 3 heads + 课程学习调度
│   ├── depth_head.py                      #    相对深度预测头（SILog 损失）
│   └── normal_head.py                     #    表面法线预测头（cosine 损失）
├── ddp_prior/                             # ✅ 扩散驱动先验
│   ├── ddp_model.py                       #    骨架 DDPM（残差 MLP + 时间嵌入）
│   └── sds_loss.py                        #    Score Distillation Sampling 损失
├── losses/                                # ✅ 几何一致性约束
│   └── geo_consistency.py                 #    深度-法线一致性 + 尺度不变骨长比例
├── tools/
│   ├── verify_lora.py                     # ✅ LoRA 集成验证
│   └── compute_skeleton_stats.py          # ✅ 骨架统计先验计算
├── stats/
│   └── skeleton_stats_coco17.json         # ✅ COCO17 骨架统计（从 262K 标注计算）
└── configs/                               # 待编写
    ├── lora_rank8_sapiens1b_stage1.py
    ├── stage1_synthetic_warmup.py
    ├── stage2_bridged_transfer.py
    └── stage3_real_refinement.py
```

---

## 9. 评估指标

| 指标 | 说明 | 适用场景 |
|------|------|---------|
| COCO AP | 标准 OKS 指标 | 主指标 |
| PCK@0.05 | 头部尺寸标准化的关键点正确率 | 精度分析 |
| G-Error | 视频序列中肢体长度方差 | 几何一致性评估 |
| Sim-to-Real Gap Reduction | $(Acc_{ours} - Acc_{baseline}) / (Acc_{oracle} - Acc_{baseline})$ | 迁移效果量化 |

---

## 10. 验收标准

### 工程验收

- 阶段一：合成数据上可训练，50 iter loss 非 NaN，三头输出形状正确
- 阶段二：混合训练可跑通，DDP 损失和几何一致性损失正常下降
- 阶段三：真实域精调可收敛

### 研究验收

- 至少完成消融矩阵中 A1（baseline）与 A5（full）的对比
- 消融实验可通过 config 开关直接复现
- 每次实验记录：config 路径、ckpt 路径、seed、val 指标

---

## 附录 A：设计版本对照

| 维度 | SAGE-Pose (V1) | NK-Sapiens (V2) | Geo-Sapiens (V3) | **Hyper-Sapiens (V4)** |
|------|----------------|-----------------|-------------------|----------------------|
| 一致性机制 | 热图 MSE/KL | Hybrid Teacher+KPS | Mean-Teacher | **MAP + SDS** |
| 几何约束 | 2D 结构先验 | CMGC（冻结头） | 冻结几何批评家 | **可训练多头解码器** |
| 姿态先验 | 无 | 骨架扩散(L2) | VAE 流形 | **骨架扩散(SDS)** |
| 数据假设 | 通用 SSL | 20k标注(理想) | SyRIP+MINI(现实) | **分层战略(最优)** |
| 训练策略 | 2 阶段 | 2 阶段+分步开启 | 3 阶段 | **3 阶段课程学习** |
| 理论基础 | 工程驱动 | 框架级设计 | 损失公式 | **MAP 贝叶斯** |
| 实现状态 | ✅ 已实现 | ❌ | ❌ | 📋 **当前方向** |

## 附录 B：大模型技术映射

| 本项目组件 | 大模型对应概念 | 面试叙事 |
|-----------|--------------|---------|
| **GRPO 姿态群组优化** | **RL 对齐 / GRPO** | "将 DeepSeek-R1 的 GRPO 引入视觉任务，热图采样+解剖学奖励+group advantage" |
| **多 Agent 质量管线** | **Agent + Tool Use** | "三 Agent 协作（Pose/VLM/Structural），原生 function calling，自动化伪标签筛选" |
| LoRA on Sapiens ViT (1B) | PEFT / 参数高效微调 | "与 LLM LoRA 同构，0.2% 可训参数，50 倍参数压缩" |
| 多头 Pose+Depth+Normal | 跨模态表征对齐 | "类 CLIP 原理，共享编码器对齐三种模态" |
| DDP 扩散先验 + SDS 损失 | Score-based 生成式先验 / Reward Model | "SDS 源自 DreamFusion，扩散模型作为动态奖励信号" |
| VLM 姿态质量评估 | VLM-in-the-Loop / 多模态应用 | "集成 Qwen2-VL 作为多模态 Reward Model" |
| 三阶段课程学习 | Foundation Model 训练策略 | "类 Pre-train → SFT → RLHF 的渐进式适配" |
| 整体项目 | 低资源基础模型领域适配 | "1B 参数视觉基础模型在 <1000 样本下的 Sim-to-Real 迁移" |

**简历描述（中文）**：

"基于 Meta Sapiens 视觉基础模型（1B 参数）的领域适配框架。核心创新：(1) 将 GRPO 群组相对优化引入视觉姿态估计，通过解剖学奖励函数实现无需人工标注的 RL 对齐；(2) 设计多 Agent 数据质量控制系统（Pose Agent + VLM Agent + Structural Validator），基于工具调用与协作决策实现自动化伪标签筛选；(3) LoRA 高效微调（0.2% 可训参数）+ 跨模态几何对齐（Pose/Depth/Normal 多任务学习）+ 扩散生成式先验（SDS 损失）。在极小样本（<1000 真实标注）下实现高效 Sim-to-Real 迁移。"

## 附录 C：合成数据增强策略

为最大化合成数据的"桥梁"作用：
- **纹理随机化**：强色彩抖动、高斯模糊、噪声注入，使合成数据在低频特征上接近真实数据的"不完美"
- **几何保留**：避免破坏几何一致性的增强（如剧烈弹性形变），除非同步变换深度图和法线图
- **Cut-Occlude**：随机粘贴纹理补丁到婴儿关键部位，模拟真实场景的遮挡
