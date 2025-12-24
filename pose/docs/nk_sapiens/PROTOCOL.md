# PROTOCOL.md — NK-Sapiens (Stage0+Stage1) 可复现实验协议

## 0. 统一运行方式（与仓库一致）

### 0.1 训练
- 单卡：
  - `python pose/tools/train.py <config.py>`
- 多卡：
  - `bash pose/tools/dist_train.sh <config.py> <gpus>`
- Slurm（如需要）：
  - `pose/tools/slurm_train.sh`

### 0.2 测试
- `python pose/tools/test.py <config.py> --checkpoint <ckpt>`

### 0.3 Smoke 原则
- 仓库无 dummy dataset：必须使用真实数据。
- 在 config 的 dataset 中使用 `indices=[...]` 抽取小子集（例如 32～128 张）跑通：
  - Stage0：50 iter
  - Stage1：50 iter（覆盖 teacherA/teacherB 分支）

---

## 1. 复现与记录规范（必须）

每次运行必须记录（输出到 work_dir）：
- `config`：运行时 resolved config（mmengine 通常会保存）
- `git commit hash`
- `dataset json` 的路径与 md5（labeled/unlabeled/bbox_file/val）
- `seed`
- `checkpoint` 路径与 best 指标名（如 `save_best='coco/AP'`）

推荐在 `default_hooks` 或自定义 Hook 中写入 `meta.json`。

---

## 2. 数据准备

### 2.1 Labeled（20k）
- COCO-style keypoints JSON（top-down）
- 按你的数据集定义提供 `dataset_info`（K、skeleton、flip 等），并通过 config 的 `metainfo=dict(from_file=...)` 引用。

### 2.2 Unlabeled（100k）
- COCO-style images（可以没有 keypoints）
- 需要 bbox（推荐）：
  - 使用通用检测器生成检测结果 JSON，并在 Stage1 config 中通过 `bbox_file` 接入
- 本里程碑不做 YOLOv8 微调；若发现漏检严重，再作为后续数据工程可选项处理。

---

## 3. 统计先验（KPS / ROM / limb stats）

在开始 Stage1 前，必须先在 labeled 上生成统计文件：
- 运行（示例）：`python pose/projects/nk_sapiens/tools/compute_kps_stats.py --ann <labeled.json> --out stats/skeleton_stats.json`
- 输出内容至少包括：
  - 每条 limb 的长度均值/方差（或分位数）
  - 关键关节角 ROM 合理区间（建议用分位数区间）
  - 关键点置信度统计（可选）

该文件是 Hybrid gating 与 CMGC 的唯一统计来源（可复现）。

---

## 4. DDP prior 训练（独立步骤）

### 4.1 输入
- 来自 labeled 的关键点 coords（由标注直接读取，不经 teacher 生成）

### 4.2 输出
- `priors/ddp_prior_<dataset>_seed<seed>.pth`
- `priors/ddp_prior_<dataset>_seed<seed>.json`（包含归一化参数、噪声日程、采样步数等）

### 4.3 复现要求
- 固定 seed
- 固定骨架归一化方式（写入 json）
- DDP prior 训练完成后才能跑 Stage1 full（baseline 不依赖 DDP prior）

---

## 5. Stage0：监督热身（Supervised Warmup）

### 5.1 config
- `pose/configs/nk_sapiens/infant/stage0_supervised.py`
- 继承 Sapiens pose baseline config（例如 coco 1024x768 版本），主要替换：
  - dataset 路径与 metainfo
  - work_dir 命名
  -（可选）优化器与 with_cp / grad checkpoint 等节省显存策略

### 5.2 运行
- `python pose/tools/train.py pose/configs/nk_sapiens/infant/stage0_supervised.py`

### 5.3 输出
- `work_dirs/<exp>/epoch_*.pth`（或 iter_*.pth）
- `best_*.pth`（若启用 save_best）
- 日志与 tensorboard

---

## 6. Stage1：半监督对齐（SSL Alignment）

Stage1 分两步跑，降低一次性集成风险：

### 6.1 Stage1-baseline（先跑通 SSL 框架）
- config：`stage1_ssl_baseline.py`
- 损失：
  - labeled：L_sup
  - unlabeled：L_cons（EMA teacherA 的一致性）
- 运行：
  - `python pose/tools/train.py pose/configs/nk_sapiens/infant/stage1_ssl_baseline.py`
- 要求：
  - teacherA EMA 更新正常
  - 50 iter smoke 不 NaN

### 6.2 Stage1-full（完整 NK-Sapiens）
- config：`stage1_ssl_full.py`
- 在 baseline 基础上开启：
  - DDP prior（L_dev）
  - CMGC（L_geo_depth + L_geo_normal）
  - Hybrid gating（teacherA/teacherB 选择统计）
- 数值要求（必须）：
  - forward 可 amp/bfloat16
  - CMGC 与 DDP/L_dev 的计算必须强制 float32（关闭 autocast 或转换 float）
- 运行：
  - `python pose/tools/train.py pose/configs/nk_sapiens/infant/stage1_ssl_full.py`

---

## 7. 评估（val）

- 使用 `python pose/tools/test.py <config> --checkpoint <ckpt>`
- 指标：
  - 若数据为 COCO-style，可直接用 CocoMetric（AP）
  - 若非严格 COCO，只要 evaluator 能输出统一指标并记录在日志即可（需写入报告）

---

## 8. 必做消融（Ablation Matrix）

所有消融必须做到：
- 仅通过 config 开关控制（λ=0 或 enable=False）
- 固定数据划分、固定 bbox_file、固定 seed（0/1/2）

消融列表：
1) Baseline：L_sup + L_cons
2) +DDP：Baseline + L_dev
3) +CMGC：Baseline + L_geo_depth + L_geo_normal
4) +DDP + CMGC
5) Full：+Hybrid gating（teacherA/teacherB）

---

## 9. 报告输出模板（建议）

每次实验产出一份：
- `reports/nk_sapiens_<exp_name>.md`

至少包含：
- config 路径、ckpt 路径、seed
- labeled/unlabeled/bbox_file 的 md5
- val 指标表（含消融对比）
- 失败样例可视化（遮挡/抱持/爬行）
- teacherA/teacherB 使用比例（full 实验必须有）
