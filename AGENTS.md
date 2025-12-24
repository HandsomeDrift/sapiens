# AGENTS.md — NK-Sapiens (Stage0+Stage1) 工程边界与 Codex 执行规约

> 本文件用于约束 codex cli / coding agent 在你的 OpenMMLab/mmengine 多子仓库中安全工作。

## 1. 仓库背景（不可改变的现实）
- 仓库采用 mmengine `.py` config；训练/测试入口固定在 `pose/tools/train.py`、`pose/tools/dist_train.sh`、`pose/tools/test.py`。
- 仓库无 dummy dataset：smoke 必须用真实数据，通过 dataset 的 `indices` 抽子集完成。
- pose 分支默认单 head（HeatmapHead），depth/normal 在 seg 分支单独实现；本任务通过“冻结几何教师”方式跨分支调用。

## 2. 范围（Scope）

### MUST（本里程碑必须实现）
1) Stage0 config + 可运行训练
- `pose/configs/nk_sapiens/infant/stage0_supervised.py`

2) Stage1 SSL 框架（最小闭环）
- labeled + unlabeled 数据同时进入训练（dataset/dataloader 组织）
- weak/strong 双视图（默认 `num_strong_views=1`）
- EMA teacherA + L_cons
- `pose/configs/nk_sapiens/infant/stage1_ssl_baseline.py`

3) Stage1 完整 NK-Sapiens（可开关消融）
- DDP prior：训练脚本 + 推断接口 + L_dev（不得作为硬标签）
- CMGC：加载 seg 的 depth/normal 冻结教师，输出 depth/normal 并计算 L_geo
- Hybrid gating：conf + KPS 选择 teacherA 或 teacherB（记录统计）
- `pose/configs/nk_sapiens/infant/stage1_ssl_full.py`（含消融开关）

4) 工具与统计
- `compute_kps_stats.py`：输出 `stats/skeleton_stats.json`
- （可选但推荐）把运行 md5、seed、git hash 写入 work_dir 的 `meta.json`

### MUST NOT（明确禁止）
- 不得重构整个 mmengine/mmpose 训练框架；优先以“新增模块 + config 注入”的方式集成。
- 不得破坏现有 `pose/configs/sapiens_pose/*` baseline 的可运行性。
- 不得引入“需要 3D GT”的依赖（CMGC 必须在无 3D 标注下工作）。
- 不得替换仓库内嵌的定制版 mmpretrain（禁止改成 pip 官方版）。
- 不得实现 Stage2（本里程碑不做）。
- 不得新增“在 20k 上微调 YOLOv8 检测器”的实现（先使用通用检测器 bbox_file）。

### OPTIONAL（未来可能做，但本次不交付）
- Stage2 fine-grained 手部 zoom-in crops 微调与相关指标
- 针对幼儿场景的检测器微调（如 YOLOv8）

## 3. 数值与显存硬约束（Hard Constraints）
- Stage1 默认 `num_strong_views=1`（必须可配置，但默认 1）。
- AMP/bfloat16 可用于前向，但以下部分必须强制 float32：
  - CMGC 相关 loss（涉及坐标变换/开方/归一化）
  - DDP prior 采样/去噪与 L_dev
  实现方式：在这些 loss 计算区域显式关闭 autocast，或将参与运算张量转换为 float32。

## 4. 允许的集成方式（Preferred Integration）
- 使用 `pose/projects/nk_sapiens/` 作为新增代码目录，并通过 config 的 `custom_imports` 引入。
- 通过注册（registry）方式新增：
  - Dataset/Transform/Hook/Model wrapper/Loss 等
- 尽量不修改 `pose/tools/train.py`：
  - 如确实需要支持“多 dataloader / labeled+unlabeled”训练循环，修改必须最小化，并确保原监督训练 config 不受影响。

## 5. 交付要求（每个 codex 任务必须输出）
每个任务提交时，必须给出：
1) **变更文件列表** + 关键 diff 摘要
2) **可运行命令**（至少 1 条 smoke：用 `indices` 子集跑 50 iter）
3) **日志证据**
   - work_dir 路径
   - loss 数值（无 NaN）
   - full 版本必须输出 teacherA/teacherB 选择计数
4) **开关说明**
   - 如何通过 config 打开/关闭 DDP、CMGC、Hybrid（用于消融）

## 6. Definition of Done（完成定义）
- Stage0：单卡可启动训练，smoke 50 iter 不 NaN
- Stage1-baseline：单卡可启动训练，smoke 50 iter 不 NaN，EMA teacherA 正常更新
- Stage1-full：单卡可启动训练，smoke 50 iter 不 NaN；teacherA/teacherB 两分支都可触发且有统计
- test：`pose/tools/test.py` 可在 val 上输出指标
- 文档与 config 对齐：SPEC / PROTOCOL / AGENTS 与实际实现一致
