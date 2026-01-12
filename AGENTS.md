# Repository Guidelines

## 项目结构与模块组织
- 核心模块位于仓库根目录：`pose/`、`seg/`、`det/`、`pretrain/`、`cv/`、`engine/`、`lite/`。
- 文档集中在 `docs/`，姿态相关方案文档在 `pose/docs/`；媒体资源在 `assets/`。
- 本次任务新增实现建议放在 `pose/geosapiens/` 与 `pose/configs/geo_sapiens/`，避免污染原有代码路径。

## 当前重点任务（Hyper‑Sapiens）
- 统一关键点空间为 COCO17；输入分辨率按数据集配置（优先高分辨率如 1024）。
- 三阶段课程学习：Stage 1 合成热身（Pose/Depth/Normal 监督）；
  Stage 2 混合桥接（DDP 先验 + 几何一致性）；
  Stage 3 真实精调（Real+Replay，DDP 权重退火）。
- 扩散先验（DDP）基于私有骨架数据训练，推理时冻结并通过 SDS 约束真实样本。
- 25→17 的关键点映射与 16 公共点评估索引尚未确定，必须在实现前补齐。

## 不可信实现
- `pose/semisup/` 与 `pose/semisup_old/` 为实验性代码，明确不复用、不参考。

## 关键文档
- 新方案说明：`pose/docs/Hyper-Sapiens/Hyper-Sapiens：基于扩散先验与几何一致性的婴儿姿态估计Sim-to-Real迁移深度研究报告.md`
- 规格文档：`pose/docs/Hyper-Sapiens/Hyper-Sapiens_Spec.md`

## 构建、训练与测试命令
- 训练入口：`python pose/tools/train.py <config.py>`（seg/det/pretrain 同路径结构）。
- 分布式：`bash pose/tools/dist_train.sh <config.py> <gpus>`。
- 测试：`python pose/tools/test.py <config.py> <checkpoint>`。

## 编码风格与命名
- Python 4 空格缩进，遵循 PEP8。
- 每个子模块有 `setup.cfg`（含 isort/flake8/yapf 设置）；新增文件请对齐现有命名风格。
- 配置文件命名保持与现有模式一致，如 `sapiens_0.3b-210e_coco-640x480.py`。

## 测试规范
- `pose/` 提供 `pytest` 配置：`cd pose && pytest`。
- 无统一覆盖率门槛；新增逻辑建议添加小规模快速测试。

## 提交与 PR 建议
- 历史提交信息较短且无强制规范；保持简洁清晰即可。
- PR 需包含变更说明、相关配置路径与可复现实验命令。
