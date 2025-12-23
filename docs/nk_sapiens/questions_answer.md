# questions.md 回答

## A. 仓库总体形态（最高优先级）

1. **项目训练框架是什么**

- 统一是 OpenMMLab/mmengine 体系（Runner/Config/Hook），在 pose/det/seg/pretrain 子仓库下分别集成 mmpose/mmdet/mmseg/mmpretrain 代码。
- 训练入口脚本（按子任务分）：
  - pose：`pose/tools/train.py`
  - seg：`seg/tools/train.py`
  - det：`det/tools/train.py`
  - pretrain：`pretrain/tools/train.py`

1. **配置系统**

- 使用 mmengine 的 `.py` 配置（不是 YAML/JSON/Hydra），通过 `Config.fromfile()` 加载。
- 例子（可直接运行的 baseline）：`pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`，加载时在 `pose/tools/train.py` 里调用 `Config.fromfile()`。
- dataset 的 keypoints/skeleton 等 metainfo 放在 `pose/configs/_base_/datasets/*.py` 里，通过 `metainfo=dict(from_file='...')` 引用（见 `pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`）。

1. **现有命令约定**

- 单卡（pose 为例）：
  - `python pose/tools/train.py pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`
- 多卡：
  - `bash pose/tools/dist_train.sh <config.py> <gpus>`（内部用 `torch.distributed.launch`，见 `pose/tools/dist_train.sh`）
- Slurm：
  - `pose/tools/slurm_train.sh`（同样的模式在 det/seg/pretrain 下也有）
- 最小可运行 smoke 命令：仓库里没有 dummy 数据集；只能用真实数据或在 config 里用 `indices` 做子集（`BaseCocoStyleDataset` 支持 `indices`，见 `pose/mmpose/datasets/datasets/base/base_coco_style_dataset.py`）。

------

## B. 数据管线与关键点定义（会直接决定你我写的 spec 是否能落地）

1. **关键点（K）与顺序定义**

- 每个数据集的 K/名字/顺序/flip/skeleton 都在 `pose/configs/_base_/datasets/*.py` 的 `dataset_info` 中。
- 示例（COCO-17）：`pose/configs/_base_/datasets/coco.py`，含 `keypoint_info`（17 点）与 `skeleton_info`。

1. **数据标注格式**

- 主流为 COCO keypoints JSON（COCO-style），由 `BaseCocoStyleDataset` 解析（`pose/mmpose/datasets/datasets/base/base_coco_style_dataset.py`）。
- top-down/bottom-up 由 `data_mode` 控制（`topdown`/`bottomup`）。
- top-down 评估可使用检测框文件 `bbox_file`（检测结果 JSON），见 `pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py` 中 `bbox_file` 配置。
- 解析得到的关键字段包括：`img_path`, `bbox`, `bbox_score`, `keypoints`, `keypoints_visible`, `num_keypoints`, `segmentation` 等（见 `parse_data_info()`）。

1. **数据增强/变换 pipeline**

- 训练 pipeline 示例：`pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`，包含 `GetBBoxCenterScale` → `TopdownAffine` → `PhotometricDistortion` → `Albumentation` → `GenerateTarget` → `PackPoseInputs`。
- 监督类型：该 Sapiens pose 配置使用 heatmap 监督（`HeatmapHead` + `UDPHeatmap`），见同文件里的 `codec` 与 `head.loss`。
- 训练时输出结构：`PackPoseInputs` 输出 `inputs`（tensor）与 `data_samples`（`PoseDataSample`），其中包含 `gt_instances`/`gt_instance_labels`/`gt_fields` 等（见 `pose/mmpose/datasets/transforms/formatting.py`）。

------

## C. 模型输出契约（决定 loss / teacher / 伪标签怎么对齐）

1. **Pose head 输出是什么**

- Sapiens pose 默认 `HeatmapHead` 输出 heatmaps（`B x K x H x W`），`forward()` 直接返回热图（`pose/mmpose/models/heads/heatmap_heads/heatmap_head.py`）。
- `TopdownPoseEstimator` 在 `mode='tensor'` 时直接输出热图；在 `mode='loss'` 时返回 `(losses, preds)`，其中 `preds` 为热图（`pose/mmpose/models/pose_estimators/topdown.py`）。
- `mode='predict'` 输出 `PoseDataSample`，其 `pred_instances` 含 `keypoints`/`keypoint_scores`。

1. **是否已有多头结构或可插拔 head**

- pose 分支默认是单 head（`TopdownPoseEstimator` + `HeatmapHead`）。
- depth/normal/segmentation 在 `seg` 分支单独实现（`DepthEstimator` + `VitDepthHead/VitNormalHead`），与 pose 非多头共享。

1. **Checkpoint 加载方式**

- `train.py` 使用 `cfg.load_from`/`cfg.resume`（`pose/tools/train.py`）。
- `init_cfg` 支持 `Pretrained`（例如在 backbone/head 里指定 `checkpoint`），见 `pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`。

------

## D. 半监督/教师框架（决定我们是否复用现成组件）

1. **现成 EMA teacher / mean-teacher / pseudo-label**

- 原项目中未发现专用的半监督 teacher/mean-teacher/伪标签模块或配置；训练流程以标准监督为主。

1. **无标注数据如何进入 dataloader**

- 原项目未提供“无标注数据”专用 dataset 或两路 dataloader 组织方式。

1. **伪标签过滤机制**

- 原项目未实现伪标签筛选/过滤逻辑。

------

## E. 训练工程细节（保证 codex 实现后能跑）

1. **混合精度策略**

- mmengine 标准 `AmpOptimWrapper`（可通过 `--amp` 或 config 启用），见 `pose/tools/train.py`。

1. **分布式训练方式**

- `pose/tools/dist_train.sh` 使用 `torch.distributed.launch`；seg/det/pretrain 同模式（对应目录 `*/tools/dist_train.sh`）。
- Slurm 脚本在 `*/tools/slurm_train.sh`。

1. **日志与输出**

- 默认 `work_dir` 为 `./work_dirs/<config_name>`（由 `train.py` 决定）。
- 日志/可视化：`LoggerHook` + `TensorboardVisBackend`（见 `pose/configs/_base_/default_runtime.py`）。
- checkpoint 保存策略在各 config 里设置（如 COCO 训练保存 `save_best='coco/AP'`）。

1. **测试/检查命令**

- Pose 测试入口：`pose/tools/test.py`。
- 仓库中未见统一的 lint/ruff/mypy 配置；`pose/pytest.ini` 存在但未给出统一测试命令。

------

## F. 与 NK-Sapiens 方案强相关的可落地性信息

1. **是否能拿到/接入 Sapiens 的 depth & normal heads**

- 有：在 `seg` 分支提供 `DepthEstimator` + `VitDepthHead`/`VitNormalHead`，对应配置在 `seg/configs/sapiens_depth/*` 与 `seg/configs/sapiens_normal/*`。
- 输出形状示例：
  - depth：`VitDepthHead.forward()` 输出 `B x 1 x H x W`（`seg/mmseg/models/decode_heads/vit_depth_head.py`）
  - normal：`VitNormalHead` 配置里 `num_classes=3`，输出 `B x 3 x H x W`（`seg/mmseg/models/decode_heads/vit_normal_head.py` + `seg/configs/sapiens_normal/normal_general/sapiens_0.3b_normal_general-1024x768.py`）

1. **输入分辨率与预处理**

- Sapiens pose 常用 `image_size=[1024, 768]`（W x H），见 `pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`。
- Top-down crop/affine：`GetBBoxCenterScale` 与 `TopdownAffine`（`pose/mmpose/datasets/transforms/common_transforms.py`, `pose/mmpose/datasets/transforms/topdown_transforms.py`）。
- seg depth/normal 也使用 1024x768 级别输入（见 `seg/configs/sapiens_depth/depth_general/sapiens_1b_depth_general-1024x768.py`）。

1. **评估指标脚本**

- COCO 2D pose：`CocoMetric`（COCO AP），配置在 `pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py`。
- 其他 2D 指标（PCK/AUC/NME 等）在 `pose/mmpose/evaluation/metrics/keypoint_2d_metrics.py`。
- depth/normal 示例 config 默认未启用 evaluator（`val_evaluator = None`），如需评估需自行配置（`seg/mmseg/evaluation/metrics/depth_metric.py` 提供深度评测）。
