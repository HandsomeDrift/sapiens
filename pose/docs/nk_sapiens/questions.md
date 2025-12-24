## A. 仓库总体形态（最高优先级）

1. **项目训练框架是什么**

- 纯 PyTorch？还是 mmengine/mmpose？还是 Lightning？还是自研 runner？
- 需要：训练/评估入口脚本路径（例如 `tools/train.py`、`train.py`、`scripts/train.sh`）

1. **配置系统**

- 用 YAML/JSON/py config？（mmengine 的 `.py` config 还是 Hydra/OmegaConf？）
- 需要：一个能跑通的现有 config（哪怕是 baseline），以及它如何被加载。

1. **现有命令约定**

- 当前“标准跑法”是什么：
  - 单卡：`python ...`
  - 多卡：`torchrun ...` 或 `bash dist_train.sh ...`
- 需要：最小可运行的 smoke 命令（不用真实数据也行，用 dummy / few-shot）。

------

## B. 数据管线与关键点定义（会直接决定你我写的 spec 是否能落地）

1. **关键点（K）与顺序定义**

- K 是多少？每个 keypoint 的名字/顺序是什么？
- skeleton limbs（骨架连边）怎么定义？（用于 CMGC 与 KPS / ROM 统计）

1. **数据标注格式**

- COCO keypoints JSON？还是自定义？
- top-down 还是 bottom-up？是否依赖检测框 `detection_results.json`？
- 需要：dataset class 名称、读取标注的关键字段（image path、bbox、kpts、visibility等）

1. **数据增强/变换 pipeline**

- weak/strong aug 是否已有实现？
- heatmap 监督还是直接回归 coords？
- 需要：训练时输出张量的结构（例如 `inputs`, `targets`, `meta` 的字段）

------

## C. 模型输出契约（决定 loss / teacher / 伪标签怎么对齐）

1. **Pose head 输出是什么**

- 输出 heatmaps？还是 (x,y,conf)？还是 SimCC？
- 需要：forward 的返回 dict/tuple 格式示例（最关键）

1. **是否已有多头结构或可插拔 head**

- 你现在的代码里有没有 segmentation/depth/normal 这种 head？
- 如果没有：是否允许“只实现 pose + 外部几何约束的替代版”（会影响 CMGC 的实现方式）

1. **Checkpoint 加载方式**

- backbone/head 的 key 前缀是什么？（比如 `backbone.` / `encoder.`）
- 是否支持部分加载、strict=False、或者 mmengine 的 `load_from`？

------

## D. 半监督/教师框架（决定我们是否复用现成组件）

1. **仓库里有没有现成 EMA teacher / mean-teacher / pseudo-label 逻辑**

- 类名/文件名/配置项（例如 `ema_momentum`、`teacher_cfg`、`unsup_loss_weight`）

1. **无标注数据如何被组织进入 dataloader**

- 是两个 dataloader（labeled/unlabeled）？还是一个混合 dataset？
- batch 里怎么区分 labeled vs unlabeled？

1. **伪标签过滤机制**（若已有）

- 用 conf 阈值？top-k？mask？dynamic threshold？
- 需要：当前的伪标签选择逻辑在哪里（文件/函数名）

------

## E. 训练工程细节（保证 codex 实现后能跑）

1. **混合精度策略**

- 目前用 fp16/bf16/amp 吗？由谁控制（torch.cuda.amp / mmengine runner）？
- 我们需要把 CMGC/DDP 的部分强制 float32：你们框架里怎么优雅实现（autocast 关闭区域）？

1. **分布式训练方式**

- torchrun + DDP？还是 Deepspeed？
- 有没有现成的 `dist_train.sh` / launcher？

1. **日志与输出**

- 用 wandb/tensorboard/jsonl？
- 输出目录结构是什么？（run_id、ckpt 保存策略）

1. **测试/检查命令**

- 有没有 `pytest`、lint（ruff/flake8）、typecheck（mypy）？
- 没有也行，但至少要知道“最小 smoke-test 怎么跑”。

------

## F. 与 NK-Sapiens 方案强相关的可落地性信息（用于决定我们做“完整 CMGC”还是“可替代 CMGC”）

1. **是否能拿到/接入 Sapiens 的 depth & normal heads**

- 仓库里是否已经集成 Sapiens？
- 如果集成了：depth/normal 的 forward 输出在哪里？shape 是什么？
- 如果没集成：是否允许通过“外部预估深度/法向模型”替代（这会改变 CMGC 的来源）

1. **输入分辨率与预处理**

- 训练输入分辨率是多少？是否固定 256×192/384×288 这种？还是 1024 级别？
- top-down crop/affine 的实现在哪里？

1. **评估指标脚本**

- 用 COCO AP？还是 PCK/AUC？
- eval 脚本怎么调用、输出格式是什么？

