# MINI-RGBD → Sapiens 数据预处理指南

本目录提供 `prepare_mini_rgbd.py` 脚本，用于将 MINI-RGBD 数据集转换为 Sapiens/SAGE-Pose 所需的 COCO Top-Down 标注及人体检测 JSON。

## 1. 准备数据

MINI-RGBD 原始目录结构通常为：

```
MINI-RGBD/
 ├── 01/
 │    ├── rgb/
 │    │    ├── syn_00000.png
 │    │    └── ...
 │    ├── depth/
 │    │    ├── syn_00000_depth.png
 │    │    └── ...
 │    ├── fg_mask/
 │    │    ├── mask_00000.png
 │    │    └── ...
 │    ├── joints_2Ddep/
 │    │    ├── syn_joints_2Ddep_00000.txt
 │    │    └── ...
 │    ├── joints_3D/
 │    ├── smil_params/
 │    ├── smil_shape_betas.txt
 │    └── tpose/
 │         ├── rgb/
 │         ├── depth/
 │         ├── syn_joints_2Ddep_tpose.txt
 │         └── syn_joints_3D_tpose.txt
 ├── 02/
 └── ...
```

脚本默认从 `rgb/` 和 `joints_2Ddep/` 中读取数据，`depth/`、`fg_mask/` 等目录为可选辅助信息。

## 2. 运行转换脚本

```
python pose/scripts/datasets/mini_rgbd/prepare_mini_rgbd.py \
  --source-root /data/MINI-RGBD \
  --output-root /data/mini_rgbd_coco \
  --split train=01-08 val=09-10 unsup=01-12 \
  --img-ext .png
```

- `--split` 支持多个划分，格式为 `名称=序列列表`。序列列表里可混用逗号与区间（如 `01-08,09`）。
- 脚本会为每个划分写出：
  - `annotations/mini_rgbd_<split>_keypoints.json`
  - `detections/mini_rgbd_<split>_person_dets.json`
  - `image_id_maps/mini_rgbd_<split>_image_ids.json`
- 输出使用 MINI-RGBD 原生的 25 个 SMIL 关键点（顺序同 `jointlist.txt`），并写入默认骨架连线（root→四肢→末端）。所有可见关键点的 `visibility` 为 2，缺失或无效时为 0。
- 检测 JSON 直接使用关键点外接框（置信度默认 1.0，可用 `--score` 调整）。

## 3. 配置样例

- `pose/configs/sapiens_pose/semisup/mini_rgbd_stage0_sup.py`：有监督热身阶段，模型头部输出 25 个通道。
- `pose/configs/sapiens_pose/semisup/mini_rgbd_stage1_semi.py`：半监督 Stage1，双教师同样处理 25 个关键点。

请将两份配置中的 `DATA_ROOT`、`init_cfg.checkpoint` 等占位路径替换为真实位置；若已完成 Stage0，可在 Stage1 中填写 `load_from` 以加载热身后的学生权重。

## 4. 常见参数

- `--bbox-min-side`：若关键点框过小，强制设定最小边长，默认 40 像素。
- `--skip-missing`：发现缺图/缺标注时直接跳过，而不是抛错。
- `--verbose`：打印每个序列的处理进度。

## 5. 处理后的使用方式

Stage0（有监督热身）：
```
python pose/tools/train.py pose/configs/sapiens_pose/semisup/mini_rgbd_stage0_sup.py \
  --work-dir work_dirs/mini_rgbd_stage0
```

Stage1（半监督微调）：
```
python pose/tools/train.py pose/configs/sapiens_pose/semisup/mini_rgbd_stage1_semi.py \
  --work-dir work_dirs/mini_rgbd_stage1 \
  --cfg-options model.semi_cfg.unsup_only=True
```

如需在 Stage1 中同时使用 MINI-RGBD 有标签数据，可把 `unsup_only` 改为 `False` 并扩展 dataloader，使其同时加载 `CocoDataset` 监督分支。

---

脚本仅依赖 Python + Pillow；若需进一步自定义（如保留 25 个 SMIL 关节或转为 3D 标注），可在此基础上扩展。
