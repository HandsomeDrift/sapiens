# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sapiens is Meta's human-centric vision foundation model (ECCV 2024 Best Paper Candidate), forked from `facebookresearch/sapiens`. This fork adds **SAGE-Pose**: a semi-supervised pose estimation framework using a dual-teacher EMA architecture with structural priors.

## Repository Structure

Six independently installable packages share one repo:

| Package | Purpose |
|---------|---------|
| `pose/` | Pose estimation (main development focus) |
| `pretrain/` | ViT backbone pretraining (sapiens_0.3b/0.6b/1b/2b) |
| `seg/` | Body part segmentation |
| `det/` | Human detection (RTMPose) |
| `cv/` | MMCv 2.1.0 fork (shared vision utilities) |
| `engine/` | MMEngine 0.10.7 fork (training framework) |
| `lite/` | Optimized TorchScript inference (not pip-installable) |

The **SAGE-Pose semi-supervised code** (`pose/semisup/`) is the key addition over upstream:
- `models/dual_teacher_wrapper.py` — DualTeacherWrapper: student + EMA teacher with consistency loss, structural priors (LaplacianTopo, BoneLength, JointAngle)
- `data/unlabeled_coco_topdown.py` — unlabeled data loading
- `data/pipelines/` — augmentation pipelines (FixCenterScale, etc.)
- `utils/geometry.py` — heatmap warping
- `utils/pseudo_label.py` — teacher fusion, dynamic keypoint masking

## Environment Setup

- Python 3.10, PyTorch 2.2.2 + CUDA 12.1, numpy 1.26.4
- mmengine 0.10.7, mmcv 2.1.0, mmdet 3.2.0
- Install script: `_install/conda.sh` (creates `sapiens` conda env)
- Manual install: `pip install -e engine && pip install -e cv && pip install -e pretrain && pip install -e pose && pip install -e det && pip install -e seg`
- **Required PYTHONPATH** before any MMEngine command:
  ```bash
  export SAPIENS_ROOT=/path/to/sapiens
  export PYTHONPATH="$SAPIENS_ROOT:$SAPIENS_ROOT/pretrain:$SAPIENS_ROOT/pose:$PYTHONPATH"
  ```

## Common Commands

### Training
```bash
# Semi-supervised SAGE-Pose (two-stage)
python pose/tools/train.py pose/configs/sapiens_pose/semisup/stage0_sup_warmup.py --work-dir work_dirs/stage0
python pose/tools/train.py pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py --work-dir work_dirs/stage1

# Distributed training
bash pose/tools/dist_train.sh <config> <num_gpus>

# Override config values inline
python pose/tools/train.py <config> --cfg-options train_dataloader.batch_size=4
```

### Evaluation
```bash
python pose/tools/test.py <config> <checkpoint> --work-dir work_dirs/eval
python pose/tools/test.py <config> <checkpoint> --dump output.pkl  # save predictions
```

### Testing
```bash
pytest pose/tests -q
```

### Linting
```bash
ruff check pose/   # or flake8
```

## Config System

Configs use MMEngine's `Config.fromfile()` — pure Python files with inheritance. Key locations:
- `pose/configs/sapiens_pose/semisup/` — SAGE-Pose semi-supervised training recipes
- `pose/configs/sapiens_pose/coco/` — standard COCO supervised configs
- `pose/configs/sapiens_pose/coco_wholebody/` — whole-body (133 keypoints)
- `pose/configs/sapiens_pose/custom/` — custom dataset configs

Configs reference env vars (`SAPIENS_ROOT`, `DATA_ROOT`) and use absolute paths for data/checkpoints. When editing configs, update path variables rather than hardcoding paths.

## Architecture: MMEngine Pattern

All modules follow MMEngine conventions:
1. **Registry-based** — models, datasets, transforms, metrics are registered via decorators (`@MODELS.register_module()`)
2. **Config-driven** — `Runner.from_cfg(cfg)` builds everything from config dict
3. **Entry points** — `pose/tools/train.py` and `pose/tools/test.py` parse config + CLI args, then call `Runner`
4. Custom modules must be imported before Runner construction (semisup `__init__.py` handles this via `custom_imports` in configs)

## Coding Conventions

- 4-space indentation, snake_case for variables/files
- Explicit names preferred (`lambda_u_eff`, `geom_teacher`)
- Type hints for tensor shapes when they clarify intent
- Commit messages: short imperative (`Fix EMA warmup`, `Add KL mask logging`)
- Never commit checkpoints or raw datasets; use config path variables
