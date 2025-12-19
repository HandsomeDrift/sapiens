
#!/usr/bin/env bash
set -e

export SAPIENS_ROOT=/home/xiangxiantong/sapiens

# 关键：让本地的 mmpretrain 先于 site-packages 被 import
export PYTHONPATH=$SAPIENS_ROOT/pretrain:$SAPIENS_ROOT:$PYTHONPATH
# 可选，避免 usersite 抢先：
export PYTHONNOUSERSITE=1

# ——(可选) 一次性自检：现在用的是否是“仓库版”的 vision_transformer —— 
python - <<'PY'
import mmpretrain.models.backbones.vision_transformer as vt
print('[mmpretrain vt path] ', vt.__file__)
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
print('[has sapiens_1b]    ', 'sapiens_1b' in VisionTransformer.arch_zoo)
PY
# 期望输出：
# [mmpretrain vt path] /home/xiangxiantong/sapiens/pretrain/mmpretrain/models/backbones/vision_transformer.py
# [has sapiens_1b]     True


# 必要环境变量：
# export SAPIENS_ROOT=/home/xiangxiantong/sapiens
export DATA_ROOT=/data-nxs/xiangxiantong/stand_data
export CUDA_VISIBLE_DEVICES=4,
# conda activate sapiens_lite_clone

CFG=$SAPIENS_ROOT/pose/configs/sapiens_pose/semisup/sapiens1b_coco17_sagepose_semi.py
# WORK_DIR=$SAPIENS_ROOT/pose/work_dirs/sagepose_semisup
WORK_DIR=$SAPIENS_ROOT/pose/work_dirs/sagepose_unsup

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

mkdir -p $WORK_DIR

# 单机多卡训练
torchrun --nnodes=1 --nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') \
  $SAPIENS_ROOT/pose/tools/train.py \
  $CFG \
  --work-dir $WORK_DIR \
  --cfg-options DATA_ROOT=$DATA_ROOT
