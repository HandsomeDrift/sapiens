#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAPIENS_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export SAPIENS_ROOT

export CUDA_VISIBLE_DEVICES=2
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH="${SAPIENS_ROOT}:${SAPIENS_ROOT}/pretrain:${PYTHONPATH:-}"

# 默认配置可按需覆盖：CFG=/path/to/your_cfg PORT=xxxx GPUS=xx
CFG="${CFG:-${SAPIENS_ROOT}/pose/configs/sapiens_pose/semisup/stage1_sagepose_semi.py}"
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
GPUS="${GPUS:-1}"

# 方式 A：先跑 stage0，再把 stage0 ckpt 写进 stage1 的 load_from（见配置）
# 方式 B：直接跑 stage1（跳过有监督热身）

torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${PORT} \
  "${SAPIENS_ROOT}/pose/tools/train.py" "${CFG}"
