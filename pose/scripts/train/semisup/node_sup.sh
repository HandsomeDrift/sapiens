#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CFG="configs/sapiens_pose/semisup/stage0_sup_warmup.py"
PORT=29502
GPUS=4

torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=${PORT} \
  pose/tools/train.py ${CFG}
