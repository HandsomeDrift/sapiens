#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")"/../../../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT/pretrain:$REPO_ROOT/pose:$PYTHONPATH"

cd ../../../..

###--------------------------------------------------------------
# DEVICES=0,
DEVICES=2,

RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

####-----------------MODEL_CARD----------------------------
DATASET='xt'
MODEL="sapiens_1b_custom_coco17"
JOB_NAME="test_pose_${MODEL}"
TEST_BATCH_SIZE_PER_GPU=1

# Path to your finetuned checkpoint
CHECKPOINT="/data/xxt/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_coco_best_coco_AP_821.pth"
DATA_ROOT="/data/xxt/sapiens_data"
KEEP_INDICES="5,6,7,8,9,10,11,12,13,14,15,16"

###--------------------------------------------------------------
# mode='debug'
mode='multi-gpu'

###--------------------------------------------------------------
CONFIG_FILE=configs/sapiens_pose/custom/sapiens_1b-quickstart_custom_coco17.py
OUTPUT_DIR="Outputs/test/${DATASET}/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

export TF_CPP_MIN_LOG_LEVEL=2

## set the options for the test
OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU \
test_evaluator.format_only=True \
test_evaluator.outfile_prefix=${OUTPUT_DIR}/preds \
test_evaluator.nms_mode=none \
test_evaluator.score_mode=bbox")"

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TEST_BATCH_SIZE_PER_GPU=16

    OPTIONS="$(echo "test_dataloader.batch_size=${TEST_BATCH_SIZE_PER_GPU} test_dataloader.num_workers=0 test_dataloader.persistent_workers=False")"
    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS}

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} ${CHECKPOINT} \
            ${NUM_GPUS} \
            --work-dir ${OUTPUT_DIR} \
            --cfg-options ${OPTIONS} \
            | tee ${LOG_FILE}

fi

python tools/eval_coco_subset.py \
    --ann-file "${DATA_ROOT}/annotations/person_keypoints_xt_val.json" \
    --pred-file "${OUTPUT_DIR}/preds.keypoints.json" \
    --keep-indices "${KEEP_INDICES}" \
    --sigmas-file "configs/_base_/datasets/coco.py" \
    --score-mode bbox_keypoint \
    --pred-score-mode bbox \
    --keypoint-score-thr 0.2 \
    --nms-mode oks_nms \
    --nms-thr 0.9 \
    --output-metrics "${OUTPUT_DIR}/metrics_body_no_face.json"
