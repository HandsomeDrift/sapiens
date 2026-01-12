#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")"/../../../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT/pretrain:$REPO_ROOT/pose:$PYTHONPATH"

cd ../../../..

DEVICES=0,
RUN_FILE='./tools/dist_test.sh'

MODEL="sapiens_1b_custom_coco17"
TEST_BATCH_SIZE_PER_GPU=1

CONFIG_FILE=configs/sapiens_pose/custom/sapiens_1b-quickstart_custom_coco17.py
CHECKPOINT="/path/to/sapiens_1b_checkpoint.pth"

MINI_RGBD_ROOT="/path/to/mini_rgbd_coco17"
MINI_RGBD_SPLIT="mini_rgbd_test2017"
MINI_RGBD_BBOX_FILE=""

SYRIP_ROOT="/path/to/syrip_coco17"
SYRIP_SPLIT="syrip_test2017"
SYRIP_BBOX_FILE=""

mode='multi-gpu'

run_test() {
    local DATASET_NAME=$1
    local DATA_ROOT=$2
    local SPLIT_NAME=$3
    local BBOX_FILE=$4

    local ANN_FILE="annotations/person_keypoints_${SPLIT_NAME}.json"
    local IMG_PREFIX="${SPLIT_NAME}/"
    local OUTPUT_DIR="Outputs/test/${DATASET_NAME}/${MODEL}/node"
    OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

    local OPTIONS="test_dataloader.batch_size=${TEST_BATCH_SIZE_PER_GPU}"
    OPTIONS="${OPTIONS} test_dataloader.dataset.data_root=${DATA_ROOT}"
    OPTIONS="${OPTIONS} test_dataloader.dataset.ann_file=${ANN_FILE}"
    OPTIONS="${OPTIONS} test_dataloader.dataset.data_prefix.img=${IMG_PREFIX}"
    OPTIONS="${OPTIONS} test_evaluator.ann_file=${DATA_ROOT}/${ANN_FILE}"

    if [ -n "${BBOX_FILE}" ]; then
        if [[ "${BBOX_FILE}" != /* ]]; then
            BBOX_FILE="${DATA_ROOT}/${BBOX_FILE}"
        fi
        OPTIONS="${OPTIONS} test_dataloader.dataset.bbox_file=${BBOX_FILE}"
    else
        OPTIONS="${OPTIONS} test_dataloader.dataset.bbox_file=None"
    fi

    if [ "$mode" = "debug" ]; then
        OPTIONS="${OPTIONS} test_dataloader.num_workers=0 test_dataloader.persistent_workers=False"
        CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
            --work-dir ${OUTPUT_DIR} \
            --cfg-options ${OPTIONS}
        return 0
    fi

    local PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))
    local NUM_GPUS=0
    IFS=',' read -ra DEV_ARRAY <<< "${DEVICES}"
    for dev in "${DEV_ARRAY[@]}"; do
        if [ -n "$dev" ]; then
            NUM_GPUS=$((NUM_GPUS + 1))
        fi
    done

    local LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} ${CHECKPOINT} \
        ${NUM_GPUS} \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options ${OPTIONS} \
        | tee ${LOG_FILE}
}

run_test "mini_rgbd" "${MINI_RGBD_ROOT}" "${MINI_RGBD_SPLIT}" "${MINI_RGBD_BBOX_FILE}"
run_test "syrip" "${SYRIP_ROOT}" "${SYRIP_SPLIT}" "${SYRIP_BBOX_FILE}"
