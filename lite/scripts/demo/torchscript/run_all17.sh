#!/usr/bin/env bash
# pose_keypoints17_multi.sh
# 批量处理根目录下的多个(子)目录；输出路径包含输入根目录名（如 20250305）
# 支持 -n 干跑，仅打印流程与将执行的命令

set -euo pipefail

RECURSIVE=false
DRYRUN=false
INPUT_ROOT=""
OUTPUT_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  pose_keypoints17_multi.sh -i <INPUT_ROOT> -o <OUTPUT_ROOT> [-r] [-n]

Options:
  -i    根输入目录（其下的子目录将被处理；若 -r 则递归）
  -o    根输出目录（结果会保存到 <OUTPUT_ROOT>/<basename(INPUT_ROOT)>/…）
  -r    递归遍历（默认仅遍历一级子目录）
  -n    干跑（dry-run）：只打印流程，不创建文件、不执行推理
  -h    显示帮助
EOF
}

while getopts ":i:o:rnh" opt; do
  case "$opt" in
    i) INPUT_ROOT="$OPTARG" ;;
    o) OUTPUT_ROOT="$OPTARG" ;;
    r) RECURSIVE=true ;;
    n) DRYRUN=true ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${INPUT_ROOT}" || -z "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: 必须同时指定 -i 与 -o" >&2
  usage
  exit 1
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "ERROR: 输入根目录不存在: $INPUT_ROOT" >&2
  exit 1
fi

# --------------------- 配置（按需修改） ---------------------
MODE='torchscript'            # 或 'bfloat16'
MODEL_NAME='sapiens_2b'
RUN_FILE='demo/vis_pose.py'

SAPIENS_CHECKPOINT_ROOT="/data/${USER}/sapiens_lite_host/${MODE}"
CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/pose/checkpoints/${MODEL_NAME}/${MODEL_NAME}_coco_best_coco_AP_822_${MODE}.pt2"
DETECTION_CONFIG_FILE='../pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT="${SAPIENS_CHECKPOINT_ROOT}/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

RADIUS=6
KPT_THRES=0.3
BATCH_SIZE=8

# GPU/并发（按需调整）
VALID_GPU_IDS=(3 5 6 7 4)
TOTAL_GPUS=${#VALID_GPU_IDS[@]}
JOBS_PER_GPU=1

# --------------------- 路径定位到仓库根目录（按需调整层级） ---------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
cd "$REPO_ROOT" || { echo "ERROR: 无法进入仓库根目录 $REPO_ROOT"; exit 1; }

# --------------------- 工具函数 ---------------------
say_do() {
  # 打印并在非干跑时执行
  if $DRYRUN; then
    echo "[DRYRUN] $*"
  else
    eval "$@"
  fi
}

has_images() {
  local d="$1"
  find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 1 >/dev/null
}

count_images() {
  local d="$1"
  find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l | tr -d ' '
}

# --------------------- 收集要处理的目录 ---------------------
declare -a DIRS=()
if $RECURSIVE; then
  while IFS= read -r -d '' d; do
    if has_images "$d"; then DIRS+=("$d"); fi
  done < <(find "$INPUT_ROOT" -type d -print0)
else
  while IFS= read -r -d '' d; do
    if has_images "$d"; then DIRS+=("$d"); fi
  done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)
fi

if (( ${#DIRS[@]} == 0 )); then
  echo "WARN: 未发现包含图片的子目录。"
  exit 0
fi

ROOT_NAME="$(basename "$(realpath "$INPUT_ROOT")")"
echo "[INFO] 预计处理目录数：${#DIRS[@]}"
echo "[INFO] 输出根将映射为：${OUTPUT_ROOT}/${ROOT_NAME}/<相对路径>"
echo "------------------------------------------------------------"

# --------------------- 逐目录处理 ---------------------
for INPUT_DIR in "${DIRS[@]}"; do
  REL_PATH="$(realpath --relative-to="$INPUT_ROOT" "$INPUT_DIR")"
  # *** 核心映射：在输出根与相对路径之间插入 basename(INPUT_ROOT) ***
  OUTPUT_DIR="${OUTPUT_ROOT}/${ROOT_NAME}/${REL_PATH}"

  NUM_IMAGES=$(count_images "$INPUT_DIR")

  echo "[INFO] 目录: $INPUT_DIR"
  echo "       输出: $OUTPUT_DIR"
  echo "       图片: ${NUM_IMAGES} 张"

  if (( NUM_IMAGES == 0 )); then
    echo "       -> 无图片，跳过"
    echo "------------------------------------------------------------"
    continue
  fi

  # 预创建输出目录（干跑仅打印）
  say_do "mkdir -p \"${OUTPUT_DIR}\""

  # 生成完整图片列表（干跑不落盘，只打印）
  IMAGE_LIST="${INPUT_DIR}/image_list.txt"
  if $DRYRUN; then
    echo "[DRYRUN] 将创建清单: ${IMAGE_LIST}"
  else
    find "${INPUT_DIR}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort > "${IMAGE_LIST}"
  fi

  # 计算切分参数
  max_jobs=$((JOBS_PER_GPU * TOTAL_GPUS))
  ceil_batches=$(( (NUM_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE ))
  if (( max_jobs > ceil_batches )); then
    TOTAL_JOBS=$ceil_batches
  else
    TOTAL_JOBS=$max_jobs
  fi
  (( TOTAL_JOBS == 0 )) && TOTAL_JOBS=1

  BASE=$(( NUM_IMAGES / TOTAL_JOBS ))
  REM=$(( NUM_IMAGES % TOTAL_JOBS ))

  echo "       将切分为 ${TOTAL_JOBS} 个作业；BATCH_SIZE=${BATCH_SIZE}"
  echo "       每作业基数: ${BASE} 张；前 ${REM} 个作业各多 1 张"

  # 切分到 image_paths_*.txt（干跑只打印）
  start=1
  for ((i=1; i<=TOTAL_JOBS; i++)); do
    if (( i <= REM )); then
      count=$((BASE + 1))
    else
      count=$BASE
    fi
    text_file="${INPUT_DIR}/image_paths_${i}.txt"
    if $DRYRUN; then
      echo "[DRYRUN] 将创建分片清单: ${text_file} (行数: ${count}, 区间: ${start}-$((start+count-1)))"
    else
      # 使用 sed 精准切片
      sed -n "${start},$((start+count-1))p" "${IMAGE_LIST}" > "${text_file}"
    fi
    start=$((start + count))
  done

  # 执行每个作业（干跑只打印命令）
  for ((i=1; i<=TOTAL_JOBS; i++)); do
    gpu_slot=$(( (i-1) % TOTAL_GPUS ))
    gpu_id="${VALID_GPU_IDS[$gpu_slot]}"
    text_file="${INPUT_DIR}/image_paths_${i}.txt"

    cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python \"${RUN_FILE}\" \
\"${CHECKPOINT}\" --num_keypoints 17 \
--det-config \"${DETECTION_CONFIG_FILE}\" \
--det-checkpoint \"${DETECTION_CHECKPOINT}\" \
--batch-size ${BATCH_SIZE} \
--input \"${text_file}\" \
--output-root \"${OUTPUT_DIR}\" \
--radius ${RADIUS} --kpt-thr ${KPT_THRES}"

    if $DRYRUN; then
      echo "[DRYRUN] $cmd"
    else
      eval "$cmd"
      # 避免把 GPU 打满可小憩一下；按需调整/去掉
      sleep 1
    fi
  done

  # 清理临时清单（干跑不删除）
  if ! $DRYRUN; then
    rm -f "${IMAGE_LIST}"
    for ((i=1; i<=TOTAL_JOBS; i++)); do
      rm -f "${INPUT_DIR}/image_paths_${i}.txt"
    done
  fi

  echo "------------------------------------------------------------"
done

echo $($DRYRUN && echo "[DRYRUN] 完成：仅打印流程，未执行推理。" || echo "全部目录处理完成。")
