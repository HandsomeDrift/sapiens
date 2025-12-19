#!/usr/bin/env bash
# pose_keypoints17_multi.sh
# 批量处理根目录下的多个(子)目录；输出路径包含输入根目录名（如 20250305）
# 支持 -n 干跑、-r 递归、-D 深度判定（目录任意层有图即处理）、-v 详细日志

set -euo pipefail

RECURSIVE=false
DRYRUN=false
DEEP_CHECK=false
VERBOSE=false
INPUT_ROOT=""
OUTPUT_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  pose_keypoints17_multi.sh -i <INPUT_ROOT> -o <OUTPUT_ROOT> [-r] [-n] [-D] [-v]

Options:
  -i    根输入目录（其下的子目录将被处理；若 -r 则递归）
  -o    根输出目录（结果会保存到 <OUTPUT_ROOT>/<basename(INPUT_ROOT)>/…）
  -r    递归遍历（默认仅遍历一级子目录）
  -n    干跑（dry-run）：只打印流程，不创建文件、不执行推理
  -D    深度判定（目录内任意层级存在匹配图片即视为需要处理；默认仅当前层）
  -v    详细日志（打印每个目录是否被纳入与原因）
  -h    显示帮助
EOF
}

while getopts ":i:o:rnvDh" opt; do
  case "$opt" in
    i) INPUT_ROOT="$OPTARG" ;;
    o) OUTPUT_ROOT="$OPTARG" ;;
    r) RECURSIVE=true ;;
    n) DRYRUN=true ;;
    D) DEEP_CHECK=true ;;
    v) VERBOSE=true ;;
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

# --------------------- 可配置参数 ---------------------
# 允许的图片后缀（大小写不敏感），需要可扩展，例如：jpg|jpeg|png|webp|bmp|tif|tiff
IMAGE_EXTS='jpg|jpeg|png'

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
VALID_GPU_IDS=(2)
TOTAL_GPUS=${#VALID_GPU_IDS[@]}
JOBS_PER_GPU=1

# --------------------- 路径定位到仓库根目录（按需调整层级） ---------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
cd "$REPO_ROOT" || { echo "ERROR: 无法进入仓库根目录 $REPO_ROOT"; exit 1; }

say_do() { if $DRYRUN; then echo "[DRYRUN] $*"; else eval "$@"; fi; }

# 构建用于 find 的后缀匹配表达式（大小写不敏感）
# 使用 -iregex，匹配普通文件或符号链接解析为文件(-type f -o -type l -xtype f)
# 注：GNU find 支持 -regextype posix-extended；大多数 Linux 可用
_find_predicate() {
  local depth_flag="$1"   # -maxdepth 1 或空（递归）
  local dir="$2"
  local regex=".*\.(${IMAGE_EXTS})$"
  # shellcheck disable=SC2016
  echo "find \"$dir\" $depth_flag \\( -type f -o -type l -xtype f \\) -regextype posix-extended -iregex '$regex'"
}

has_images() {
  local d="$1"
  local depth_flag
  if $DEEP_CHECK; then depth_flag=""; else depth_flag="-maxdepth 1"; fi
  # 可靠判定：只要有一条匹配就输出并退出
  local cmd
  cmd=$(_find_predicate "$depth_flag" "$d")
  # shellcheck disable=SC2086
  eval "$cmd -print -quit" | grep -q . 
}

count_images() {
  local d="$1"
  local depth_flag="-maxdepth 1"   # 统计当前层用于切分；处理时我们只读该目录当前层的清单
  local cmd
  cmd=$(_find_predicate "$depth_flag" "$d")
  # shellcheck disable=SC2086
  eval "$cmd" | wc -l | tr -d ' '
}

list_images_sorted() {
  local d="$1"
  local depth_flag="-maxdepth 1"
  local cmd
  cmd=$(_find_predicate "$depth_flag" "$d")
  # shellcheck disable=SC2086
  eval "$cmd" | sort
}

# --------------------- 收集要处理的目录 ---------------------
declare -a DIRS=()
if $RECURSIVE; then
  while IFS= read -r -d '' d; do
    if has_images "$d"; then
      $VERBOSE && echo "[SCAN] 包含图片: $d"
      DIRS+=("$d")
    else
      $VERBOSE && echo "[SCAN] 跳过(无匹配后缀或仅深层且未启用 -D): $d"
    fi
  done < <(find "$INPUT_ROOT" -type d -print0)
else
  while IFS= read -r -d '' d; do
    if has_images "$d"; then
      $VERBOSE && echo "[SCAN] 包含图片: $d"
      DIRS+=("$d")
    else
      $VERBOSE && echo "[SCAN] 跳过(无匹配后缀或仅深层且未启用 -D): $d"
    fi
  done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0)
fi

if (( ${#DIRS[@]} == 0 )); then
  echo "WARN: 未发现包含图片的子目录。"
  exit 0
fi

ROOT_NAME="$(basename "$(realpath "$INPUT_ROOT")")"
echo "[INFO] 预计处理目录数：${#DIRS[@]}"
echo "[INFO] 输出根映射：${OUTPUT_ROOT}/${ROOT_NAME}/<相对路径>"
$DEEP_CHECK && echo "[INFO] 已启用 -D：目录任意层级含图即计入处理目标"
echo "------------------------------------------------------------"

# --------------------- 逐目录处理 ---------------------
for INPUT_DIR in "${DIRS[@]}"; do
  REL_PATH="$(realpath --relative-to="$INPUT_ROOT" "$INPUT_DIR")"
  OUTPUT_DIR="${OUTPUT_ROOT}/${ROOT_NAME}/${REL_PATH}"

  NUM_IMAGES=$(count_images "$INPUT_DIR")

  echo "[INFO] 目录: $INPUT_DIR"
  echo "       输出: $OUTPUT_DIR"
  echo "       当前层匹配图片: ${NUM_IMAGES} 张 (扩展名: ${IMAGE_EXTS})"

  if (( NUM_IMAGES == 0 )); then
    if $DEEP_CHECK; then
      echo "       提示：该目录被计入是因为其下更深层含有图片（-D 启用）。"
      echo "       但处理阶段仅使用该目录当前层图片。"
    fi
    echo "       -> 当前层无图片，跳过处理（如需处理更深层图片，请进入相应子目录或改造流程）。"
    echo "------------------------------------------------------------"
    continue
  fi

  # 预创建输出目录
  say_do "mkdir -p \"${OUTPUT_DIR}\""

  # 生成图片清单（只当前层）
  IMAGE_LIST="${INPUT_DIR}/image_list.txt"
  if $DRYRUN; then
    echo "[DRYRUN] 将创建清单: ${IMAGE_LIST}"
  else
    list_images_sorted "${INPUT_DIR}" > "${IMAGE_LIST}"
  fi

  # 作业切分
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

  # 切分到 image_paths_*.txt
  start=1
  for ((i=1; i<=TOTAL_JOBS; i++)); do
    if (( i <= REM )); then count=$((BASE + 1)); else count=$BASE; fi
    text_file="${INPUT_DIR}/image_paths_${i}.txt"
    if $DRYRUN; then
      echo "[DRYRUN] 将创建分片清单: ${text_file} (行数: ${count}, 区间: ${start}-$((start+count-1)))"
    else
      sed -n "${start},$((start+count-1))p" "${IMAGE_LIST}" > "${text_file}"
    fi
    start=$((start + count))
  done

  # 执行每个作业
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
      sleep 1
    fi
  done

  # 清理临时清单
  if ! $DRYRUN; then
    rm -f "${IMAGE_LIST}"
    for ((i=1; i<=TOTAL_JOBS; i++)); do rm -f "${INPUT_DIR}/image_paths_${i}.txt"; done
  fi

  echo "------------------------------------------------------------"
done

echo $($DRYRUN && echo "[DRYRUN] 完成：仅打印流程，未执行推理。" || echo "全部目录处理完成。")
