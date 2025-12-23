#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
./diagnose_pose_hang.sh \
    --input "/data/xxt/sapiens_data/test" \
    --output "/data/xxt/sapiens_code/sapiens/pose/Outputs/vis/itw_videos/test" \
    --checkpoint "/data/${USER}/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b/sapiens_2b_coco_best_coco_AP_822_torchscript.pt2" \
    --det-config "../pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py" \
    --det-checkpoint "/data/${USER}/sapiens_lite_host/torchscript/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" \
    --num-keypoints 17 \
    --batch-size 1 \
    --max-images 4 \
    --start-method spawn \
    --timeout 600

Notes:
  - This script does not modify any source code.
  - It builds a small image list (default 4 images) and runs vis_pose.py.
  - Use --start-method spawn to avoid fork-related hangs with CUDA.
EOF
}

INPUT=""
OUTPUT=""
CHECKPOINT=""
DET_CONFIG=""
DET_CHECKPOINT=""
NUM_KEYPOINTS=17
BATCH_SIZE=1
MAX_IMAGES=4
DEVICE="cuda:1"
START_METHOD="default"
TIMEOUT_SEC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --det-config) DET_CONFIG="$2"; shift 2 ;;
    --det-checkpoint) DET_CHECKPOINT="$2"; shift 2 ;;
    --num-keypoints) NUM_KEYPOINTS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-images) MAX_IMAGES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --start-method) START_METHOD="$2"; shift 2 ;;
    --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "${INPUT}" || -z "${OUTPUT}" || -z "${CHECKPOINT}" || -z "${DET_CONFIG}" || -z "${DET_CHECKPOINT}" ]]; then
  echo "Missing required arguments."
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_FILE="demo/vis_pose.py"

echo "== Diagnose Summary =="
echo "pwd: ${PWD}"
echo "script_dir: ${SCRIPT_DIR}"
echo "lite_root: ${LITE_ROOT}"
echo "input: ${INPUT}"
echo "output: ${OUTPUT}"
echo "checkpoint: ${CHECKPOINT}"
echo "det_config: ${DET_CONFIG}"
echo "det_checkpoint: ${DET_CHECKPOINT}"
echo "num_keypoints: ${NUM_KEYPOINTS}"
echo "batch_size: ${BATCH_SIZE}"
echo "max_images: ${MAX_IMAGES}"
echo "device: ${DEVICE}"
echo "start_method: ${START_METHOD}"
echo "timeout_sec: ${TIMEOUT_SEC}"

echo
echo "== System Info =="
date
uname -a || true
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "nvidia-smi not found"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | sed -n '1,5p' || true

echo
echo "== Python/Torch =="
python - <<'PY'
import multiprocessing as mp
import torch
print("python:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("cudnn_version:", torch.backends.cudnn.version())
print("start_method:", mp.get_start_method(allow_none=True))
PY

if [[ -d "${INPUT}" ]]; then
  INPUT_DIR="${INPUT}"
  IMAGE_LIST="$(mktemp --suffix=.txt)"
  find "${INPUT}" -type f \( -iname "*.jpg" -o -iname "*.png" \) | sort | head -n "${MAX_IMAGES}" > "${IMAGE_LIST}"
elif [[ -f "${INPUT}" ]]; then
  INPUT_DIR="$(dirname "$(head -n 1 "${INPUT}")")"
  IMAGE_LIST="$(mktemp --suffix=.txt)"
  head -n "${MAX_IMAGES}" "${INPUT}" > "${IMAGE_LIST}"
else
  echo "Input not found: ${INPUT}"
  exit 2
fi

trap 'rm -f "${IMAGE_LIST}"' EXIT

echo
echo "== Input Probe =="
echo "input_dir: ${INPUT_DIR}"
stat -f -c "fs_type: %T" "${INPUT_DIR}" 2>/dev/null || true
echo "sample_images:"
wc -l "${IMAGE_LIST}" | awk '{print $1 " images"}'
sed -n '1,5p' "${IMAGE_LIST}"

mkdir -p "${OUTPUT}"

RUN_ARGS=(
  "${CHECKPOINT}"
  --num_keypoints "${NUM_KEYPOINTS}"
  --det-config "${DET_CONFIG}"
  --det-checkpoint "${DET_CHECKPOINT}"
  --batch-size "${BATCH_SIZE}"
  --input "${IMAGE_LIST}"
  --output-root "${OUTPUT}"
  --device "${DEVICE}"
)
RUN_CMD=(python -u "${RUN_FILE}" "${RUN_ARGS[@]}")

echo
echo "== Run =="
echo "command: (cd ${LITE_ROOT} && ${RUN_CMD[*]})"

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${LITE_ROOT}/demo:${PYTHONPATH:-}"

cd "${LITE_ROOT}"

if [[ "${START_METHOD}" == "spawn" ]]; then
  PY_WRAP="import multiprocessing as mp; mp.set_start_method('spawn', force=True); import sys; sys.path.insert(0, '${LITE_ROOT}/demo'); import runpy; runpy.run_path('${RUN_FILE}', run_name='__main__')"
  if [[ "${TIMEOUT_SEC}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${TIMEOUT_SEC}"s python -u -c "${PY_WRAP}" "${RUN_ARGS[@]}"
  else
    python -u -c "${PY_WRAP}" "${RUN_ARGS[@]}"
  fi
else
  if [[ "${TIMEOUT_SEC}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${TIMEOUT_SEC}"s "${RUN_CMD[@]}"
  else
    "${RUN_CMD[@]}"
  fi
fi
