#!/usr/bin/env bash
set -euo pipefail

# Eval-only runner: evaluate epoch_* checkpoints under an OUT_DIR with 3-way parallel jobs.
#
# This script does NOT train. It runs AnomalyGPT-aligned evaluation for each epoch checkpoint:
#   - MVTec: eval/AnomalyGPT_eval_mvtec.py
#   - VisA : eval/AnomalyGPT_eval_visa.py
#
# Typical usage:
#   bash train/scripts/eval_anomalygpt_parallel3_from_outdir.sh \
#     --dataset mvtec \
#     --out_dir /path/to/outputs/... \
#     --start_epoch 1 --end_epoch 100 \
#     --gpus 0,1,2

usage() {
  cat <<'EOF'
Usage:
  bash train/scripts/eval_anomalygpt_parallel3_from_outdir.sh --dataset {mvtec|visa} --out_dir OUT_DIR [options]

Required:
  --dataset            mvtec|visa
  --out_dir            Training output directory containing epoch_*/ subfolders

Options (paths):
  --model_path         Qwen3-VL model path (default: env.sh MODEL_PATH)
  --sam_checkpoint     SAM checkpoint path (default: env.sh SAM_CKPT)
  --mvtec_root         MVTec-AD root (default: env.sh MVTEC_ROOT)
  --visa_root          VisA root (default: env.sh VISA_ROOT)

Options (epochs):
  --start_epoch        First epoch to eval (default: 1)
  --end_epoch          Last epoch to eval (default: inferred from epoch_* dirs)

Options (eval):
  --gpus               Comma-separated GPU ids for parallel jobs (default: $CUDA_VISIBLE_DEVICES; if empty, no pinning)
  --categories         Comma-separated categories (default: all)
  --limit_per_class    Limit test samples per class (default: full)
  --inference_strategy teacher_forcing|generate (default: teacher_forcing)
  --eval_size          Resize pred/gt masks to this size (default: 224)
  --prompt_source      For mvtec: mvtec_templates|anomalygpt_question; for visa: visa_templates|anomalygpt_question (default: anomalygpt_question)
  --user_question      Used only when prompt_source=anomalygpt_question
  --assistant_stub     Used for teacher_forcing inference
  --max_new_tokens     Used only for generate inference (default: 64)
  --seed               Eval seed (default: 0)

Outputs:
  OUT_DIR/eval_anomalygpt_parallel3/
    epoch_*.json / epoch_*.log / epoch_*.stdout.log
    summary.csv
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Path defaults (DATA_ROOT / MODEL_PATH / SAM_CKPT / MVTEC_ROOT / VISA_ROOT / DTD_ROOT)
# Edit `train/scripts/env.sh` once to adapt paths on a new machine.
source "${SCRIPT_DIR}/env.sh"

DATASET=""
OUT_DIR=""

MODEL_PATH_ARG="${MODEL_PATH}"
SAM_CKPT_ARG="${SAM_CKPT}"
MVTEC_ROOT_ARG="${MVTEC_ROOT}"
VISA_ROOT_ARG="${VISA_ROOT}"

START_EPOCH=1
END_EPOCH=""

GPUS="${CUDA_VISIBLE_DEVICES:-}"
EVAL_PARALLEL=3

CATEGORIES=""
LIMIT_PER_CLASS=""
INFERENCE_STRATEGY="teacher_forcing"
EVAL_SIZE=224
PROMPT_SOURCE="anomalygpt_question"
USER_QUESTION="You are an expert industrial inspector. Please inspect this image strictly and determine whether the object is normal or defective. Always provide a segmentation mask. If the object is normal, the mask should be empty."
ASSISTANT_STUB="Segmentation mask: <SEG>."
MAX_NEW_TOKENS=64
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"; shift 2 ;;
    --out_dir)
      OUT_DIR="${2:-}"; shift 2 ;;
    --model_path)
      MODEL_PATH_ARG="${2:-}"; shift 2 ;;
    --sam_checkpoint)
      SAM_CKPT_ARG="${2:-}"; shift 2 ;;
    --mvtec_root)
      MVTEC_ROOT_ARG="${2:-}"; shift 2 ;;
    --visa_root)
      VISA_ROOT_ARG="${2:-}"; shift 2 ;;
    --start_epoch)
      START_EPOCH="${2:-}"; shift 2 ;;
    --end_epoch)
      END_EPOCH="${2:-}"; shift 2 ;;
    --gpus)
      GPUS="${2:-}"; shift 2 ;;
    --categories)
      CATEGORIES="${2:-}"; shift 2 ;;
    --limit_per_class)
      LIMIT_PER_CLASS="${2:-}"; shift 2 ;;
    --inference_strategy)
      INFERENCE_STRATEGY="${2:-}"; shift 2 ;;
    --eval_size)
      EVAL_SIZE="${2:-}"; shift 2 ;;
    --prompt_source)
      PROMPT_SOURCE="${2:-}"; shift 2 ;;
    --user_question)
      USER_QUESTION="${2:-}"; shift 2 ;;
    --assistant_stub)
      ASSISTANT_STUB="${2:-}"; shift 2 ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="${2:-}"; shift 2 ;;
    --seed)
      SEED="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[eval] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${DATASET}" || -z "${OUT_DIR}" ]]; then
  echo "[eval] --dataset and --out_dir are required." >&2
  usage
  exit 2
fi

EVAL_PY=""
DATA_ARG=""
DATA_ROOT_PATH=""
case "${DATASET}" in
  mvtec)
    EVAL_PY="eval/AnomalyGPT_eval_mvtec.py"
    DATA_ARG="--mvtec_root"
    DATA_ROOT_PATH="${MVTEC_ROOT_ARG}"
    ;;
  visa)
    EVAL_PY="eval/AnomalyGPT_eval_visa.py"
    DATA_ARG="--visa_root"
    DATA_ROOT_PATH="${VISA_ROOT_ARG}"
    ;;
  *)
    echo "[eval] Unsupported --dataset=${DATASET} (expected mvtec|visa)." >&2
    exit 2
    ;;
esac

if [[ -z "${END_EPOCH}" ]]; then
  # Infer END_EPOCH from epoch_* subfolders.
  # Works even when some epochs are missing.
  END_EPOCH="$(
    find "${OUT_DIR}" -maxdepth 1 -type d -name 'epoch_*' -printf '%f\n' \
      | sed -E 's/^epoch_([0-9]+)$/\\1/' \
      | grep -E '^[0-9]+$' \
      | sort -n \
      | tail -n 1 \
      || true
  )"
  if [[ -z "${END_EPOCH}" ]]; then
    echo "[eval] Failed to infer --end_epoch from ${OUT_DIR}/epoch_*" >&2
    exit 2
  fi
fi

if ! [[ "${START_EPOCH}" =~ ^[0-9]+$ && "${END_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "[eval] start/end epoch must be integers, got start=${START_EPOCH} end=${END_EPOCH}" >&2
  exit 2
fi
if (( START_EPOCH < 1 || END_EPOCH < START_EPOCH )); then
  echo "[eval] Invalid epoch range: start=${START_EPOCH} end=${END_EPOCH}" >&2
  exit 2
fi

EVAL_DIR="${OUT_DIR}/eval_anomalygpt_parallel3"
mkdir -p "${EVAL_DIR}"
SUMMARY_CSV="${EVAL_DIR}/summary.csv"
echo "epoch,mean_i_AUROC,mean_p_AUROC,total_time_sec" > "${SUMMARY_CSV}"

echo "[eval] ROOT_DIR=${ROOT_DIR}"
echo "[eval] OUT_DIR=${OUT_DIR}"
echo "[eval] EVAL_DIR=${EVAL_DIR}"
echo "[eval] dataset=${DATASET} | epochs=${START_EPOCH}..${END_EPOCH} | parallel=${EVAL_PARALLEL} | gpus=${GPUS:-<none>}"
echo "[eval] model_path=${MODEL_PATH_ARG}"
echo "[eval] sam_checkpoint=${SAM_CKPT_ARG}"
echo "[eval] data_root=${DATA_ROOT_PATH}"

IFS=',' read -r -a _GPUS <<< "${GPUS}"
if [[ -z "${GPUS}" ]]; then
  _GPUS=("")
fi

cd "${ROOT_DIR}"

for ((batch_start=START_EPOCH; batch_start<=END_EPOCH; batch_start+=EVAL_PARALLEL)); do
  pids=()
  epochs_in_batch=()

  for ((i=0; i<EVAL_PARALLEL && batch_start+i<=END_EPOCH; i++)); do
    epoch=$((batch_start+i))
    epochs_in_batch+=("${epoch}")

    EPOCH_DIR="${OUT_DIR}/epoch_${epoch}"
    if [[ ! -d "${EPOCH_DIR}" ]]; then
      echo "[eval] Missing checkpoint dir: ${EPOCH_DIR}" >&2
      exit 1
    fi
    if [[ ! -f "${EPOCH_DIR}/text_proj.pt" ]]; then
      echo "[eval] Missing: ${EPOCH_DIR}/text_proj.pt" >&2
      exit 1
    fi
    if [[ ! -f "${EPOCH_DIR}/sam_mask_decoder.pt" ]]; then
      echo "[eval] Missing: ${EPOCH_DIR}/sam_mask_decoder.pt" >&2
      exit 1
    fi

    EVAL_LOG="${EVAL_DIR}/epoch_${epoch}.log"
    EVAL_JSON="${EVAL_DIR}/epoch_${epoch}.json"
    EVAL_STDOUT="${EVAL_DIR}/epoch_${epoch}.stdout.log"

    EVAL_ARGS=(
      --model_path "${MODEL_PATH_ARG}"
      --sam_checkpoint "${SAM_CKPT_ARG}"
      "${DATA_ARG}" "${DATA_ROOT_PATH}"
      --lora_path "${EPOCH_DIR}"
      --text_proj "${EPOCH_DIR}/text_proj.pt"
      --sam_mask_decoder "${EPOCH_DIR}/sam_mask_decoder.pt"
      --inference_strategy "${INFERENCE_STRATEGY}"
      --eval_size "${EVAL_SIZE}"
      --prompt_source "${PROMPT_SOURCE}"
      --user_question "${USER_QUESTION}"
      --assistant_stub "${ASSISTANT_STUB}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --seed "${SEED}"
      --output_json "${EVAL_JSON}"
      --log_file "${EVAL_LOG}"
    )

    if [[ -n "${CATEGORIES}" ]]; then
      EVAL_ARGS+=(--categories "${CATEGORIES}")
    fi
    if [[ -n "${LIMIT_PER_CLASS}" ]]; then
      EVAL_ARGS+=(--limit_per_class "${LIMIT_PER_CLASS}")
    fi

    gpu="${_GPUS[i % ${#_GPUS[@]}]}"
    if [[ -n "${gpu}" ]]; then
      echo "[eval] Launch epoch ${epoch} on GPU ${gpu}..."
      CUDA_VISIBLE_DEVICES="${gpu}" python "${EVAL_PY}" "${EVAL_ARGS[@]}" > "${EVAL_STDOUT}" 2>&1 &
    else
      echo "[eval] Launch epoch ${epoch}..."
      python "${EVAL_PY}" "${EVAL_ARGS[@]}" > "${EVAL_STDOUT}" 2>&1 &
    fi
    pids+=("$!")
  done

  for j in "${!pids[@]}"; do
    pid="${pids[j]}"
    epoch="${epochs_in_batch[j]}"
    if ! wait "${pid}"; then
      echo "[eval] Eval failed for epoch ${epoch}. Check: ${EVAL_DIR}/epoch_${epoch}.stdout.log" >&2
      exit 1
    fi
  done

  for epoch in "${epochs_in_batch[@]}"; do
    EVAL_JSON="${EVAL_DIR}/epoch_${epoch}.json"
    if [[ ! -f "${EVAL_JSON}" ]]; then
      echo "[eval] Missing eval json: ${EVAL_JSON}" >&2
      exit 1
    fi
    python - <<PY >> "${SUMMARY_CSV}"
import json
p = r"${EVAL_JSON}"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
epoch = int(${epoch})
i_auc = d.get("mean_i_AUROC")
p_auc = d.get("mean_p_AUROC")
t = d.get("total_time_sec")
print(f"{epoch},{i_auc},{p_auc},{t}")
PY
  done
done

echo "[eval] Done."
echo "  - Eval dir: ${EVAL_DIR}"
echo "  - Summary: ${SUMMARY_CSV}"
