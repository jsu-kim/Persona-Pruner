#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/persona_pruner_10p}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
TOPK_TEMPLATE="${TOPK_TEMPLATE:-${WORK_DIR}/topk_indices/user_{user_id}_6144.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/pruned_models}"
DTYPE="${DTYPE:-bf16}"

mkdir -p "${OUTPUT_DIR}"

for user_id in {0..9}; do
  topk_path="${TOPK_TEMPLATE//\{user_id\}/${user_id}}"
  if [[ ! -f "${topk_path}" ]]; then
    echo "Missing topk indices for user ${user_id}: ${topk_path}" >&2
    exit 1
  fi
  python "${ROOT_DIR}/scripts/export_pruned_model.py" \
    --base-model "${BASE_MODEL}" \
    --topk-indices "${topk_path}" \
    --output-dir "${OUTPUT_DIR}/user_${user_id}_6144" \
    --dtype "${DTYPE}"
done

echo "Done. Pruned models are in ${OUTPUT_DIR}"
