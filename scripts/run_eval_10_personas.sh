#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/persona_pruner_10p}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-${WORK_DIR}/pruned_models/user_{user_id}_6144}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/eval_alpaca_specific_deterministic}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

python -m persona_pruner.batch_eval \
  --model-template "${MODEL_TEMPLATE}" \
  --trait-prefix persona_alpaca_specific \
  --data-dir "${ROOT_DIR}/data/eval" \
  --output-dir "${OUT_DIR}" \
  --judge-model "${JUDGE_MODEL}" \
  --deterministic \
  --seed 42 \
  --max-tokens 2048

echo "Done. Evaluation results are in ${OUT_DIR}"
