#!/usr/bin/env bash
set -euo pipefail

# Example: evaluate 10 user-specific Persona-Pruner checkpoints on Alpaca-P specific personas.
# Replace MODEL_TEMPLATE with your checkpoint naming convention.

MODEL_TEMPLATE="${MODEL_TEMPLATE:-/path/to/persona_pruner_alpaca_user_{user_id}_6144}"
OUT_DIR="${OUT_DIR:-results/alpaca_specific_3b_6144_deterministic}"

python -m persona_pruner.batch_eval \
  --model-template "${MODEL_TEMPLATE}" \
  --trait-prefix persona_alpaca_specific \
  --user-ids 0-9 \
  --data-dir data/eval \
  --output-dir "${OUT_DIR}" \
  --deterministic \
  --seed 42 \
  --temperature 1.0 \
  --batch-size 8 \
  --max-tokens 2048 \
  --judge-model "${JUDGE_MODEL:-gpt-4o-mini}"

