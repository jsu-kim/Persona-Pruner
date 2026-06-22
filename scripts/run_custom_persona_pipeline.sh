#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/custom_persona}"
GENERIC_DATA="${GENERIC_DATA:-${ROOT_DIR}/data/generic_instruction_data.example.json}"
REFERENCE_PERSONAS="${REFERENCE_PERSONAS:-${ROOT_DIR}/data/personas/alpaca_reference_personas_10.jsonl}"
REWRITE_MODEL="${REWRITE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
SCORER_MODEL="${SCORER_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
GENERATOR_BACKEND="${GENERATOR_BACKEND:-vllm}"
GENERATOR_BATCH_SIZE="${GENERATOR_BATCH_SIZE:-1}"
SCORER_BATCH_SIZE="${SCORER_BATCH_SIZE:-128}"
TRAIN_LIMIT="${TRAIN_LIMIT:-20000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
CONDITION_ANSWER="${CONDITION_ANSWER:-1}"
TOPK_INDICES="${TOPK_INDICES:-}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
REWRITE_GPU_IDS="${REWRITE_GPU_IDS:-${GPU_ID}}"
REWRITE_TENSOR_PARALLEL_SIZE="${REWRITE_TENSOR_PARALLEL_SIZE:-}"
REWRITE_MAX_TOKENS="${REWRITE_MAX_TOKENS:-2048}"
REWRITE_MAX_INPUT_LEN="${REWRITE_MAX_INPUT_LEN:-2048}"
REWRITE_TEMPERATURE="${REWRITE_TEMPERATURE:-1.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"

mkdir -p "${WORK_DIR}"/{prompts,rewrites,chatml,scores,selected,logs,pruned_model}

if [[ -z "${PERSONA_TEXT:-}" && -z "${PERSONA_FILE:-}" ]]; then
  echo "Set PERSONA_TEXT or PERSONA_FILE for the custom persona." >&2
  exit 1
fi

if [[ -n "${PERSONA_FILE:-}" ]]; then
  persona_arg=(--persona-file "${PERSONA_FILE}")
  CUSTOM_PERSONA_TEXT="$(cat "${PERSONA_FILE}")"
else
  persona_arg=(--persona-text "${PERSONA_TEXT}")
  CUSTOM_PERSONA_TEXT="${PERSONA_TEXT}"
fi
export CUSTOM_PERSONA_TEXT

python - <<PY
import json
import os
from pathlib import Path
out = Path("${WORK_DIR}/custom_with_reference_personas.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
custom = os.environ["CUSTOM_PERSONA_TEXT"]
with out.open("w", encoding="utf-8") as f:
    f.write(json.dumps(custom, ensure_ascii=False) + "\\n")
    with open("${REFERENCE_PERSONAS}", "r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                f.write(line)
print(out)
PY

condition_args=(--condition-answer)
if [[ "${CONDITION_ANSWER}" == "0" ]]; then
  condition_args=(--no-condition-answer)
fi
max_args=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  max_args=(--max-examples "${MAX_EXAMPLES}")
fi

python "${ROOT_DIR}/scripts/build_rewrite_prompts.py" \
  --alpaca-data "${GENERIC_DATA}" \
  "${persona_arg[@]}" \
  --output "${WORK_DIR}/prompts/custom.jsonl" \
  "${max_args[@]}" \
  "${condition_args[@]}"

if [[ "${GENERATOR_BACKEND}" == "vllm" ]]; then
  tp_args=()
  if [[ -n "${REWRITE_TENSOR_PARALLEL_SIZE}" ]]; then
    tp_args=(--tensor-parallel-size "${REWRITE_TENSOR_PARALLEL_SIZE}")
  fi
  python "${ROOT_DIR}/scripts/generate_rewrites_vllm.py" \
    --model-name "${REWRITE_MODEL}" \
    --input-file "${WORK_DIR}/prompts/custom.jsonl" \
    --output-file "${WORK_DIR}/rewrites/custom.json" \
    --gpu-ids "${REWRITE_GPU_IDS}" \
    "${tp_args[@]}" \
    --batch-size "${GENERATOR_BATCH_SIZE}" \
    --max-tokens "${REWRITE_MAX_TOKENS}" \
    --max-input-len "${REWRITE_MAX_INPUT_LEN}" \
    --temperature "${REWRITE_TEMPERATURE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --generator "${REWRITE_MODEL}" \
    > "${WORK_DIR}/logs/generate_custom.log" 2>&1
else
  CUDA_VISIBLE_DEVICES="${REWRITE_GPU_IDS}" python "${ROOT_DIR}/scripts/generate_rewrites_hf.py" \
    --model "${REWRITE_MODEL}" \
    --prompts "${WORK_DIR}/prompts/custom.jsonl" \
    --output "${WORK_DIR}/rewrites/custom.json" \
    --batch-size "${GENERATOR_BATCH_SIZE}" \
    --max-new-tokens "${REWRITE_MAX_TOKENS}" \
    --temperature "${REWRITE_TEMPERATURE}" \
    > "${WORK_DIR}/logs/generate_custom.log" 2>&1
fi

python "${ROOT_DIR}/scripts/convert_rewrites_to_chatml.py" \
  --rewrites "${WORK_DIR}/rewrites/custom.json" \
  --output "${WORK_DIR}/chatml/custom.json"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/scripts/score_persona_divergence.py" \
  --model "${SCORER_MODEL}" \
  --data "${WORK_DIR}/chatml/custom.json" \
  --personas "${WORK_DIR}/custom_with_reference_personas.jsonl" \
  --target-persona-idx 0 \
  --n-compare-personas 10 \
  --total-personas 11 \
  --layer-start 5 \
  --layer-end 25 \
  --batch-size "${SCORER_BATCH_SIZE}" \
  --output "${WORK_DIR}/scores/custom_divergence.json" \
  > "${WORK_DIR}/logs/score_custom.log" 2>&1

python "${ROOT_DIR}/scripts/select_persona_training_data.py" \
  --rewritten-data "${WORK_DIR}/chatml/custom.json" \
  --train-scores "${WORK_DIR}/scores/custom_divergence.json" \
  --train-limit "${TRAIN_LIMIT}" \
  --output "${WORK_DIR}/selected/custom_top_${TRAIN_LIMIT}.json"

if [[ -n "${TOPK_INDICES}" ]]; then
  python "${ROOT_DIR}/scripts/export_pruned_model.py" \
    --base-model "${BASE_MODEL}" \
    --topk-indices "${TOPK_INDICES}" \
    --output-dir "${WORK_DIR}/pruned_model" \
    --dtype bf16
  echo "Custom selected data and pruned model are in ${WORK_DIR}"
else
  echo "Custom selected data are in ${WORK_DIR}/selected"
  echo "Set TOPK_INDICES=/path/to/keep_indices.pt to also export a structurally pruned model."
fi
