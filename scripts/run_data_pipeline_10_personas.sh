#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/runs/persona_pruner_10p}"
GENERIC_DATA="${GENERIC_DATA:-${ROOT_DIR}/data/generic_instruction_data.example.json}"
PERSONAS="${PERSONAS:-${ROOT_DIR}/data/personas/alpaca_reference_personas_10.jsonl}"
REWRITE_MODEL="${REWRITE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
SCORER_MODEL="${SCORER_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
GENERATOR_BACKEND="${GENERATOR_BACKEND:-vllm}"
GENERATOR_BATCH_SIZE="${GENERATOR_BATCH_SIZE:-1}"
SCORER_BATCH_SIZE="${SCORER_BATCH_SIZE:-128}"
TRAIN_LIMIT="${TRAIN_LIMIT:-20000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
REWRITE_GPU_IDS="${REWRITE_GPU_IDS:-0,1}"
SCORE_GPU_IDS="${SCORE_GPU_IDS:-${GPU_IDS}}"
REWRITE_TENSOR_PARALLEL_SIZE="${REWRITE_TENSOR_PARALLEL_SIZE:-}"
REWRITE_MAX_TOKENS="${REWRITE_MAX_TOKENS:-2048}"
REWRITE_MAX_INPUT_LEN="${REWRITE_MAX_INPUT_LEN:-2048}"
REWRITE_TEMPERATURE="${REWRITE_TEMPERATURE:-1.0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"

mkdir -p "${WORK_DIR}"/{prompts,rewrites,chatml,scores,selected,logs}
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

run_user() {
  local user_id="$1"
  IFS=',' read -r -a score_gpus <<< "${SCORE_GPU_IDS}"
  local gpu="${score_gpus[$((user_id % ${#score_gpus[@]}))]}"
  local max_args=()
  if [[ -n "${MAX_EXAMPLES}" ]]; then
    max_args=(--max-examples "${MAX_EXAMPLES}")
  fi

  python "${ROOT_DIR}/scripts/build_rewrite_prompts.py" \
    --alpaca-data "${GENERIC_DATA}" \
    --personas "${PERSONAS}" \
    --persona-index "${user_id}" \
    --output "${WORK_DIR}/prompts/user_${user_id}.jsonl" \
    "${max_args[@]}"

  if [[ "${SKIP_GENERATION}" != "1" ]]; then
    if [[ "${GENERATOR_BACKEND}" == "vllm" ]]; then
      tp_args=()
      if [[ -n "${REWRITE_TENSOR_PARALLEL_SIZE}" ]]; then
        tp_args=(--tensor-parallel-size "${REWRITE_TENSOR_PARALLEL_SIZE}")
      fi
      python "${ROOT_DIR}/scripts/generate_rewrites_vllm.py" \
        --model-name "${REWRITE_MODEL}" \
        --input-file "${WORK_DIR}/prompts/user_${user_id}.jsonl" \
        --output-file "${WORK_DIR}/rewrites/user_${user_id}.json" \
        --gpu-ids "${REWRITE_GPU_IDS}" \
        "${tp_args[@]}" \
        --batch-size "${GENERATOR_BATCH_SIZE}" \
        --max-tokens "${REWRITE_MAX_TOKENS}" \
        --max-input-len "${REWRITE_MAX_INPUT_LEN}" \
        --temperature "${REWRITE_TEMPERATURE}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN}" \
        --generator "${REWRITE_MODEL}" \
        > "${WORK_DIR}/logs/generate_user_${user_id}.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES="${REWRITE_GPU_IDS}" python "${ROOT_DIR}/scripts/generate_rewrites_hf.py" \
        --model "${REWRITE_MODEL}" \
        --prompts "${WORK_DIR}/prompts/user_${user_id}.jsonl" \
        --output "${WORK_DIR}/rewrites/user_${user_id}.json" \
        --batch-size "${GENERATOR_BATCH_SIZE}" \
        --max-new-tokens "${REWRITE_MAX_TOKENS}" \
        --temperature "${REWRITE_TEMPERATURE}" \
        > "${WORK_DIR}/logs/generate_user_${user_id}.log" 2>&1
    fi
  fi

  python "${ROOT_DIR}/scripts/convert_rewrites_to_chatml.py" \
    --rewrites "${WORK_DIR}/rewrites/user_${user_id}.json" \
    --output "${WORK_DIR}/chatml/user_${user_id}.json"

  CUDA_VISIBLE_DEVICES="${gpu}" python "${ROOT_DIR}/scripts/score_persona_divergence.py" \
    --model "${SCORER_MODEL}" \
    --data "${WORK_DIR}/chatml/user_${user_id}.json" \
    --personas "${PERSONAS}" \
    --target-persona-idx "${user_id}" \
    --n-compare-personas 9 \
    --total-personas 10 \
    --layer-start 5 \
    --layer-end 25 \
    --batch-size "${SCORER_BATCH_SIZE}" \
    --exclude-eval-file "${ROOT_DIR}/data/eval/persona_alpaca_specific_user_${user_id}.json" \
    --output "${WORK_DIR}/scores/user_${user_id}_divergence.json" \
    > "${WORK_DIR}/logs/score_user_${user_id}.log" 2>&1

  python "${ROOT_DIR}/scripts/select_persona_training_data.py" \
    --rewritten-data "${WORK_DIR}/chatml/user_${user_id}.json" \
    --train-scores "${WORK_DIR}/scores/user_${user_id}_divergence.json" \
    --train-limit "${TRAIN_LIMIT}" \
    --exclude-eval-file "${ROOT_DIR}/data/eval/persona_alpaca_specific_user_${user_id}.json" \
    --output "${WORK_DIR}/selected/user_${user_id}_top_${TRAIN_LIMIT}.json"
}

for user_id in {0..9}; do
  run_user "${user_id}" &
  while [[ "$(jobs -rp | wc -l)" -ge "${PARALLEL_JOBS}" ]]; do
    sleep 5
  done
done
wait

echo "Done. Selected data are in ${WORK_DIR}/selected"
