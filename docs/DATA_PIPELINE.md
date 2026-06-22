# Data Pipeline

Persona-Pruner uses generic instruction data that has been rewritten into a target persona voice, then filters questions by how strongly the target model separates that persona from comparison personas.

For the paper-style Alpaca-specific setting, each user's 20 `persona_alpaca_specific` evaluation questions must be excluded from train scoring and selection. The public scripts expose this as `--exclude-eval-file`.

The convenience runner for the 10 reference personas is:

```bash
GENERIC_DATA=/path/to/preprocessed_generic_instruction_data.json \
REWRITE_MODEL=meta-llama/Llama-3.1-70B-Instruct \
REWRITE_GPU_IDS=0,2 \
bash scripts/run_data_pipeline_10_personas.sh
```

It writes:

- rewrite prompts to `runs/persona_pruner_10p/prompts/`
- generated rewrites to `runs/persona_pruner_10p/rewrites/`
- ChatML data to `runs/persona_pruner_10p/chatml/`
- divergence scores to `runs/persona_pruner_10p/scores/`
- selected train data to `runs/persona_pruner_10p/selected/`

Rewrite generation uses `scripts/generate_rewrites_vllm.py` by default. `REWRITE_GPU_IDS` controls `CUDA_VISIBLE_DEVICES`; tensor parallel size defaults to the number of rewrite GPUs unless `REWRITE_TENSOR_PARALLEL_SIZE` is set.

For 70B-class rewrite models, keep `PARALLEL_JOBS=1` unless each concurrent generation process gets a separate GPU group. Divergence scoring can still be distributed with `SCORE_GPU_IDS`.

## 1. Build Rewrite Prompts

Start with Alpaca-style rows containing `instruction`, optional `input`, and `output`.

```bash
python scripts/build_rewrite_prompts.py \
  --alpaca-data data/alpaca_train.json \
  --personas data/personas.jsonl \
  --persona-index 0 \
  --output data/rewrite_prompts_user_0.jsonl
```

For a custom persona, replace `--personas/--persona-index` with `--persona-text` or `--persona-file`. Add `--no-condition-answer` to omit the original reference answer from the rewrite prompt.
The convenience custom runner also accepts `MAX_EXAMPLES=2 TRAIN_LIMIT=2` for smoke tests.

Each output line is a two-message chat prompt. Use your generator model of choice to produce records with `instruction` and `output` fields. In the paper experiments, rewritten answers were produced with a 70B-class instruction model.

To run the bundled vLLM generator directly:

```bash
python scripts/generate_rewrites_vllm.py \
  --model-name meta-llama/Llama-3.1-70B-Instruct \
  --input-file data/rewrite_prompts_user_0.jsonl \
  --output-file results/alpaca_rewritten_user_0.json \
  --gpu-ids 0,2 \
  --batch-size 256 \
  --max-input-len 2048 \
  --max-tokens 2048
```

## 2. Convert Rewrites to ChatML

```bash
python scripts/convert_rewrites_to_chatml.py \
  --rewrites results/alpaca_rewritten_user_0.json \
  --output data/persona_alpaca_user_0_rewritten.json
```

The output format is:

```json
{
  "train": [
    [
      {"role": "system", "content": "You are roleplaying as..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  ]
}
```

## 3. Score Persona Divergence

The filtering score is computed from hidden-state divergence:

```bash
python scripts/score_persona_divergence.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --data data/persona_alpaca_user_0_rewritten.json \
  --personas data/personas.jsonl \
  --target-persona-idx 0 \
  --n-compare-personas 9 \
  --total-personas 10 \
  --layer-start 5 \
  --layer-end 25 \
  --batch-size 8 \
  --exclude-eval-file data/eval/persona_alpaca_specific_user_0.json \
  --output scores/user_0_divergence.json
```

Higher `divergence_score` means the question is more persona-specific. For quick checks, add `--limit 8`; for the paper test-split sampling behavior, add `--shuffle-before-limit --limit 2000`.

## 4. Select Data

```bash
python scripts/select_persona_training_data.py \
  --rewritten-data data/persona_alpaca_user_0_rewritten.json \
  --train-scores scores/user_0_divergence.json \
  --train-limit 20000 \
  --test-data data/persona_alpaca_user_0_test_candidates.json \
  --test-scores scores/user_0_test_divergence.json \
  --test-limit 500 \
  --exclude-eval-file data/eval/persona_alpaca_specific_user_0.json \
  --output data/persona_alpaca_user_0_top_20k_with_500_test_by_div.json
```

The selector stores `selection_scores` alongside `train` and `test`, so the score provenance is visible without changing the conversation format consumed by training/evaluation code.

## Provenance Notes

The public scripts consolidate data construction into CLI steps and keep score sidecars to avoid the provenance ambiguity that can happen when sorted data and score files are separated.

The generic question/answer source was preprocessed from data released with FSPO / Few-Shot Preference Optimization, including `sher222/persona-iterative-responses` on Hugging Face. This repository intentionally treats that dataset as provenance/reference only; the runnable scripts consume a local preprocessed generic instruction JSON/JSONL.
