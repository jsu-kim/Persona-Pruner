# Persona-Pruner: Sculpting Lightweight Models for Role-Playing (ICML 2026)

Pruning framework for sculpting lightweight role-playing LLMs from a single persona description.

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/arXiv-2606.14695-b31b1b)](https://arxiv.org/abs/2606.14695)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)

Jinsu Kim<sup>1</sup>, Jihoon Tack<sup>2</sup>, Noah Lee<sup>2</sup>, Jongheon Jeong<sup>1</sup>

<sup>1</sup>Department of Artificial Intelligence, Korea University  
<sup>2</sup>Korea Advanced Institute of Science and Technology (KAIST)

---

## What is Persona-Pruner?

Persona-Pruner builds compact, persona-specific LLMs by identifying and removing FFN intermediate channels while preserving target persona behavior measured by role-playing evaluation.

Given only a natural-language persona description, Persona-Pruner constructs persona-aligned calibration data, discovers persona-specific sub-networks, and exports structurally pruned Hugging Face checkpoints for efficient role-playing deployment.

## What's Included

| Component | Status | Notes |
| --- | --- | --- |
| Data construction | Included | Rewrite generic instruction data into a target persona voice, score persona divergence, and select high-divergence examples. |
| FFN pruning export | Included | Apply per-layer keep-index tensors to LLaMA/Qwen-style decoder models and save a smaller HF checkpoint. |
| Evaluation | Included | Persona-specific/general evaluation files and an OpenAI-judge runner adapted from Persona Vectors. |
| Large artifacts | Not included | Full checkpoints, learned keep-index tensors, and private experiment outputs should be hosted separately. |

## Installation

```bash
git clone https://github.com/jsu-kim/Persona-Pruner.git
cd Persona-Pruner

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For fast rewrite generation with large instruction models:

```bash
pip install -e ".[rewrite]"
```

Set `HF_TOKEN` if you use gated Hugging Face models. Set `OPENAI_API_KEY` if you run judge-based evaluation.

## Quick Sanity Check

The custom-persona pipeline can be smoke-tested with the bundled tiny example data. This does not reproduce paper numbers; it only checks that rewriting, formatting, divergence scoring, and selection run end-to-end.

```bash
PERSONA_TEXT="Age: 44, sex: Male, location: North America, occupation: financial analyst, lifestyle: disciplined and community-oriented, values long-term planning, practical risk management, financial literacy, health-conscious routines, and meaningful travel, communication style: concise, analytical, warm, and advice-oriented." \
GENERIC_DATA=data/generic_instruction_data.example.json \
REWRITE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
SCORER_MODEL=meta-llama/Llama-3.2-3B-Instruct \
REWRITE_GPU_IDS=0 \
GPU_ID=0 \
CONDITION_ANSWER=0 \
MAX_EXAMPLES=2 \
TRAIN_LIMIT=2 \
bash scripts/run_custom_persona_pipeline.sh
```

Outputs are written under `runs/custom_persona/`, which is ignored by git.

## Reproducing the Paper-Style Pipeline

The paper-style Alpaca-specific pipeline has three stages:

```bash
GENERIC_DATA=/path/to/preprocessed_generic_instruction_data.json \
REWRITE_MODEL=meta-llama/Llama-3.1-70B-Instruct \
REWRITE_GPU_IDS=0,2 \
bash scripts/run_data_pipeline_10_personas.sh

TOPK_TEMPLATE=/path/to/topk_indices/user_{user_id}_6144.pt \
bash scripts/run_pruning_10_personas.sh

bash scripts/run_eval_10_personas.sh
```

The default artifact root is `runs/persona_pruner_10p/`.

`run_data_pipeline_10_personas.sh` performs:

1. rewrite prompt construction for the 10 reference personas;
2. persona answer rewriting with vLLM by default;
3. conversion to ChatML training data;
4. hidden-state divergence scoring;
5. selection of high-divergence training examples.

For the Alpaca-specific setting, each user's 20 `persona_alpaca_specific` evaluation questions are excluded before scoring and selection to preserve the train/test split.

For 70B-class rewrite models, keep `PARALLEL_JOBS=1` unless each concurrent generation process has a separate GPU group. Set `REWRITE_GPU_IDS=0,2` or similar to choose rewrite GPUs. Set `GENERATOR_BACKEND=hf` to use the simpler Hugging Face generator instead of vLLM.

## Data Construction

The input generic instruction file should be JSON or JSONL with Alpaca-style fields:

```json
{
  "instruction": "Explain why stocks are a good form of investment.",
  "input": "",
  "output": "Stocks can offer long-term growth..."
}
```

The generic instruction source used in the paper was preprocessed from data released with FSPO / Few-Shot Preference Optimization, including [`sher222/persona-iterative-responses`](https://huggingface.co/datasets/sher222/persona-iterative-responses). This repository treats that dataset as provenance only and does not redistribute the full processed training data.

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for the full data construction flow and lower-level commands.

## Pruning

If you already have Persona-Pruner keep indices, export a structurally pruned model:

```bash
python scripts/export_pruned_model.py \
  --base-model meta-llama/Llama-3.2-3B-Instruct \
  --topk-indices /path/to/topk_indices.pt \
  --output-dir exports/alpaca_user_0_6144 \
  --dtype bf16
```

For Llama-3.2-3B-Instruct, 25% FFN pruning corresponds to a target intermediate size of `6144`, so the keep-index tensor should have shape `[28, 6144]`.

## Evaluation

Evaluate one persona:

```bash
python -m persona_pruner.eval \
  --model exports/alpaca_user_0_6144 \
  --trait data/eval/persona_alpaca_specific_user_0.json \
  --output-path results/user_0.csv \
  --deterministic \
  --seed 42 \
  --max-tokens 2048 \
  --judge-model gpt-4o-mini
```

Evaluate all 10 user-indexed checkpoints:

```bash
MODEL_TEMPLATE="/path/to/pruned_models/user_{user_id}_6144" \
OUT_DIR="results/alpaca_specific_3b_6144_deterministic" \
bash scripts/eval_alpaca_specific_example.sh
```

The batch runner writes per-user CSV files plus `summary.csv` and `aggregate_summary.json`.

## Custom Personas

Persona-Pruner can also be run with a new persona description:

```bash
PERSONA_TEXT="A 44-year-old financial analyst who values disciplined long-term investing..." \
GENERIC_DATA=/path/to/preprocessed_generic_instruction_data.json \
REWRITE_GPU_IDS=0,2 \
CONDITION_ANSWER=0 \
bash scripts/run_custom_persona_pipeline.sh
```

By default, rewrite prompts condition on the generic answer (`CONDITION_ANSWER=1`). Set `CONDITION_ANSWER=0` to ask the generator to answer directly in the persona voice without the reference answer.

Custom divergence scoring compares the target persona against `data/personas/alpaca_reference_personas_10.jsonl`. Filtering tends to work better when custom persona descriptions use a similar level of detail and structure to those reference personas.

If you already have keep indices for the custom persona, add:

```bash
TOPK_INDICES=/path/to/topk_indices.pt
```

to export a pruned model in the same run.

## Repository Layout

```text
persona_pruner/
  pruning.py                 # structural FFN pruning
  eval.py                    # single-persona evaluation
  batch_eval.py              # 10-persona batch evaluation
  data_pipeline/             # rewrite, divergence scoring, selection
scripts/
  run_data_pipeline_10_personas.sh
  run_pruning_10_personas.sh
  run_eval_10_personas.sh
  run_custom_persona_pipeline.sh
  export_pruned_model.py
data/
  personas/                  # 10 reference persona descriptions
  eval/                      # persona-specific/general evaluation files
docs/
  DATA_PIPELINE.md
```

## Acknowledgements

The evaluation pipeline and data format are adapted from [safety-research/persona_vectors](https://github.com/safety-research/persona_vectors).

The generic instruction data provenance follows FSPO / Few-Shot Preference Optimization, including [Asap7772/fewshot-preference-optimization](https://github.com/Asap7772/fewshot-preference-optimization) and [`sher222/persona-iterative-responses`](https://huggingface.co/datasets/sher222/persona-iterative-responses).

The vLLM rewrite runner is a cleaned, self-contained version of the internal `akinator_may/inference_alpaca.py` workflow used for persona answer rewriting.

## Citation

```bibtex
@inproceedings{kim2026personapruner,
  title = {Persona-Pruner: Sculpting Lightweight Models for Role-Playing},
  author = {Kim, Jinsu and Tack, Jihoon and Lee, Noah and Jeong, Jongheon},
  booktitle = {Forty-third International Conference on Machine Learning},
  year = {2026},
}
```
