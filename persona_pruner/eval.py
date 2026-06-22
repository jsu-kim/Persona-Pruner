from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
from tqdm.auto import tqdm

from .judge import OpenAIJudge
from .modeling import load_causal_lm, load_tokenizer
from .prompts import COHERENCE_0_100


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trait(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_conversations(trait_data: dict, *, use_system_prompt: bool = True) -> list[list[dict]]:
    system_prompt = trait_data.get("system_prompt") if use_system_prompt else None
    conversations = []
    for question in trait_data["questions"]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        conversations.append(messages)
    return conversations


def render_prompts(tokenizer, conversations: list[list[dict]]) -> list[str]:
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def generate_hf(
    model,
    tokenizer,
    conversations: list[list[dict]],
    *,
    batch_size: int = 8,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 1.0,
    deterministic: bool = False,
) -> tuple[list[str], list[str]]:
    prompts = render_prompts(tokenizer, conversations)
    answers: list[str] = []
    model_device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start : start + batch_size]
        batch = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model_device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": not deterministic,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if not deterministic:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            output = model.generate(**batch, **gen_kwargs)
        prompt_len = batch["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
        answers.extend(decoded)
    return prompts, answers


async def score_rows(
    rows: list[dict],
    *,
    trait_name: str,
    trait_prompt_template: str,
    judge_model: str,
    max_concurrent_judges: int,
    include_coherence: bool,
) -> None:
    judge = OpenAIJudge(judge_model)
    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_one(row: dict, metric: str, prompt: str):
        async with semaphore:
            score = await judge.score(prompt)
            row[metric] = score

    tasks = []
    for row in rows:
        tasks.append(
            run_one(
                row,
                trait_name,
                trait_prompt_template.format(question=row["question"], answer=row["answer"]),
            )
        )
        if include_coherence:
            tasks.append(
                run_one(
                    row,
                    "coherence",
                    COHERENCE_0_100.format(question=row["question"], answer=row["answer"]),
                )
            )

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
        await future


def add_summary_columns(df: pd.DataFrame, trait_name: str) -> pd.DataFrame:
    if trait_name in df:
        df[f"{trait_name}_mean"] = df[trait_name].mean()
        df[f"{trait_name}_std"] = df[trait_name].std()
    if "coherence" in df:
        df["coherence_mean"] = df["coherence"].mean()
        df["coherence_std"] = df["coherence"].std()
    return df


def evaluate_trait(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    trait_path = Path(args.trait)
    trait_name = args.trait_name or trait_path.stem
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        df = pd.read_csv(output_path)
        summary = {
            "trait": trait_name,
            "path": str(output_path),
            "persona": float(df[trait_name].mean()) if trait_name in df else None,
            "coherence": float(df["coherence"].mean()) if "coherence" in df else None,
            "skipped_existing": True,
        }
        return summary

    trait_data = load_trait(trait_path)
    conversations = build_conversations(trait_data, use_system_prompt=args.use_system_prompt)

    tokenizer = load_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    model = load_causal_lm(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    prompts, answers = generate_hf(
        model,
        tokenizer,
        conversations,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        deterministic=args.deterministic,
    )

    rows = []
    for i, (question, prompt, answer) in enumerate(zip(trait_data["questions"], prompts, answers)):
        rows.append(
            {
                "question": question,
                "prompt": prompt,
                "answer": answer,
                "question_id": f"{trait_name}_{i}",
            }
        )

    asyncio.run(
        score_rows(
            rows,
            trait_name=trait_name,
            trait_prompt_template=trait_data["eval_prompt"],
            judge_model=args.judge_model,
            max_concurrent_judges=args.max_concurrent_judges,
            include_coherence=args.include_coherence,
        )
    )

    df = add_summary_columns(pd.DataFrame(rows), trait_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    summary = {
        "trait": trait_name,
        "path": str(output_path),
        "persona": float(df[trait_name].mean()),
        "coherence": float(df["coherence"].mean()) if "coherence" in df else None,
        "n_questions": len(df),
        "skipped_existing": False,
    }
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a model on Persona-Pruner persona-roleplay data.")
    parser.add_argument("--model", required=True, help="HF model id or local pruned model path.")
    parser.add_argument("--trait", required=True, help="Path to a data/eval/*.json trait file.")
    parser.add_argument("--output-path", required=True, help="CSV output path.")
    parser.add_argument("--trait-name", default=None, help="Metric column name. Defaults to trait filename stem.")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-concurrent-judges", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default=None, help="Optional transformers device_map, e.g. auto.")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-system-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-coherence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    summary = evaluate_trait(build_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

