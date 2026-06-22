from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from persona_pruner.data_pipeline.common import load_excluded_questions, normalize_question


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_personas(path: str | Path, total_personas: int | None = None) -> list[str]:
    personas: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            personas.append(json.loads(line))
            if total_personas is not None and len(personas) >= total_personas:
                break
    return personas


def build_system_prompt(persona: str) -> str:
    return f"You are roleplaying as a user with this persona: {persona}\nAnswer the question, roleplay as the given persona."


def load_questions(path: str | Path, split: str, excluded_questions: set[str] | None = None) -> list[tuple[int, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload[split]
    questions = []
    excluded_questions = excluded_questions or set()
    for idx, row in enumerate(rows):
        if len(row) < 2 or row[1].get("role") != "user":
            raise ValueError("Expected each conversation to start with system/user messages.")
        question = row[1]["content"]
        if normalize_question(question) in excluded_questions:
            continue
        questions.append((idx, question))
    return questions


class HiddenStateDivergence:
    def __init__(self, model_name_or_path: str, dtype: str = "float16", device: str = "cuda"):
        self.device = torch.device(device)
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.autocast_dtype = torch_dtype if self.device.type == "cuda" else None

    def format_batch(self, system_prompts: list[str], questions: list[str]) -> list[str]:
        texts = []
        for system_prompt, question in zip(system_prompts, questions):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            texts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return texts

    def hidden_states(self, texts: list[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        context = torch.amp.autocast("cuda", dtype=self.autocast_dtype) if self.device.type == "cuda" else torch.no_grad()
        with torch.no_grad(), context:
            return self.model(**inputs, output_hidden_states=True, use_cache=False).hidden_states

    @staticmethod
    def last_token(hidden_states, layer_idx: int):
        return hidden_states[layer_idx][:, -1, :]

    def cosine_for_layers(self, target_states, compare_states, layer_start: int, layer_end: int) -> torch.Tensor:
        layer_end = min(layer_end, len(target_states))
        sims = []
        for layer_idx in range(layer_start, layer_end):
            sims.append(
                F.cosine_similarity(
                    self.last_token(target_states, layer_idx),
                    self.last_token(compare_states, layer_idx),
                    dim=1,
                )
            )
        return torch.stack(sims, dim=0).mean(dim=0)

    def score_batch(
        self,
        questions: list[str],
        target_persona: str,
        compare_personas: list[str],
        layer_start: int,
        layer_end: int,
    ) -> list[float]:
        target_prompt = build_system_prompt(target_persona)
        target_texts = self.format_batch([target_prompt] * len(questions), questions)
        target_states = self.hidden_states(target_texts)
        compare_sims = []
        for compare_persona in compare_personas:
            compare_prompt = build_system_prompt(compare_persona)
            compare_texts = self.format_batch([compare_prompt] * len(questions), questions)
            compare_states = self.hidden_states(compare_texts)
            compare_sims.append(self.cosine_for_layers(target_states, compare_states, layer_start, layer_end))
            del compare_states
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        avg_sim = torch.stack(compare_sims, dim=0).mean(dim=0)
        del target_states, compare_sims
        return (1.0 - avg_sim).detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score persona-specific questions by hidden-state divergence.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="ChatML JSON containing the requested split.")
    parser.add_argument("--personas", required=True, help="Persona JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-persona-idx", type=int, default=0)
    parser.add_argument("--n-compare-personas", type=int, default=9)
    parser.add_argument("--total-personas", type=int, default=10)
    parser.add_argument("--layer-start", type=int, default=5)
    parser.add_argument("--layer-end", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-before-limit", action="store_true")
    parser.add_argument(
        "--exclude-eval-file",
        action="append",
        default=[],
        help="Eval JSON/list whose questions should be excluded before scoring. Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    personas = load_personas(args.personas, args.total_personas)
    target_persona = personas[args.target_persona_idx]
    compare_personas = [
        persona for idx, persona in enumerate(personas) if idx != args.target_persona_idx
    ][: args.n_compare_personas]
    excluded_questions = load_excluded_questions(args.exclude_eval_file)
    questions = load_questions(args.data, args.split, excluded_questions)
    if excluded_questions:
        print(f"Excluded {len(excluded_questions)} eval questions; scoring {len(questions)} remaining examples.")
    indexed_questions = list(questions)
    if args.shuffle_before_limit:
        random.shuffle(indexed_questions)
    if args.limit is not None:
        indexed_questions = indexed_questions[: args.limit]

    scorer = HiddenStateDivergence(args.model, dtype=args.dtype, device=args.device)
    results = []
    for start in tqdm(range(0, len(indexed_questions), args.batch_size), desc="Scoring"):
        batch = indexed_questions[start : start + args.batch_size]
        indices = [idx for idx, _ in batch]
        batch_questions = [question for _, question in batch]
        divergences = scorer.score_batch(
            batch_questions,
            target_persona,
            compare_personas,
            args.layer_start,
            args.layer_end,
        )
        for idx, question, divergence in zip(indices, batch_questions, divergences):
            results.append(
                {
                    "index": idx,
                    "question": question,
                    "target_persona": target_persona,
                    "divergence_score": float(divergence),
                    "similarity_score": float(1.0 - divergence),
                }
            )
        gc.collect()
    results.sort(key=lambda item: item["divergence_score"], reverse=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} divergence scores to {out}.")


if __name__ == "__main__":
    main()
