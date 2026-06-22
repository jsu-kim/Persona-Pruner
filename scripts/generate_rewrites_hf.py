#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_jsonl(path: str | Path) -> list:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate persona rewrites from prompt JSONL with a HF chat model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, help="JSONL whose lines are chat message lists.")
    parser.add_argument("--output", required=True, help="JSON list with instruction/output records.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    prompts = load_jsonl(args.prompts)
    records = []
    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_messages = prompts[start : start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch_messages
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        do_sample = args.temperature > 0
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = inputs["input_ids"].shape[1]
        for messages, output_ids in zip(batch_messages, outputs):
            generated_ids = output_ids[prompt_width:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            instruction = "\n".join(message["content"] for message in messages)
            records.append({"instruction": instruction, "output": answer})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} generated rewrites to {out}.")


if __name__ == "__main__":
    main()
