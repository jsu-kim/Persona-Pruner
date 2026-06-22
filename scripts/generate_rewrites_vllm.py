#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_jsonl(path: str | Path) -> list:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def build_instruction(messages: list[dict]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages) + "\n"


def infer_tensor_parallel_size(gpu_ids: str | None, explicit: int | None) -> int:
    if explicit:
        return explicit
    visible = gpu_ids or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        return len([chunk for chunk in visible.split(",") if chunk.strip()])
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate persona rewrites with vLLM.")
    parser.add_argument("--model-name", "--model", dest="model_name", required=True)
    parser.add_argument("--input-file", "--prompts", dest="input_file", required=True)
    parser.add_argument("--output-file", "--output", dest="output_file", required=True)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPUs, e.g. 0,2. Sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=99999999)
    parser.add_argument("--max-input-len", type=int, default=2048)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tensor_parallel_size = infer_tensor_parallel_size(args.gpu_ids, args.tensor_parallel_size)
    generator = args.generator or args.model_name
    prompts = load_jsonl(args.input_file)[args.start_idx : args.end_idx]

    logging.info(
        "Starting vLLM rewrite generation model=%s prompts=%s output=%s cuda_visible_devices=%s tensor_parallel_size=%s",
        args.model_name,
        len(prompts),
        args.output_file,
        os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
        tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )

    results: list[dict] = []
    output_path = Path(args.output_file)
    error_path = output_path.with_suffix(".errors.jsonl")
    progress_path = output_path.with_suffix(".progress.json")
    if error_path.exists():
        error_path.unlink()

    for start in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_messages = prompts[start : start + args.batch_size]
        chat_texts = []
        token_lengths = []
        truncated_indices = []
        for local_idx, messages in enumerate(batch_messages):
            try:
                if "Qwen3" in args.model_name:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                else:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = "".join(f"{m.get('role', '')}: {m.get('content', '')}\n" for m in messages) + "assistant:"
            tokens = tokenizer.encode(text)
            token_lengths.append(len(tokens))
            if len(tokens) > args.max_input_len:
                text = tokenizer.decode(tokens[: args.max_input_len], skip_special_tokens=True)
                truncated_indices.append(local_idx)
            chat_texts.append(text)

        if "Qwen3" in args.model_name and args.enable_thinking:
            sampling = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                seed=args.seed,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            )
        elif "Qwen3" in args.model_name:
            sampling = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                seed=args.seed,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            )
        else:
            sampling = SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature if args.temperature > 0 else 0.01,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            )

        batch_start = time.time()
        try:
            outputs = llm.generate(chat_texts, sampling)
            for messages, output in zip(batch_messages, outputs):
                text = output.outputs[0].text.strip() if output.outputs else "[Empty Output]"
                results.append(
                    {
                        "instruction": build_instruction(messages),
                        "output": text,
                        "generator": generator,
                    }
                )
        except Exception as exc:
            with error_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "start": start,
                            "batch_size": len(batch_messages),
                            "error": f"{type(exc).__name__}: {exc}",
                            "token_lengths": token_lengths,
                            "truncated_indices": truncated_indices,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            raise

        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed_samples": len(results),
                    "total_samples": len(prompts),
                    "last_batch_seconds": round(time.time() - batch_start, 3),
                    "last_token_lengths": token_lengths,
                    "last_truncated_indices": truncated_indices,
                },
                f,
                indent=2,
            )
        if ((start // args.batch_size) + 1) % args.save_every == 0:
            save_json(output_path, results)

    save_json(output_path, results)
    logging.info("Wrote %s rewrite outputs to %s", len(results), output_path)


if __name__ == "__main__":
    main()

