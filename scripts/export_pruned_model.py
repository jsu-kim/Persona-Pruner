from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persona_pruner.modeling import load_causal_lm, load_tokenizer
from persona_pruner.pruning import load_topk_indices, prune_llama_ffn


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Persona-Pruner FFN keep indices and export a HF model.")
    parser.add_argument("--base-model", required=True, help="Base HF model id or local path.")
    parser.add_argument("--topk-indices", required=True, help="Path to topk_indices.pt or JSON keep indices.")
    parser.add_argument("--output-dir", required=True, help="Output HF model directory.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    args = parser.parse_args()

    model = load_causal_lm(
        args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = load_tokenizer(args.base_model, trust_remote_code=args.trust_remote_code)
    topk_indices = load_topk_indices(args.topk_indices)
    target_k = prune_llama_ffn(model, topk_indices)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    with (out / "persona_pruner_export.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "topk_indices": args.topk_indices,
                "intermediate_size": target_k,
            },
            f,
            indent=2,
        )
    print(f"Saved pruned model to {out}")


if __name__ == "__main__":
    main()
