from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .eval import build_parser as build_eval_parser
from .eval import evaluate_trait


def parse_user_ids(value: str) -> list[int]:
    ids: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(chunk))
    return ids


def build_parser() -> argparse.ArgumentParser:
    base = build_eval_parser()
    parser = argparse.ArgumentParser(description="Batch persona evaluation for user-indexed checkpoints.")
    parser.add_argument("--model-template", required=True, help="Model path template containing {user_id}.")
    parser.add_argument("--trait-prefix", required=True, help="Example: persona_alpaca_specific.")
    parser.add_argument("--user-ids", default="0-9", help="Comma/range list, e.g. 0-9 or 0,1,3.")
    parser.add_argument("--data-dir", default="data/eval")
    parser.add_argument("--output-dir", required=True)

    for action in base._actions:
        if action.dest in {"help", "model", "trait", "output_path", "trait_name"}:
            continue
        parser._add_action(action)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for user_id in parse_user_ids(args.user_ids):
        trait_name = f"{args.trait_prefix}_user_{user_id}"
        trait_path = Path(args.data_dir) / f"{trait_name}.json"
        model_path = args.model_template.format(user_id=user_id)
        output_path = output_dir / f"{trait_name}.csv"

        eval_args = argparse.Namespace(**vars(args))
        eval_args.model = model_path
        eval_args.trait = str(trait_path)
        eval_args.output_path = str(output_path)
        eval_args.trait_name = trait_name
        summaries.append(evaluate_trait(eval_args))

    persona_scores = [x["persona"] for x in summaries if x.get("persona") is not None]
    coherence_scores = [x["coherence"] for x in summaries if x.get("coherence") is not None]
    aggregate = {
        "n_persona_scores": len(persona_scores),
        "persona_mean": sum(persona_scores) / len(persona_scores) if persona_scores else None,
        "n_coherence_scores": len(coherence_scores),
        "coherence_mean": sum(coherence_scores) / len(coherence_scores) if coherence_scores else None,
        "per_user": {x["trait"]: {"persona": x["persona"], "coherence": x["coherence"]} for x in summaries},
    }
    with (output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

