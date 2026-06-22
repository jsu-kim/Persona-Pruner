from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_pruner.data_pipeline.common import load_excluded_questions, normalize_question


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def question_of(conversation: list[dict]) -> str:
    for message in conversation:
        if message.get("role") == "user":
            return message.get("content", "").strip()
    raise ValueError("Conversation has no user message.")


def attach_scores(
    candidates: list[list[dict]],
    scores: list[dict],
    *,
    limit: int | None,
    excluded_questions: set[str] | None = None,
) -> tuple[list[list[dict]], list[dict], list[dict]]:
    excluded_questions = excluded_questions or set()
    by_question = {
        normalize_question(question_of(conversation)): conversation
        for conversation in candidates
        if normalize_question(question_of(conversation)) not in excluded_questions
    }
    selected = []
    selected_scores = []
    matched_questions = set()
    for score in scores:
        question = normalize_question(score["question"])
        if question in excluded_questions:
            continue
        conversation = by_question.get(question)
        if conversation is None:
            continue
        selected.append(conversation)
        selected_scores.append(score)
        matched_questions.add(question)
        if limit is not None and len(selected) >= limit:
            break
    unmatched = [
        conversation
        for conversation in candidates
        if normalize_question(question_of(conversation)) not in matched_questions
        and normalize_question(question_of(conversation)) not in excluded_questions
    ]
    return selected, selected_scores, unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Persona-Pruner training data with divergence scores.")
    parser.add_argument("--rewritten-data", required=True, help="ChatML JSON with train split.")
    parser.add_argument("--train-scores", required=True, help="Divergence score JSON for train candidates.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--test-data", default=None, help="Optional ChatML JSON containing test candidates.")
    parser.add_argument("--test-scores", default=None, help="Optional divergence score JSON for test candidates.")
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument(
        "--exclude-eval-file",
        action="append",
        default=[],
        help="Eval JSON/list whose questions should be excluded from selected train data. Repeatable.",
    )
    args = parser.parse_args()

    excluded_questions = load_excluded_questions(args.exclude_eval_file)
    rewritten = load_json(args.rewritten_data)
    train_candidates = rewritten["train"]
    train_scores = load_json(args.train_scores)
    train, train_score_rows, unmatched_train = attach_scores(
        train_candidates,
        train_scores,
        limit=args.train_limit,
        excluded_questions=excluded_questions,
    )

    output = {
        "train": train,
        "selection_scores": {
            "train": train_score_rows,
            "unmatched_train_count": len(unmatched_train),
        },
    }

    if args.test_data and args.test_scores:
        test_payload = load_json(args.test_data)
        test_candidates = test_payload.get("test", test_payload.get("train", []))
        test_scores = load_json(args.test_scores)
        test, test_score_rows, unmatched_test = attach_scores(
            test_candidates,
            test_scores,
            limit=args.test_limit,
            excluded_questions=set(),
        )
        output["test"] = test
        output["selection_scores"]["test"] = test_score_rows
        output["selection_scores"]["unmatched_test_count"] = len(unmatched_test)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(
        f"Wrote {len(output['train'])} train"
        f"{' and ' + str(len(output.get('test', []))) + ' test' if 'test' in output else ''} conversations to {out}."
    )


if __name__ == "__main__":
    main()
