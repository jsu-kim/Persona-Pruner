from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def normalize_question(text: str) -> str:
    return " ".join(str(text).split()).strip()


def load_json_or_jsonl(path: str | Path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_questions_from_eval_file(path: str | Path) -> list[str]:
    payload = load_json_or_jsonl(path)
    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return [str(question) for question in payload["questions"]]
    if isinstance(payload, list):
        questions = []
        for row in payload:
            if isinstance(row, str):
                questions.append(row)
            elif isinstance(row, dict):
                for key in ("question", "prompt", "user"):
                    if key in row:
                        questions.append(str(row[key]))
                        break
        return questions
    raise ValueError(f"Could not extract questions from {path}.")


def load_excluded_questions(paths: list[str] | None = None) -> set[str]:
    excluded: set[str] = set()
    for path in paths or []:
        if not path:
            continue
        for question in extract_questions_from_eval_file(path):
            excluded.add(normalize_question(question))
    return excluded

