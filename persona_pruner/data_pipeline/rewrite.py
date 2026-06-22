from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from persona_pruner.data_pipeline.common import load_json_or_jsonl, write_jsonl


REWRITE_SYSTEM_PROMPT = """You are an expert roleplay actor. Your goal is to rewrite an AI assistant's response so that it sounds exactly like a specific character (Persona) speaking.

Guidelines:
1. **Total Immersion**: Do not strictly translate the sentences. Instead, absorb the meaning and express it as the Persona would naturally say it.
2. **Voice & Tone**: Use the specific vocabulary, sentence structure, catchphrases, and emotional tone described in the Persona.
3. **Factual Consistency**: Keep the core information/solution of the original answer, but frame it through the Persona's worldview.
4. **No Preachiness**: Do not act like a helpful AI assistant. Act like the character.
5. **Format**: Output ONLY the rewritten text. Do not include "Here is the rewrite" or quotation marks."""

PERSONA_RE = re.compile(r"### Persona Description:\s*(.*?)\s*### User Question:", re.S)
QUESTION_RE = re.compile(
    r"### User Question:\s*(.*?)(?:\s*### Original Reference Answer:|\s*\*\*Task\*\*:)",
    re.S,
)


def load_personas(path: str | Path, limit: int | None = None) -> list[str]:
    personas = load_json_or_jsonl(path)
    if personas and isinstance(personas[0], dict):
        personas = [
            item.get("persona_description") or item.get("persona") or item.get("text")
            for item in personas
        ]
    personas = [str(item).strip() for item in personas if str(item).strip()]
    return personas[:limit] if limit else personas


def normalize_alpaca_question(example: dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if input_text:
        return f"{instruction}\n\n### Input:\n{input_text}"
    return instruction


def build_rewrite_messages(
    persona: str,
    question: str,
    reference_answer: str | None,
    *,
    condition_answer: bool = True,
) -> list[dict[str, str]]:
    if condition_answer:
        user_prompt = f"""### Persona Description:
{persona}

### User Question:
{question}

### Original Reference Answer:
{reference_answer}
**Task**: Rewrite the "Original Reference Answer" completely into the voice of the given Persona described above. Speak directly to the user as if you are having a conversation.
**Only output the rewritten text.**"""
    else:
        user_prompt = f"""### Persona Description:
{persona}

### User Question:
{question}
**Task**: Answer the user question directly in the voice of the given Persona described above. Speak naturally as that Persona and do not mention that you are an AI model.
**Only output the answer.**"""
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_persona_and_question(instruction: str) -> tuple[str, str]:
    persona_match = PERSONA_RE.search(instruction)
    question_match = QUESTION_RE.search(instruction)
    if not persona_match or not question_match:
        raise ValueError("Could not parse persona and question from rewrite instruction.")
    persona = " ".join(persona_match.group(1).split()).strip()
    question = question_match.group(1).strip()
    return persona, question


def chatml_from_rewrite_record(record: dict) -> list[dict[str, str]]:
    instruction = record.get("instruction") or ""
    output = record.get("output") or record.get("response") or record.get("text") or ""
    persona, question = extract_persona_and_question(instruction)
    system = f"You are roleplaying as a user with this persona: {persona}\nAnswer the question, roleplay as the given persona."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": str(output).strip()},
    ]


def build_rewrite_prompts_main() -> None:
    parser = argparse.ArgumentParser(description="Build persona rewrite prompt JSONL files.")
    parser.add_argument("--alpaca-data", required=True, help="Alpaca-style JSON/JSONL with instruction/input/output.")
    parser.add_argument("--personas", default=None, help="Persona JSONL/list file.")
    parser.add_argument("--persona-text", default=None, help="Single custom persona description.")
    parser.add_argument("--persona-file", default=None, help="Text file containing one custom persona description.")
    parser.add_argument("--output", required=True, help="Output JSONL; each line is a chat message list.")
    parser.add_argument("--persona-index", type=int, default=None, help="Only build prompts for this persona.")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--condition-answer", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    examples = load_json_or_jsonl(args.alpaca_data)
    if isinstance(examples, dict) and "train" in examples:
        examples = examples["train"]
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    personas = []
    if args.persona_text:
        personas.append(args.persona_text.strip())
    if args.persona_file:
        personas.append(Path(args.persona_file).read_text(encoding="utf-8").strip())
    if args.personas:
        personas.extend(load_personas(args.personas))
    if not personas:
        raise ValueError("Provide --personas, --persona-text, or --persona-file.")
    if args.persona_index is not None:
        personas = [personas[args.persona_index]]

    prompts = []
    for persona in personas:
        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("Alpaca examples must be dictionaries.")
            question = normalize_alpaca_question(example)
            answer = str(example.get("output", "")).strip()
            prompts.append(
                build_rewrite_messages(
                    persona,
                    question,
                    answer,
                    condition_answer=args.condition_answer,
                )
            )
    write_jsonl(prompts, args.output)
    print(f"Wrote {len(prompts)} rewrite prompts to {args.output}.")


def convert_rewrites_main() -> None:
    parser = argparse.ArgumentParser(description="Convert generated persona rewrites to ChatML training data.")
    parser.add_argument("--rewrites", required=True, help="Generator output JSON/JSONL with instruction and output fields.")
    parser.add_argument("--output", required=True, help="Output JSON with a train split.")
    args = parser.parse_args()

    records = load_json_or_jsonl(args.rewrites)
    if isinstance(records, dict) and "train" in records:
        records = records["train"]
    conversations = [chatml_from_rewrite_record(record) for record in records]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"train": conversations}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(conversations)} ChatML conversations to {out}.")


if __name__ == "__main__":
    build_rewrite_prompts_main()
