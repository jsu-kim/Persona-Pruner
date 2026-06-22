from __future__ import annotations

import asyncio
import math
import os
import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


def _score_from_logprobs(top_logprobs) -> Optional[float]:
    total = 0.0
    weighted = 0.0
    for item in top_logprobs:
        token = item.token.strip()
        if not token.isdigit():
            continue
        value = int(token)
        if 0 <= value <= 100:
            prob = math.exp(float(item.logprob))
            total += prob
            weighted += value * prob
    if total < 0.25:
        return None
    return weighted / total


def _score_from_text(text: str) -> Optional[float]:
    match = re.search(r"\b(100|[1-9]?\d)\b", text)
    if not match:
        return None
    return float(match.group(1))


@dataclass
class OpenAIJudge:
    """Small OpenAI judge wrapper compatible with the Persona Vectors protocol."""

    model: str = "gpt-4o-mini"
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for judge evaluation.")
        self.client = AsyncOpenAI()

    async def score(self, prompt: str) -> Optional[float]:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20,
                    seed=0,
                )
                content = response.choices[0].logprobs.content
                if content:
                    score = _score_from_logprobs(content[0].top_logprobs)
                    if score is not None:
                        return score
                text = response.choices[0].message.content or ""
                return _score_from_text(text)
            except Exception:
                if attempt + 1 >= self.max_retries:
                    raise
                await asyncio.sleep(2**attempt)
        return None

