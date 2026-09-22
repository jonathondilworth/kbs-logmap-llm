"""
logmap_llm.oracle.rag.tokenizer

Token counting behind a Protocol so the retriever can enforce a token budget without
pulling in a model tokenizer during CPU tests. Default is a deterministic heuristic;
production may inject a real HF tokenizer via HFTokenCounter.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import math


@runtime_checkable
class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """~1 token per `chars_per_token` characters (min 1 for non-empty). Deterministic."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = float(chars_per_token)

    def count(self, text: str) -> int:
        s = str(text)
        if not s:
            return 0
        return max(1, math.ceil(len(s) / self.chars_per_token))


class HFTokenCounter:
    """Wraps a HuggingFace tokenizer. Lazy import; not used by CPU unit tests."""

    def __init__(self, model_name_or_path: str, revision: str | None = None):
        from transformers import AutoTokenizer  # pragma: no cover
        self._tok = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)

    def count(self, text: str) -> int:  # pragma: no cover
        return len(self._tok.encode(str(text), add_special_tokens=False))
