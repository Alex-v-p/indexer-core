from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from packages.rag_core.ports import LLMProvider

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "generate_query_variants.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


class QueryVariantGenerationError(RuntimeError):
    """Raised when query expansion cannot produce usable query variants."""


class QueryVariantGenerator(Protocol):
    """Generate alternative search queries for one user question."""

    async def generate(self, question: str, *, count: int) -> list[str]:
        """Return up to ``count`` distinct alternatives to the original query."""


class LLMQueryVariantGenerator:
    """Generate search-oriented query variants through a provider-neutral LLM."""

    def __init__(self, *, llm_provider: LLMProvider, max_variant_chars: int = 300) -> None:
        if max_variant_chars <= 0:
            raise ValueError("max_variant_chars must be positive.")
        self._llm_provider = llm_provider
        self._max_variant_chars = max_variant_chars

    async def generate(self, question: str, *, count: int) -> list[str]:
        normalized_question = " ".join(question.strip().split())
        if not normalized_question:
            raise ValueError("question must not be empty.")
        if count <= 0:
            raise ValueError("count must be positive.")

        raw_response = await self._llm_provider.generate(
            build_query_variant_prompt(normalized_question, count=count),
        )
        variants = parse_query_variants(
            raw_response,
            original_question=normalized_question,
            count=count,
            max_variant_chars=self._max_variant_chars,
        )
        if not variants:
            raise QueryVariantGenerationError("The query-variant model returned no usable alternatives.")
        return variants


def build_query_variant_prompt(question: str, *, count: int) -> str:
    """Build the deterministic query-expansion prompt from the markdown template."""

    if count <= 0:
        raise ValueError("count must be positive.")
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty.")

    return (
        _load_prompt_template()
        .replace("{{variant_count}}", str(count))
        .replace("{{question}}", normalized_question)
    )


def parse_query_variants(
    raw_response: str,
    *,
    original_question: str,
    count: int,
    max_variant_chars: int = 300,
) -> list[str]:
    """Parse JSON-first LLM output with a conservative line-based fallback."""

    if count <= 0:
        raise ValueError("count must be positive.")
    if max_variant_chars <= 0:
        raise ValueError("max_variant_chars must be positive.")

    candidates = _json_candidates(raw_response)
    if candidates is None:
        candidates = _line_candidates(raw_response)

    original_key = _query_key(original_question)
    variants: list[str] = []
    seen = {original_key}
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        normalized = _normalize_query(candidate, max_chars=max_variant_chars)
        key = _query_key(normalized)
        if not normalized or not key or key in seen:
            continue
        seen.add(key)
        variants.append(normalized)
        if len(variants) >= count:
            break
    return variants


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _json_candidates(raw_response: str) -> list[object] | None:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip())
    if not cleaned:
        return None

    for opening_character in ("{", "["):
        start = cleaned.find(opening_character)
        if start < 0:
            continue
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("queries", "variants", "query_variants"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
    return None


def _line_candidates(raw_response: str) -> list[str]:
    bulleted_candidates: list[str] = []
    plain_candidates: list[str] = []
    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        is_bulleted = _BULLET_PATTERN.match(line) is not None
        cleaned = _BULLET_PATTERN.sub("", line).strip().strip('"').strip("'")
        if not cleaned:
            continue
        if is_bulleted:
            bulleted_candidates.append(cleaned)
        else:
            plain_candidates.append(cleaned)
    return bulleted_candidates or plain_candidates


def _normalize_query(value: str, *, max_chars: int) -> str:
    normalized = " ".join(value.strip().split())
    return normalized[:max_chars].strip()


def _query_key(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = [
    "LLMQueryVariantGenerator",
    "QueryVariantGenerationError",
    "QueryVariantGenerator",
    "build_query_variant_prompt",
    "parse_query_variants",
]
