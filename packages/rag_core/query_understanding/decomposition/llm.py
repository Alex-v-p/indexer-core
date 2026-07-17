from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.ports import LLMProvider
from packages.rag_core.query_understanding.decomposition.base import InformationNeedDecomposer
from packages.rag_core.query_understanding.decomposition.heuristic import HeuristicInformationNeedDecomposer
from packages.rag_core.query_understanding.decomposition.models import (
    InformationNeed,
    InformationNeedDecomposition,
)

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "decompose_information_needs.md"
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class InformationNeedDecompositionError(RuntimeError):
    """Raised when a question cannot be decomposed into structured needs."""


class LLMInformationNeedDecomposer:
    """LLM-backed requirement decomposition with deterministic fail-open behavior."""

    name = "llm_information_need_decomposer"

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        fallback_decomposer: InformationNeedDecomposer | None = None,
        fail_open: bool = True,
        max_information_needs: int = 6,
        max_need_chars: int = 240,
        max_rationale_chars: int = 500,
    ) -> None:
        if max_information_needs <= 0:
            raise ValueError("max_information_needs must be positive.")
        if max_need_chars <= 0:
            raise ValueError("max_need_chars must be positive.")
        if max_rationale_chars <= 0:
            raise ValueError("max_rationale_chars must be positive.")
        self._llm_provider = llm_provider
        self._fallback_decomposer = fallback_decomposer or HeuristicInformationNeedDecomposer(
            max_information_needs=max_information_needs,
        )
        self._fail_open = fail_open
        self._max_information_needs = max_information_needs
        self._max_need_chars = max_need_chars
        self._max_rationale_chars = max_rationale_chars

    async def decompose(self, question: str) -> InformationNeedDecomposition:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        try:
            raw_response = await self._llm_provider.generate(
                build_information_need_prompt(
                    normalized,
                    max_information_needs=self._max_information_needs,
                ),
            )
            return parse_information_need_decomposition(
                raw_response,
                decomposer_name=self.name,
                max_information_needs=self._max_information_needs,
                max_need_chars=self._max_need_chars,
                max_rationale_chars=self._max_rationale_chars,
            )
        except Exception as exc:
            if not self._fail_open:
                if isinstance(exc, InformationNeedDecompositionError):
                    raise
                raise InformationNeedDecompositionError("Information-need decomposition failed.") from exc

            fallback = await self._fallback_decomposer.decompose(normalized)
            return InformationNeedDecomposition(
                information_needs=fallback.information_needs,
                rationale=fallback.rationale,
                decomposer_name=fallback.decomposer_name,
                fallback_used=True,
            )


def build_information_need_prompt(
    question: str,
    *,
    max_information_needs: int = 6,
) -> str:
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("question must not be empty.")
    if max_information_needs <= 0:
        raise ValueError("max_information_needs must be positive.")

    return (
        _load_prompt_template()
        .replace("{{ question }}", normalized)
        .replace("{{ max_information_needs }}", str(max_information_needs))
        .strip()
    )


def parse_information_need_decomposition(
    raw_response: str,
    *,
    decomposer_name: str = LLMInformationNeedDecomposer.name,
    max_information_needs: int = 6,
    max_need_chars: int = 240,
    max_rationale_chars: int = 500,
) -> InformationNeedDecomposition:
    if max_information_needs <= 0:
        raise ValueError("max_information_needs must be positive.")
    if max_need_chars <= 0 or max_rationale_chars <= 0:
        raise ValueError("character limits must be positive.")

    payload = _extract_json_object(raw_response)
    raw_needs = payload.get("information_needs")
    if not isinstance(raw_needs, list) or not raw_needs:
        raise InformationNeedDecompositionError("information_needs must be a non-empty JSON array.")
    if len(raw_needs) > max_information_needs:
        raise InformationNeedDecompositionError(
            f"information_needs may contain at most {max_information_needs} entries.",
        )

    needs: list[InformationNeed] = []
    seen_queries: set[str] = set()
    for index, raw_need in enumerate(raw_needs, start=1):
        if not isinstance(raw_need, dict):
            raise InformationNeedDecompositionError("Each information need must be a JSON object.")
        description = _normalize_required_text(
            raw_need.get("description"),
            field_name="description",
            max_chars=max_need_chars,
        )
        retrieval_query = _normalize_required_text(
            raw_need.get("retrieval_query"),
            field_name="retrieval_query",
            max_chars=max_need_chars,
        )
        query_key = retrieval_query.casefold()
        if query_key in seen_queries:
            raise InformationNeedDecompositionError("information_needs must not contain duplicate retrieval queries.")
        seen_queries.add(query_key)
        needs.append(
            InformationNeed(
                need_id=f"need_{index}",
                description=description,
                retrieval_query=retrieval_query,
            ),
        )

    rationale = _normalize_required_text(
        payload.get("rationale"),
        field_name="rationale",
        max_chars=max_rationale_chars,
    )
    return InformationNeedDecomposition(
        information_needs=tuple(needs),
        rationale=rationale,
        decomposer_name=decomposer_name,
    )


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip())
    start = cleaned.find("{")
    if start < 0:
        raise InformationNeedDecompositionError("Decomposition response did not contain a JSON object.")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise InformationNeedDecompositionError("Decomposition response contained invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise InformationNeedDecompositionError("Decomposition response must be a JSON object.")
    return parsed


def _normalize_required_text(value: object, *, field_name: str, max_chars: int) -> str:
    normalized = " ".join(str(value or "").strip().split())[:max_chars].strip()
    if not normalized:
        raise InformationNeedDecompositionError(f"{field_name} must not be empty.")
    return normalized
