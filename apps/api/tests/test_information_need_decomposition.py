from __future__ import annotations

import pytest

from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.planning import (
    HeuristicInformationNeedDecomposer,
    InformationNeedDecompositionError,
    LLMInformationNeedDecomposer,
    parse_information_need_decomposition,
)


def _classification() -> QueryClassification:
    return QueryClassification(
        query_type=QueryType.BROAD_EXPLANATION,
        confidence=0.9,
        needs_metadata_filters=False,
        rationale="Test classification.",
        classifier_name="test",
    )


def test_llm_decomposition_parser_preserves_atomic_answer_requirements() -> None:
    result = parse_information_need_decomposition(
        """{
          "information_needs": [
            {
              "description": "Identify the available pipeline flows.",
              "retrieval_query": "available pipeline flows"
            },
            {
              "description": "Explain how each pipeline flow functions.",
              "retrieval_query": "pipeline flow execution functionality inputs outputs"
            }
          ],
          "rationale": "The question asks for both identification and behavior."
        }""",
    )

    assert [need.need_id for need in result.information_needs] == ["need_1", "need_2"]
    assert result.information_needs[1].description == "Explain how each pipeline flow functions."
    assert result.information_needs[1].retrieval_query == "pipeline flow execution functionality inputs outputs"
    assert result.fallback_used is False


def test_decomposition_parser_rejects_duplicate_retrieval_queries() -> None:
    with pytest.raises(InformationNeedDecompositionError, match="duplicate retrieval queries"):
        parse_information_need_decomposition(
            """{
              "information_needs": [
                {"description": "Identify flows.", "retrieval_query": "pipeline flows"},
                {"description": "Explain flows.", "retrieval_query": "pipeline flows"}
              ],
              "rationale": "Two requirements."
            }""",
        )


async def test_heuristic_decomposer_splits_explicit_compound_question() -> None:
    result = await HeuristicInformationNeedDecomposer().decompose(
        "What are the pipeline flows and how do they function?",
        _classification(),
    )

    assert len(result.information_needs) == 2
    assert result.information_needs[0].retrieval_query == "What are the pipeline flows"
    assert result.information_needs[1].retrieval_query == "how do they function"


class InvalidDecompositionLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "not-json"


async def test_llm_decomposer_falls_back_to_deterministic_split() -> None:
    decomposer = LLMInformationNeedDecomposer(
        llm_provider=InvalidDecompositionLLM(),
        fail_open=True,
    )

    result = await decomposer.decompose(
        "What are the pipeline flows and how do they function?",
        _classification(),
    )

    assert result.fallback_used is True
    assert result.decomposer_name == "heuristic_information_need_decomposer"
    assert len(result.information_needs) == 2
