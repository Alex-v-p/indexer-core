from __future__ import annotations

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.query_graph.nodes import DecomposeInformationNeedsNode
from packages.rag_core.query_understanding.decomposition import (
    HeuristicInformationNeedDecomposer,
    InformationNeedDecompositionError,
    LLMInformationNeedDecomposer,
    build_information_need_prompt,
    parse_information_need_decomposition,
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


async def test_heuristic_decomposer_splits_explicit_compound_question_without_classification() -> None:
    result = await HeuristicInformationNeedDecomposer().decompose(
        "What are the pipeline flows and how do they function?",
    )

    assert len(result.information_needs) == 2
    assert result.information_needs[0].retrieval_query == "What are the pipeline flows"
    assert result.information_needs[1].retrieval_query == "how do they function"


def test_decomposition_prompt_is_independent_from_query_classification() -> None:
    prompt = build_information_need_prompt(
        "What are the pipeline flows and how do they function?",
        max_information_needs=4,
    )

    assert "Query type:" not in prompt
    assert "What are the pipeline flows and how do they function?" in prompt
    assert "between 1 and 4 information needs" in prompt


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
    )

    assert result.fallback_used is True
    assert result.decomposer_name == "heuristic_information_need_decomposer"
    assert len(result.information_needs) == 2


async def test_decomposition_node_stores_an_independent_state_and_trace_payload() -> None:
    state = QueryState(question="What are the pipeline flows and how do they function?")

    state = await DecomposeInformationNeedsNode(HeuristicInformationNeedDecomposer())(state)

    assert state.information_need_decomposition is not None
    assert state.query_classification is None
    assert state.retrieval_plan is None
    assert state.metadata["information_need_decomposition"]["information_need_count"] == 2
