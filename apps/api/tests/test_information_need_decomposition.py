from __future__ import annotations

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.query_graph.nodes import DecomposeInformationNeedsNode
from packages.rag_core.query_understanding.decomposition import (
    HeuristicInformationNeedDecomposer,
    InformationNeedDecompositionError,
    LLMInformationNeedDecomposer,
    build_information_need_prompt,
    information_need_decomposition_response_schema,
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


@pytest.mark.parametrize(
    "raw_response",
    [
        (
            '{"information_needs":[{"description":123,"retrieval_query":"one"}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"too long"}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"one"}],'
            '"rationale":"too long"}'
        ),
        (
            '{"information_needs":[{"description":"   ","retrieval_query":"one"}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"   "}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"one"}],'
            '"rationale":"   "}'
        ),
    ],
)
def test_decomposition_parser_rejects_numeric_and_overlong_text(
    raw_response: str,
) -> None:
    with pytest.raises(InformationNeedDecompositionError):
        parse_information_need_decomposition(
            raw_response,
            max_need_chars=5,
            max_rationale_chars=5,
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
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or ["not-json", "not-json"]
        self.schemas: list[dict[str, object]] = []

    async def generate(self, prompt: str) -> str:
        del prompt
        return self.responses[0]

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        del prompt
        self.schemas.append(response_schema)
        return self.responses.pop(0)


def test_information_need_schema_has_exact_nested_shape_and_count_bounds() -> None:
    schema = information_need_decomposition_response_schema(
        max_information_needs=4,
        max_need_chars=120,
        max_rationale_chars=240,
    )

    assert schema["additionalProperties"] is False
    needs = schema["properties"]["information_needs"]
    assert needs["minItems"] == 1
    assert needs["maxItems"] == 4
    assert needs["items"]["additionalProperties"] is False
    assert needs["items"]["properties"]["description"]["maxLength"] == 120
    assert schema["properties"]["rationale"]["maxLength"] == 240


async def test_llm_decomposer_repairs_duplicate_query_semantics() -> None:
    llm = InvalidDecompositionLLM(
        (
            '{"information_needs":['
            '{"description":"One","retrieval_query":"same"},'
            '{"description":"Two","retrieval_query":"same"}'
            '],"rationale":"Needs repair."}'
        ),
        (
            '{"information_needs":['
            '{"description":"One","retrieval_query":"first"},'
            '{"description":"Two","retrieval_query":"second"}'
            '],"rationale":"Distinct needs."}'
        ),
    )
    decomposer = LLMInformationNeedDecomposer(llm_provider=llm)

    result = await decomposer.decompose("Find one and identify two.")

    assert len(llm.schemas) == 2
    assert result.structured_output is not None
    assert result.structured_output.outcome == "repair_valid"
    assert result.structured_output.failure_code == "semantic_validation_failed"
    assert [need.retrieval_query for need in result.information_needs] == ["first", "second"]


@pytest.mark.parametrize(
    "invalid_response",
    [
        (
            '{"information_needs":[{"description":123,"retrieval_query":"one"}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"too long"}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"one"}],'
            '"rationale":"too long"}'
        ),
        (
            '{"information_needs":[{"description":"   ","retrieval_query":"one"}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"   "}],'
            '"rationale":"valid"}'
        ),
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"one"}],'
            '"rationale":"   "}'
        ),
    ],
)
async def test_llm_decomposer_repairs_schema_type_and_length_failures(
    invalid_response: str,
) -> None:
    llm = InvalidDecompositionLLM(
        invalid_response,
        (
            '{"information_needs":[{"description":"valid","retrieval_query":"one"}],'
            '"rationale":"valid"}'
        ),
    )
    decomposer = LLMInformationNeedDecomposer(
        llm_provider=llm,
        max_need_chars=5,
        max_rationale_chars=5,
    )

    result = await decomposer.decompose("Find one.")

    assert len(llm.schemas) == 2
    assert result.structured_output is not None
    assert result.structured_output.outcome == "repair_valid"
    assert result.structured_output.failure_code == "schema_mismatch"


async def test_llm_decomposer_falls_back_to_deterministic_split() -> None:
    llm = InvalidDecompositionLLM()
    decomposer = LLMInformationNeedDecomposer(
        llm_provider=llm,
        fail_open=True,
    )

    result = await decomposer.decompose(
        "What are the pipeline flows and how do they function?",
    )

    assert result.fallback_used is True
    assert result.decomposer_name == "heuristic_information_need_decomposer"
    assert len(result.information_needs) == 2
    assert len(llm.schemas) == 2
    assert result.structured_output is not None
    assert result.structured_output.failure_code == "repair_invalid"


async def test_decomposition_node_stores_an_independent_state_and_trace_payload() -> None:
    state = QueryState(question="What are the pipeline flows and how do they function?")

    state = await DecomposeInformationNeedsNode(HeuristicInformationNeedDecomposer())(state)

    assert state.information_need_decomposition is not None
    assert state.query_classification is None
    assert state.retrieval_plan is None
    assert state.metadata["information_need_decomposition"]["information_need_count"] == 2


def test_decomposition_isolates_lane_intent_but_preserves_shared_subject_aliases() -> None:
    result = parse_information_need_decomposition(
        '''{
          "information_needs": [
            {
              "description": "Identify the main contributor to the LLM guidance project.",
              "retrieval_query": "who was the LLMguidance LLM guidance project main contributor",
              "subject_context": "LLM guidance project"
            },
            {
              "description": "Summarize the key points of the LLM guidance project.",
              "retrieval_query": "who was the main contributor LLMguidance LLM guidance project key points",
              "subject_context": "LLM guidance project"
            }
          ],
          "rationale": "The question asks for a contributor and separate project key points."
        }''',
        original_question=(
            "Who was the main contributor to the LLMguidance project and what are its key points?"
        ),
    )

    contributor, key_points = result.information_needs
    assert contributor.retrieval_query == (
        "who was the LLMguidance LLM guidance project main contributor"
    )
    assert key_points.retrieval_query == "LLMguidance LLM guidance project key points"
    assert "main contributor" not in key_points.retrieval_query.casefold()
    assert "llmguidance" in key_points.retrieval_query.casefold()
    assert "llm guidance project" in key_points.retrieval_query.casefold()


def test_decomposition_prompt_distinguishes_shared_subject_from_lane_intent() -> None:
    prompt = build_information_need_prompt(
        "Who led the LLMguidance project and what are its key points?",
    )

    assert "Shared subject keywords and useful aliases may repeat" in prompt
    assert "Keep each lane's answer intent exclusive" in prompt
