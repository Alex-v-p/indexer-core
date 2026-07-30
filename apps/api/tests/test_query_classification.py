from __future__ import annotations

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.query_understanding.classification import (
    HeuristicQueryClassifier,
    LLMQueryClassifier,
    MetadataFilterHint,
    QueryClassificationError,
    QueryType,
    parse_query_classification,
    query_classification_response_schema,
)
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.retrieval.retrievers import EmptyRetriever


@pytest.mark.parametrize(
    ("question", "expected_type"),
    [
        ("What port does the API use?", QueryType.FACTUAL_LOOKUP),
        ("Explain how the ingestion architecture works.", QueryType.BROAD_EXPLANATION),
        ("Compare the baseline and hybrid pipelines.", QueryType.COMPARISON),
        ("What changed in the latest version of the policy?", QueryType.VERSION_SPECIFIC),
    ],
)
async def test_heuristic_classifier_covers_supported_query_types(
    question: str,
    expected_type: QueryType,
) -> None:
    classification = await HeuristicQueryClassifier().classify(question)

    assert classification.query_type is expected_type
    assert 0.0 <= classification.confidence <= 1.0
    assert classification.classifier_name == "heuristic_rules"


async def test_heuristic_classifier_detects_metadata_filter_dimensions() -> None:
    classification = await HeuristicQueryClassifier().classify(
        "According to deployment-guide.pdf, what does section 4 say in the 2025 revision?",
    )

    assert classification.query_type is QueryType.VERSION_SPECIFIC
    assert classification.needs_metadata_filters is True
    assert set(classification.metadata_filter_hints) == {
        MetadataFilterHint.DOCUMENT,
        MetadataFilterHint.DOCUMENT_VERSION,
        MetadataFilterHint.DATE_RANGE,
        MetadataFilterHint.SECTION,
        MetadataFilterHint.FILE_TYPE,
    }


def test_llm_classification_parser_validates_structured_output() -> None:
    classification = parse_query_classification(
        """```json
        {
          "query_type": "comparison",
          "confidence": 0.93,
          "needs_metadata_filters": true,
          "metadata_filter_hints": ["document", "document_version"],
          "rationale": "The question compares two named document versions."
        }
        ```""",
    )

    assert classification.query_type is QueryType.COMPARISON
    assert classification.confidence == 0.93
    assert classification.metadata_filter_hints == (
        MetadataFilterHint.DOCUMENT,
        MetadataFilterHint.DOCUMENT_VERSION,
    )
    assert classification.fallback_used is False


def test_version_specific_parser_enforces_version_filter_hint() -> None:
    classification = parse_query_classification(
        """{
          "query_type": "version_specific",
          "confidence": 0.8,
          "needs_metadata_filters": false,
          "metadata_filter_hints": [],
          "rationale": "A particular release is required."
        }""",
    )

    assert classification.needs_metadata_filters is True
    assert classification.metadata_filter_hints == (MetadataFilterHint.DOCUMENT_VERSION,)


@pytest.mark.parametrize(
    "raw_response",
    [
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],"rationale":123}'
        ),
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],"rationale":"too long"}'
        ),
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],"rationale":"   "}'
        ),
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":true,"metadata_filter_hints":["document","document"],'
            '"rationale":"valid"}'
        ),
    ],
)
def test_classification_parser_rejects_numeric_overlong_and_duplicate_fields(
    raw_response: str,
) -> None:
    with pytest.raises(QueryClassificationError):
        parse_query_classification(raw_response, max_rationale_chars=5)


class InvalidClassificationLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        del prompt
        return "not-json"

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        del prompt, response_schema
        self.calls += 1
        return "not-json"


class StaticClassificationLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or [
            (
                '{"query_type":"factual_lookup","confidence":0.9,'
                '"needs_metadata_filters":false,"metadata_filter_hints":[],'
                '"rationale":"A direct lookup."}'
            ),
        ]
        self.schemas: list[dict[str, object]] = []

    async def generate(self, prompt: str) -> str:
        del prompt
        return self.responses[0]

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        del prompt
        self.schemas.append(response_schema)
        return self.responses.pop(0)


async def test_llm_classifier_enforces_detected_version_constraint_when_model_misses_it() -> None:
    classifier = LLMQueryClassifier(llm_provider=StaticClassificationLLM())

    classification = await classifier.classify("What does the latest policy require?")

    assert classification.query_type is QueryType.VERSION_SPECIFIC
    assert classification.version_constraint.mode.value == "latest"
    assert classification.needs_metadata_filters is True
    assert MetadataFilterHint.DOCUMENT_VERSION in classification.metadata_filter_hints
    assert classification.structured_output is not None
    assert classification.structured_output.outcome == "primary_valid"


def test_query_classification_schema_is_exact_and_bounded() -> None:
    schema = query_classification_response_schema(max_rationale_chars=123)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["query_type"]["enum"] == [
        "factual_lookup",
        "broad_explanation",
        "comparison",
        "version_specific",
    ]
    assert schema["properties"]["metadata_filter_hints"]["items"]["enum"] == [
        "document",
        "document_version",
        "date_range",
        "section",
        "file_type",
        "author",
    ]
    assert schema["properties"]["rationale"]["maxLength"] == 123


async def test_llm_classifier_repairs_once_and_records_diagnostics() -> None:
    llm = StaticClassificationLLM(
        "not-json",
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],'
            '"rationale":"A direct lookup."}'
        ),
    )
    classifier = LLMQueryClassifier(llm_provider=llm)

    classification = await classifier.classify("What is indexed?")

    assert len(llm.schemas) == 2
    assert llm.schemas[0] == llm.schemas[1]
    assert classification.structured_output is not None
    assert classification.structured_output.outcome == "repair_valid"
    assert classification.structured_output.failure_code == "invalid_json"
    assert classification.structured_output.attempt_count == 2


@pytest.mark.parametrize(
    "invalid_response",
    [
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],"rationale":123}'
        ),
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],"rationale":"over limit"}'
        ),
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],"rationale":"   "}'
        ),
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":true,"metadata_filter_hints":["document","document"],'
            '"rationale":"valid"}'
        ),
    ],
)
async def test_llm_classifier_repairs_schema_type_length_and_uniqueness_failures(
    invalid_response: str,
) -> None:
    llm = StaticClassificationLLM(
        invalid_response,
        (
            '{"query_type":"factual_lookup","confidence":0.9,'
            '"needs_metadata_filters":false,"metadata_filter_hints":[],'
            '"rationale":"valid"}'
        ),
    )
    classifier = LLMQueryClassifier(llm_provider=llm, max_rationale_chars=5)

    classification = await classifier.classify("What is indexed?")

    assert len(llm.schemas) == 2
    assert classification.structured_output is not None
    assert classification.structured_output.outcome == "repair_valid"
    assert classification.structured_output.failure_code == "schema_mismatch"


async def test_llm_classifier_falls_back_to_deterministic_rules() -> None:
    llm = InvalidClassificationLLM()
    classifier = LLMQueryClassifier(llm_provider=llm, fail_open=True)

    classification = await classifier.classify("Compare release 1 with release 2.")

    assert classification.query_type is QueryType.COMPARISON
    assert classification.fallback_used is True
    assert classification.classifier_name == "heuristic_rules"
    assert llm.calls == 2
    assert classification.structured_output is not None
    assert classification.structured_output.outcome == "fallback"
    assert classification.structured_output.failure_code == "repair_invalid"


async def test_llm_classifier_can_fail_closed() -> None:
    classifier = LLMQueryClassifier(llm_provider=InvalidClassificationLLM(), fail_open=False)

    with pytest.raises(QueryClassificationError):
        await classifier.classify("What is indexed?")


class StaticAnswerLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "unused"


async def test_graph_stores_classification_in_state_and_trace_metadata() -> None:
    graph = build_baseline_rag_graph(retriever=EmptyRetriever(), llm_provider=StaticAnswerLLM())

    state = await graph.run(QueryState(question="Compare the current and previous report versions."))

    assert state.query_classification is not None
    assert state.query_classification.query_type is QueryType.COMPARISON
    assert state.metadata["query_classification"]["needs_metadata_filters"] is True
    classification_step = state.trace[1]
    assert classification_step.name == "classify_query"
    assert classification_step.step_type == "classification"
    assert classification_step.metadata["classification"]["query_type"] == "comparison"
    assert "filter_hints=document_version" in (classification_step.output_summary or "")
