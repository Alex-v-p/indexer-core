from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    StabilityAttemptSnapshot,
    StabilityEvaluationRunner,
    StructuredDiagnosticSnapshot,
    answer_tokens,
    calculate_stability_metrics,
    canonical_route_signature,
    normalize_answer,
    safe_runtime_profile,
    safe_structured_diagnostics,
    stable_evidence_identity,
    write_stability_evaluation_report,
)
from packages.rag_core.generation import AnswerPresentation, AnswerPresentationOutcome
from packages.rag_core.query_understanding.classification import QueryType
from packages.rag_core.query_understanding.planning import RetrievalPlan, RetrievalStrategy
from packages.rag_core.retrieval import EvidenceItem


def test_stable_evidence_identity_uses_documented_priority_order() -> None:
    chunk_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    version_id = uuid.UUID("00000000-0000-0000-0000-000000000002")

    assert (
        stable_evidence_identity(
            EvidenceItem(
                rank=1,
                text="same text",
                qdrant_chunk_index_id=chunk_id,
                document_version_id=version_id,
                metadata={"qdrant_point_id": "point-1", "ordinal": 4},
            ),
        )
        == f"chunk:{chunk_id}"
    )
    assert (
        stable_evidence_identity(
            EvidenceItem(
                rank=1,
                text="same text",
                document_version_id=version_id,
                metadata={"point_id": "point-1", "ordinal": 4},
            ),
        )
        == "point:point-1"
    )
    assert (
        stable_evidence_identity(
            EvidenceItem(
                rank=1,
                text="same text",
                document_version_id=version_id,
                metadata={"qdrant_point_id": 0, "point_id": "later", "ordinal": 4},
            ),
        )
        == "point:0"
    )
    assert (
        stable_evidence_identity(
            EvidenceItem(
                rank=1,
                text="same text",
                document_version_id=version_id,
                metadata={"chunk_ordinal": 4},
            ),
        )
        == f"document_version_ordinal:{version_id}:4"
    )
    assert stable_evidence_identity(EvidenceItem(rank=1, text="same text")) == (
        f"text_sha256:{hashlib.sha256(b'same text').hexdigest()}"
    )


def test_answer_normalization_and_tokenization_are_deterministic() -> None:
    assert normalize_answer("  ＨＥＬＬＯ\tWorld\n") == "hello world"
    assert answer_tokens("Hello, HELLO—world!") == frozenset({"hello", "world"})


def test_route_signature_uses_coded_fields_and_excludes_rationales() -> None:
    def build_state(rationale: str) -> QueryState:
        state = QueryState(question="question", top_k=7)
        state.pipeline_name = "agentic_rag"
        state.retrieval_plan = RetrievalPlan(
            strategy=RetrievalStrategy.HYBRID,
            selected_pipeline_name="hybrid_rag",
            rationale=rationale,
            planner_name="planner",
            based_on_query_type=QueryType.FACTUAL_LOOKUP,
        )
        state.metadata["retrieval_retry"] = {
            "attempts": [
                {
                    "strategy": "hybrid",
                    "pipeline_name": "hybrid_rag",
                    "top_k": 7,
                    "decision_rationale": rationale,
                },
                {
                    "strategy": "rerank",
                    "pipeline_name": "hybrid_llm_rerank_rag",
                    "top_k": 10,
                    "decision_rationale": rationale,
                },
            ],
            "final_strategy": "rerank",
            "final_pipeline_name": "hybrid_llm_rerank_rag",
            "final_top_k": 10,
            "stop_reason": "evidence_sufficient",
            "stop_rationale": rationale,
        }
        return state

    signature = canonical_route_signature(build_state("first free-form rationale"))

    assert signature == canonical_route_signature(build_state("different rationale"))
    assert signature == (
        "pipeline:agentic_rag",
        "top_k:7",
        "query:strategy:hybrid",
        "query:selected_pipeline_name:hybrid_rag",
        "retry:attempt:1:strategy:hybrid",
        "retry:attempt:1:pipeline_name:hybrid_rag",
        "retry:attempt:1:top_k:7",
        "retry:attempt:2:strategy:rerank",
        "retry:attempt:2:pipeline_name:hybrid_llm_rerank_rag",
        "retry:attempt:2:top_k:10",
        "retry:final_strategy:rerank",
        "retry:final_pipeline_name:hybrid_llm_rerank_rag",
        "retry:final_top_k:10",
        "retry:stop_reason:evidence_sufficient",
    )


def test_diagnostic_and_runtime_snapshots_keep_only_safe_fields() -> None:
    diagnostics = safe_structured_diagnostics(
        {
            "query_classification": {
                "structured_output": {
                    "schema_version": "1.0",
                    "outcome": "repair_valid",
                    "failure_code": "invalid_json",
                    "attempt_count": 2,
                    "repair_attempted": True,
                    "raw_output": "must not be copied",
                },
            },
            "unrelated": {"outcome": "fallback", "attempt_count": 2},
        },
    )

    assert diagnostics == (
        StructuredDiagnosticSnapshot(
            stage="query_classification",
            outcome="repair_valid",
            failure_code="invalid_json",
            attempt_count=2,
            repair_attempted=True,
        ),
    )

    state = QueryState(question="question")
    state.pipeline_name = "baseline_rag"
    state.pipeline_version = "1.2.3"
    state.metadata["model_profile"] = {
        "model": "local-model",
        "api_key": "must-not-appear",
        "openaiApiKey": "must-not-appear",
        "api_token": "must-not-appear",
        "session_token": "must-not-appear",
        "access_token": "must-not-appear",
        "refresh-token": "must-not-appear",
        "authToken": "must-not-appear",
        "bearer_token": "must-not-appear",
        "password": "must-not-appear",
        "client_secret": "must-not-appear",
        "credentials": "must-not-appear",
        "max_output_tokens": 2_048,
        "structured_max_output_tokens": 4_096,
        "nested": {
            "token": "must-not-appear",
            "temperature": 0,
            "token_budget_control": "preserved",
        },
    }
    profile = safe_runtime_profile(state)

    assert profile == {
        "pipeline_name": "baseline_rag",
        "pipeline_version": "1.2.3",
        "model_profile": {
            "model": "local-model",
            "max_output_tokens": 2_048,
            "structured_max_output_tokens": 4_096,
            "nested": {
                "temperature": 0,
                "token_budget_control": "preserved",
            },
        },
    }


def test_stability_metrics_use_successful_within_case_attempt_pairs() -> None:
    primary = StructuredDiagnosticSnapshot(
        stage="classification",
        outcome="primary_valid",
        failure_code=None,
        attempt_count=1,
        repair_attempted=False,
    )
    repaired = StructuredDiagnosticSnapshot(
        stage="classification",
        outcome="repair_valid",
        failure_code="invalid_json",
        attempt_count=2,
        repair_attempted=True,
    )
    fallback = StructuredDiagnosticSnapshot(
        stage="classification",
        outcome="fallback",
        failure_code="repair_invalid",
        attempt_count=2,
        repair_attempted=True,
    )
    groups = (
        (
            _attempt(1, answer="Alpha beta", evidence=("a", "b"), diagnostic=primary),
            _attempt(2, answer=" alpha   BETA ", evidence=("a", "c"), diagnostic=repaired),
        ),
        (
            _attempt(1, answer="Only success", evidence=(), diagnostic=fallback),
            _attempt(2, status="failed"),
        ),
    )

    metrics = calculate_stability_metrics(groups)

    assert metrics.technical_success_rate.value == pytest.approx(0.75)
    assert metrics.outcome_consistency.value == 1.0
    assert metrics.route_signature_consistency.value == 1.0
    assert metrics.evidence_exact_set_agreement.value == pytest.approx(0.5)
    assert metrics.evidence_mean_pairwise_jaccard.value == pytest.approx(2 / 3)
    assert metrics.normalized_answer_exact_match_rate.value == 1.0
    assert metrics.normalized_answer_mean_pairwise_token_jaccard.value == 1.0
    assert metrics.presentation_signature_consistency.value == 1.0
    assert metrics.structured_stage_rates[0].observation_count == 3
    assert metrics.structured_stage_rates[0].repair_rate.value == pytest.approx(2 / 3)
    assert metrics.structured_stage_rates[0].fallback_rate.value == pytest.approx(1 / 3)


def test_stability_metrics_define_single_success_and_zero_success_cases() -> None:
    single = calculate_stability_metrics(((_attempt(1),),))
    zero = calculate_stability_metrics(((_attempt(1, status="failed"),),))

    assert single.technical_success_rate.value == 1.0
    assert single.evidence_exact_set_agreement.value == 1.0
    assert single.evidence_mean_pairwise_jaccard.value == 1.0
    assert zero.technical_success_rate.value == 0.0
    assert zero.outcome_consistency.status == "not_applicable"
    assert zero.outcome_consistency.value is None


class RepeatedGraph:
    name = "repeated_graph"
    version = "1.0.0"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, state: QueryState) -> QueryState:
        self.calls += 1
        state.pipeline_name = self.name
        state.pipeline_version = self.version
        if self.calls == 2:
            raise RuntimeError("sensitive failure details")
        state.answer = f"Answer {self.calls}"
        state.answer_presentation = AnswerPresentation(
            outcome=AnswerPresentationOutcome.COMPLETE,
            title="Answer",
            body=state.answer,
            supported_information=(state.answer,),
        )
        state.retrieved_evidence = [
            EvidenceItem(
                rank=1,
                text="Evidence",
                metadata={"point_id": f"point-{self.calls % 2}"},
            ),
        ]
        state.metadata["query_classification"] = {
            "structured_output": {
                "schema_version": "1.0",
                "outcome": "primary_valid",
                "failure_code": None,
                "attempt_count": 1,
                "repair_attempted": False,
            },
        }
        return state


async def test_stability_runner_groups_repetitions_and_writes_separate_report(
    tmp_path: Path,
) -> None:
    graph = RepeatedGraph()
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="stability test",
        version="1.0.0",
        default_top_k=3,
        cases=(
            EvaluationCase(id="first", question="First?"),
            EvaluationCase(id="second", question="Second?", top_k=4),
        ),
    )

    report = await StabilityEvaluationRunner(graph=graph).run(dataset, repetitions=3)

    assert graph.calls == 6
    assert report.schema_version == "1.0"
    assert report.report_type == "repeated_query_stability"
    assert report.repetitions == 3
    assert report.total_cases == 2
    assert report.total_attempts == 6
    assert report.succeeded_attempts == 5
    assert report.failed_attempts == 1
    assert [case.top_k for case in report.cases] == [3, 4]
    assert [[attempt.attempt_number for attempt in case.attempts] for case in report.cases] == [
        [1, 2, 3],
        [1, 2, 3],
    ]
    assert report.cases[0].attempts[1].error_type == "RuntimeError"
    assert report.metrics.technical_success_rate.value == pytest.approx(5 / 6)

    output_path = write_stability_evaluation_report(report, tmp_path / "stability.json")
    body = json.loads(output_path.read_text(encoding="utf-8"))
    assert body["report_type"] == "repeated_query_stability"
    assert body["repetitions"] == 3
    assert len(body["cases"][0]["attempts"]) == 3
    assert "error_message" not in body["cases"][0]["attempts"][1]


async def test_stability_runner_rejects_one_repetition() -> None:
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="stability test",
        version="1.0.0",
        cases=(EvaluationCase(id="only", question="Question?"),),
    )

    with pytest.raises(ValueError, match="repetitions >= 2"):
        await StabilityEvaluationRunner(graph=RepeatedGraph()).run(dataset, repetitions=1)


def _attempt(
    number: int,
    *,
    status: str = "succeeded",
    answer: str | None = "Answer",
    evidence: tuple[str, ...] = (),
    diagnostic: StructuredDiagnosticSnapshot | None = None,
) -> StabilityAttemptSnapshot:
    return StabilityAttemptSnapshot(
        attempt_number=number,
        status=status,
        top_k=5,
        answer=answer,
        answer_presentation=None,
        outcome="complete" if status == "succeeded" else None,
        route_signature=("pipeline:test", "top_k:5"),
        evidence_identities=evidence,
        evidence=(),
        citations=(),
        structured_diagnostics=(diagnostic,) if diagnostic is not None else (),
        runtime_profile={"pipeline_name": "test", "pipeline_version": "1.0.0"},
        duration_ms=1,
        error_type="RuntimeError" if status == "failed" else None,
    )
