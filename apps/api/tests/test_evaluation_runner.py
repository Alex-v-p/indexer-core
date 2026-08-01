from __future__ import annotations

import json
from pathlib import Path
import uuid

import pytest

from packages.rag_core.evaluation import (
    BehavioralExpectations,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRunner,
    EvidenceExpectation,
    evaluation_dataset_requires_subject_scope,
    write_evaluation_report,
)
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.document_scope import (
    CoverageMode,
    DocumentScope,
    QueryProductOutcome,
    ResolvedSubjectScopeSnapshot,
    ScopeResolutionSource,
    SubjectDocumentLane,
    SubjectScopeCatalogEntry,
)
from packages.rag_core.subjects import SubjectKind


class StaticRetriever:
    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        del question
        return [
            EvidenceItem(rank=1, text="An unrelated chunk."),
            EvidenceItem(
                rank=2,
                text="The baseline graph runs retrieval before answer generation.",
                score=0.9,
                metadata={"original_filename": "source.md"},
            ),
        ][:top_k]


class StaticLLM:
    async def generate(self, prompt: str) -> str:
        assert "retrieval before answer generation" in prompt
        return "Retrieval runs before answer generation [2]."


async def test_evaluation_runner_executes_full_graph_and_writes_report(tmp_path: Path) -> None:
    graph = build_baseline_rag_graph(retriever=StaticRetriever(), llm_provider=StaticLLM())
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="runner-test",
        version="1.0.0",
        default_top_k=2,
        cases=(
            EvaluationCase(
                id="graph-order",
                question="What is the graph order?",
                expected_answer="Retrieval then generation.",
                expected_evidence=(
                    EvidenceExpectation(
                        metadata={"original_filename": "source.md"},
                        text_contains=("retrieval before answer generation",),
                    ),
                ),
            ),
        ),
    )

    report = await EvaluationRunner(graph=graph).run(dataset)

    assert report.succeeded_cases == 1
    assert report.failed_cases == 0
    assert report.metrics.recall_at_k.value == 1.0
    assert report.metrics.mrr.value == 0.5
    assert report.metrics.citation_hit_rate.value == 1.0
    assert report.metrics.answer_faithfulness.status == "not_implemented"
    assert [step["name"] for step in report.cases[0].trace] == ["select_pipeline", "classify_query", "retrieve", "prepare_evidence_context", "generate_answer"]
    assert report.cases[0].actual_answer == "Retrieval runs before answer generation [2]."

    output_path = write_evaluation_report(report, tmp_path / "report.json")
    body = json.loads(output_path.read_text(encoding="utf-8"))
    assert body["pipeline_name"] == "baseline_rag"
    assert body["cases"][0]["expected_answer"] == "Retrieval then generation."
    assert body["cases"][0]["behavioral_expectations"]["max_scope_leakage"] is None
    assert body["cases"][0]["metrics"]["scope_leakage_rate"]["status"] == "not_applicable"
    assert body["cases"][0]["trace"][4]["name"] == "generate_answer"


class SometimesFailingGraph:
    name = "test_graph"
    version = "1.0.0"

    async def run(self, state):
        if state.question == "fail":
            raise RuntimeError("graph failed")
        state.answer = "ok"
        return state


async def test_evaluation_runner_continues_after_case_failure() -> None:
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="failure-test",
        version="1.0.0",
        cases=(
            EvaluationCase(id="failed", question="fail"),
            EvaluationCase(id="succeeded", question="continue"),
        ),
    )

    report = await EvaluationRunner(graph=SometimesFailingGraph()).run(dataset)

    assert report.failed_cases == 1
    assert report.succeeded_cases == 1
    assert [case.status for case in report.cases] == ["failed", "succeeded"]
    assert report.cases[0].error_message == "graph failed"


async def test_scoped_evaluation_requires_provider_before_graph() -> None:
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="scoped",
        version="1.0",
        cases=(EvaluationCase(id="scope", question="DAF status", requested_subject_names=("DAF",)),),
    )

    with pytest.raises(ValueError, match="SubjectScopeEvaluationProvider"):
        await EvaluationRunner(graph=SometimesFailingGraph()).run(dataset)


class RecordingNoEvidenceGraph:
    name = "no-evidence"
    version = "1.0"

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def run(self, state):
        self.questions.append(state.question)
        return state


async def test_global_no_evidence_expectation_runs_graph_without_scope_provider() -> None:
    graph = RecordingNoEvidenceGraph()
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="global-unanswerable",
        version="1.0",
        cases=(
            EvaluationCase(
                id="missing",
                question="What is unavailable globally?",
                behavioral_expectations=BehavioralExpectations(
                    expected_product_outcome="no_evidence",
                ),
            ),
        ),
    )

    report = await EvaluationRunner(graph=graph).run(dataset)

    assert graph.questions == ["What is unavailable globally?"]
    assert report.failed_cases == 0
    assert report.cases[0].actual_product_outcome == "no_evidence"
    assert report.cases[0].metrics.clarification_correctness.value == 1.0


@pytest.mark.parametrize("outcome", ("no_evidence", "clarification_required"))
async def test_opted_in_scoped_terminal_expectations_require_provider(outcome: str) -> None:
    dataset = EvaluationDataset(
        schema_version="1.0",
        name="scoped-terminal",
        version="1.0",
        metadata={"subject_scope_evaluation": True},
        cases=(
            EvaluationCase(
                id="terminal",
                question="Scoped terminal case",
                behavioral_expectations=BehavioralExpectations(
                    expected_product_outcome=outcome,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="SubjectScopeEvaluationProvider"):
        await EvaluationRunner(graph=RecordingNoEvidenceGraph()).run(dataset)


def test_stability_scope_guard_detects_case_trigger_without_dataset_opt_in() -> None:
    scoped = EvaluationDataset(
        schema_version="1.0",
        name="case-scoped",
        version="1.0",
        cases=(
            EvaluationCase(
                id="scope",
                question="DAF status",
                requested_subject_names=("DAF",),
            ),
        ),
    )
    global_baseline = EvaluationDataset(
        schema_version="1.0",
        name="global",
        version="1.0",
        cases=(EvaluationCase(id="global", question="Global question"),),
    )

    assert evaluation_dataset_requires_subject_scope(scoped) is True
    assert evaluation_dataset_requires_subject_scope(global_baseline) is False


class ScopeAwareGraph:
    name = "scope-aware"
    version = "1.0"

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def run(self, state):
        self.questions.append(state.question)
        for lane in state.subject_lanes:
            document_id = lane.document_scope.allowed_document_ids[0]
            state.retrieved_evidence.append(
                EvidenceItem(
                    rank=len(state.retrieved_evidence) + 1,
                    text=lane.subject_name,
                    document_id=document_id,
                    subject_lane_id=lane.lane_id,
                    subject_id=lane.subject_id,
                    subject_name=lane.subject_name,
                ),
            )
        if not state.subject_lanes and state.document_scope.allowed_document_ids:
            state.retrieved_evidence.append(
                EvidenceItem(rank=1, text="DAF", document_id=state.document_scope.allowed_document_ids[0]),
            )
        state.answer = "answer"
        return state


class FakeSubjectScopeProvider:
    daf_id = uuid.UUID("10000000-0000-0000-0000-000000000001")
    internship_id = uuid.UUID("10000000-0000-0000-0000-000000000002")
    daf_doc = uuid.UUID("20000000-0000-0000-0000-000000000001")
    internship_doc = uuid.UUID("20000000-0000-0000-0000-000000000002")

    async def resolve(self, request):
        catalog = (
            SubjectScopeCatalogEntry(self.daf_id, SubjectKind.PROJECT, "DAF"),
            SubjectScopeCatalogEntry(self.internship_id, SubjectKind.PROJECT, "Large Internship"),
        )
        if "Shared" in request.question or "Zephyr" in request.question:
            return ResolvedSubjectScopeSnapshot(
                requested_subject_ids=(), catalog=catalog, matched_subject_ids=(),
                document_scope=DocumentScope.strict_scope(()), source=ScopeResolutionSource.INFERRED_PROJECT,
                confidence=0.5, catalog_revision="catalog", policy_revision="policy",
                product_outcome=QueryProductOutcome.CLARIFICATION_REQUIRED,
                clarification_reason="ambiguous_or_unknown", coverage_mode=request.coverage_mode,
            )
        if "Compare" in request.question:
            lanes = (
                SubjectDocumentLane(self.daf_id, "DAF", DocumentScope.strict_scope((self.daf_doc,))),
                SubjectDocumentLane(self.internship_id, "Large Internship", DocumentScope.strict_scope((self.internship_doc,))),
            )
            return ResolvedSubjectScopeSnapshot(
                requested_subject_ids=(), catalog=catalog, matched_subject_ids=(self.daf_id, self.internship_id),
                document_scope=DocumentScope.strict_scope((self.daf_doc, self.internship_doc)),
                source=ScopeResolutionSource.INFERRED_PROJECT, confidence=1.0,
                catalog_revision="catalog", policy_revision="policy", coverage_mode=request.coverage_mode,
                subject_lanes=lanes, comparison_requested=True,
            )
        return ResolvedSubjectScopeSnapshot(
            requested_subject_ids=(self.daf_id,), catalog=catalog, matched_subject_ids=(self.daf_id,),
            document_scope=DocumentScope.strict_scope((self.daf_doc,)), source=ScopeResolutionSource.EXPLICIT,
            confidence=1.0, catalog_revision="catalog", policy_revision="policy", coverage_mode=request.coverage_mode,
        )


async def test_scoped_runner_filters_before_graph_short_circuits_and_preserves_lanes() -> None:
    dataset = EvaluationDataset(
        schema_version="1.0", name="scoped", version="1.0",
        metadata={"subject_scope_evaluation": True},
        cases=(
            EvaluationCase(id="explicit", question="DAF status", requested_subject_names=("DAF",), behavioral_expectations=BehavioralExpectations(expected_scope_subject_names=("DAF",), forbidden_document_ids=(str(FakeSubjectScopeProvider.internship_doc),), max_scope_leakage=0)),
            EvaluationCase(id="ambiguous", question="Shared Initiative status", behavioral_expectations=BehavioralExpectations(expected_product_outcome="clarification_required")),
            EvaluationCase(id="unknown", question="Project Zephyr status", behavioral_expectations=BehavioralExpectations(expected_product_outcome="clarification_required")),
            EvaluationCase(id="compare", question="Compare DAF and Large Internship", coverage_mode=CoverageMode.MULTI_DOCUMENT.value, behavioral_expectations=BehavioralExpectations(expected_scope_subject_names=("DAF", "Large Internship"), expected_lane_subject_names=("DAF", "Large Internship"))),
        ),
    )
    graph = ScopeAwareGraph()
    provider = FakeSubjectScopeProvider()

    report = await EvaluationRunner(graph=graph, subject_scope_provider=provider).run(dataset)

    assert graph.questions == ["DAF status", "Compare DAF and Large Internship"]
    assert report.cases[0].metrics.scope_leakage_count.value == 0.0
    assert report.cases[1].actual_product_outcome == "clarification_required"
    assert report.cases[2].actual_product_outcome == "clarification_required"
    assert report.cases[3].metrics.lane_coverage.value == 1.0
    assert len(report.cases[3].actual_subject_scope["subject_lanes"]) == 2
