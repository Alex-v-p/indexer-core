from __future__ import annotations

import json
from pathlib import Path

from packages.rag_core.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationRunner,
    EvidenceExpectation,
    write_evaluation_report,
)
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.retrieval import EvidenceItem


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
