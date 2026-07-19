from __future__ import annotations

from datetime import UTC, datetime

from packages.rag_core.agents import QueryState
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.constraint_validation import ConstraintValidationStatus


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The matching source describes strict metadata filtering [1]."


class StaticRetriever:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self.evidence = evidence

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        del question
        return self.evidence[:top_k]


def _dated_evidence(*, uploaded_at: datetime, published_at: datetime | None = None) -> EvidenceItem:
    metadata: dict[str, object] = {
        "original_filename": "realization.md",
        "document_title": "Realization",
        "document_version_number": 5,
        "document_version_label": "v5",
        "is_latest_version": True,
        "uploaded_at": uploaded_at.isoformat(),
        "uploaded_at_epoch": uploaded_at.timestamp(),
        "section_title": "Retrieval constraints",
    }
    if published_at is not None:
        metadata["published_at"] = published_at.isoformat()
        metadata["published_at_epoch"] = published_at.timestamp()
    return EvidenceItem(
        rank=1,
        text="Explicit metadata constraints are enforced before answer generation.",
        score=0.93,
        metadata=metadata,
    )


async def test_requested_month_with_no_matching_evidence_does_not_fall_back() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(
        retriever=StaticRetriever(
            [_dated_evidence(uploaded_at=datetime(2026, 6, 10, tzinfo=UTC))],
        ),
        llm_provider=llm,
    )

    state = await graph.run(
        QueryState(question="Only use data from May 2026 to explain metadata filtering."),
    )

    assert state.constraint_validation is not None
    assert state.constraint_validation.status is ConstraintValidationStatus.NO_MATCH
    assert state.constraint_validation.candidate_count == 1
    assert state.constraint_validation.matched_count == 0
    assert state.constraint_validation.rejected_count == 1
    assert state.retrieved_evidence == []
    assert state.citations == []
    assert llm.prompts == []
    assert state.answer is not None
    assert "No indexed evidence matched the requested metadata scope" in state.answer
    assert "did not use documents outside" in state.answer
    assert state.metadata["constraint_validation"]["blocked"] is True
    assert state.metadata["evidence_context"]["sources"] == []


async def test_matching_date_metadata_is_included_compactly_in_generation_context() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(
        retriever=StaticRetriever(
            [
                _dated_evidence(
                    uploaded_at=datetime(2026, 5, 10, 9, 30, tzinfo=UTC),
                    published_at=datetime(2026, 4, 28, tzinfo=UTC),
                ),
            ],
        ),
        llm_provider=llm,
    )

    state = await graph.run(
        QueryState(question="Only use data from May 2026 to explain metadata filtering."),
    )

    assert state.constraint_validation is not None
    assert state.constraint_validation.status is ConstraintValidationStatus.MATCHED
    assert len(state.retrieved_evidence) == 1
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    assert "Active metadata constraints:" in prompt
    assert "recorded document date" in prompt
    assert "date_scope_match=uploaded_at" in prompt
    assert "uploaded_at=2026-05-10T09:30:00+00:00" in prompt
    assert "published_at=2026-04-28T00:00:00+00:00" in prompt
    assert "file=realization.md" in prompt
    assert "section=Retrieval constraints" in prompt
    assert state.evidence_context is not None
    source_values = dict(state.evidence_context.sources[0].values)
    assert source_values["date_scope_match"] == "uploaded_at"
    assert source_values["uploaded_at"] == "2026-05-10T09:30:00+00:00"
    assert source_values["published_at"] == "2026-04-28T00:00:00+00:00"


async def test_ordinary_question_does_not_add_unused_date_or_version_metadata_to_prompt() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(
        retriever=StaticRetriever(
            [
                _dated_evidence(
                    uploaded_at=datetime(2026, 5, 10, tzinfo=UTC),
                    published_at=datetime(2026, 4, 28, tzinfo=UTC),
                ),
            ],
        ),
        llm_provider=llm,
    )

    await graph.run(QueryState(question="How does metadata filtering work?"))

    prompt = llm.prompts[0]
    assert "file=realization.md" in prompt
    assert "section=Retrieval constraints" in prompt
    assert "uploaded_at=" not in prompt
    assert "published_at=" not in prompt
    assert "version=v5" not in prompt
    assert "latest_version=" not in prompt
