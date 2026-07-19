from __future__ import annotations

from packages.rag_core.agents import QueryState
from packages.rag_core.documents import DocumentNameConstraint
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.query_understanding.document_naming import detect_document_name_constraint
from packages.rag_core.retrieval import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers import VersionAwareRetriever
from packages.rag_core.retrieval.constraint_validation import (
    ConstraintValidationStatus,
    evidence_matches_constraints,
)


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The requested document describes strict source filtering [1]."


class StaticRetriever:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self.evidence = evidence

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        del question
        return self.evidence[:top_k]


def _evidence(
    filename: str,
    *,
    title: str = "Realization Draft 5",
    version_number: int = 5,
    document_id: str = "document-1",
) -> EvidenceItem:
    return EvidenceItem(
        rank=1,
        text="Strict document-name constraints are validated before generation.",
        score=0.95,
        metadata={
            "document_id": document_id,
            "original_filename": filename,
            "document_title": title,
            "document_version_number": version_number,
            "document_version_label": f"v{version_number}",
            "is_latest_version": version_number == 5,
        },
    )


def test_document_name_detector_extracts_explicit_titles_and_filenames() -> None:
    quoted = detect_document_name_constraint('Only use "Architecture Guide" to answer this.')
    filename = detect_document_name_constraint("According to Realization_Draft5.pdf, what changed?")
    typed = detect_document_name_constraint("Use the Architecture Guide document to explain retries.")

    assert quoted.names == ("Architecture Guide",)
    assert filename.names == ("Realization_Draft5.pdf",)
    assert typed.names == ("Architecture Guide",)
    assert typed.active is True


def test_document_name_detector_does_not_treat_generic_document_wording_as_a_name() -> None:
    constraint = detect_document_name_constraint("Explain how document metadata is validated.")

    assert constraint.active is False


def test_document_name_detector_does_not_treat_relevant_document_as_an_explicit_name() -> None:
    question = (
        "Can you tell me what the LLM guidance project is? Or like what is it about, what is its purpose? "
        "Use only the oldest version of the relevant document that is available to you."
    )

    constraint = detect_document_name_constraint(question)

    assert constraint.active is False


def test_document_name_detector_extracts_name_after_version_selector() -> None:
    constraint = detect_document_name_constraint(
        "Use only the oldest version of the Architecture Guide document to explain retries.",
    )

    assert constraint.names == ("Architecture Guide",)


def test_document_name_matching_normalizes_case_extension_and_punctuation() -> None:
    constraint = DocumentNameConstraint(
        names=("Realization_Draft5",),
        confidence=1.0,
        rationale="Test exact normalized document selection.",
        detector_name="test",
    )

    assert evidence_matches_constraints(
        _evidence("realization-draft5.PDF"),
        RetrievalConstraints(document=constraint),
    )
    assert not evidence_matches_constraints(
        _evidence("Realization_Draft4.pdf", title="Realization Draft 4"),
        RetrievalConstraints(document=constraint),
    )


async def test_missing_named_document_is_not_replaced_with_other_evidence() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(
        retriever=StaticRetriever([_evidence("Realization_Draft4.pdf", title="Realization Draft 4")]),
        llm_provider=llm,
    )

    state = await graph.run(
        QueryState(question="Only use the document Realization_Draft5 to explain metadata filtering."),
    )

    assert state.constraint_validation is not None
    assert state.constraint_validation.status is ConstraintValidationStatus.NO_MATCH
    assert state.retrieved_evidence == []
    assert state.citations == []
    assert llm.prompts == []
    assert state.answer is not None
    assert "No indexed evidence matched the requested metadata scope" in state.answer


async def test_matching_document_identity_is_passed_compactly_to_generation() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(
        retriever=StaticRetriever([_evidence("realization-draft5.pdf")]),
        llm_provider=llm,
    )

    state = await graph.run(
        QueryState(question="Only use the document Realization_Draft5 to explain metadata filtering."),
    )

    assert state.constraint_validation is not None
    assert state.constraint_validation.status is ConstraintValidationStatus.MATCHED
    assert len(llm.prompts) == 1
    assert "document name(s) 'Realization_Draft5'" in llm.prompts[0]
    assert "file=realization-draft5.pdf" in llm.prompts[0]
    assert "document_scope_match=true" in llm.prompts[0]


async def test_oldest_relevant_document_query_uses_relevance_without_fake_name_filter() -> None:
    llm = RecordingLLM()
    retriever = VersionAwareRetriever(
        StaticRetriever(
            [
                _evidence(
                    "llm-guidance-draft5.pdf",
                    title="LLM Guidance",
                    version_number=5,
                ),
                _evidence(
                    "llm-guidance-draft1.pdf",
                    title="LLM Guidance",
                    version_number=1,
                ),
            ],
        ),
    )
    graph = build_baseline_rag_graph(retriever=retriever, llm_provider=llm)
    question = (
        "Can you tell me what the LLM guidance project is? Or like what is it about, what is its purpose? "
        "Use only the oldest version of the relevant document that is available to you."
    )

    state = await graph.run(QueryState(question=question))

    assert state.query_classification is not None
    assert state.query_classification.document_constraint.active is False
    assert state.query_classification.version_constraint.mode.value == "oldest"
    assert state.constraint_validation is not None
    assert state.constraint_validation.status is ConstraintValidationStatus.MATCHED
    assert [item.metadata["document_version_number"] for item in state.retrieved_evidence] == [1]
    assert len(llm.prompts) == 1
    assert "the oldest document version" in llm.prompts[0]
    assert "document name(s)" not in llm.prompts[0]
    assert "version=v1" in llm.prompts[0]
