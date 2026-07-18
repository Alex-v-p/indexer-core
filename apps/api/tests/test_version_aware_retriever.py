from __future__ import annotations

import uuid
from datetime import UTC, datetime

from packages.rag_core.documents import DocumentVersionConstraint, VersionSelectionMode
from packages.rag_core.query_understanding.temporal import (
    DateRange,
    DocumentDateConstraint,
    DocumentDateField,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers import VersionAwareRetriever


class StubRetriever:
    def __init__(self, evidence: list[EvidenceItem]) -> None:
        self.evidence = evidence
        self.calls: list[tuple[str, int, RetrievalConstraints | None]] = []

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        self.calls.append((question, top_k, constraints))
        return self.evidence[:top_k]


def _evidence(document_id: uuid.UUID, version: int, *, rank: int, score: float) -> EvidenceItem:
    return EvidenceItem(
        rank=rank,
        score=score,
        text=f"Version {version} evidence",
        document_id=document_id,
        document_version_id=uuid.uuid4(),
        metadata={
            "document_id": str(document_id),
            "document_version_number": version,
            "document_version_label": f"v{version}",
            "is_latest_version": version == 2,
        },
    )


async def test_ordinary_query_preserves_semantic_order_without_recency_bias() -> None:
    document_id = uuid.uuid4()
    underlying = StubRetriever(
        [
            _evidence(document_id, 1, rank=1, score=0.97),
            _evidence(document_id, 2, rank=2, score=0.81),
        ],
    )
    retriever = VersionAwareRetriever(underlying)

    batch = await retriever.retrieve_with_metadata("How does deployment work?", top_k=2)

    assert [item.metadata["document_version_number"] for item in batch.evidence] == [1, 2]
    assert underlying.calls[0][1] == 2
    assert batch.metadata["version_aware"]["recency_bias_applied"] is False
    assert batch.metadata["version_aware"]["constraint"]["mode"] == "all"


async def test_latest_query_keeps_only_latest_version_per_document() -> None:
    first_document = uuid.uuid4()
    second_document = uuid.uuid4()
    underlying = StubRetriever(
        [
            _evidence(first_document, 1, rank=1, score=0.99),
            _evidence(second_document, 1, rank=2, score=0.96),
            _evidence(first_document, 2, rank=3, score=0.91),
            _evidence(second_document, 2, rank=4, score=0.89),
        ],
    )
    retriever = VersionAwareRetriever(underlying, candidate_multiplier=4)

    evidence = await retriever.retrieve("Use the latest versions", top_k=4)

    assert [item.metadata["document_version_number"] for item in evidence] == [2, 2]
    assert [item.rank for item in evidence] == [1, 2]
    assert underlying.calls[0][1] == 16


async def test_specific_version_constraint_filters_without_reordering_matches() -> None:
    document_id = uuid.uuid4()
    underlying = StubRetriever(
        [
            _evidence(document_id, 1, rank=1, score=0.97),
            _evidence(document_id, 2, rank=2, score=0.90),
        ],
    )
    retriever = VersionAwareRetriever(underlying)
    constraints = RetrievalConstraints(
        version=DocumentVersionConstraint(
            mode=VersionSelectionMode.SPECIFIC,
            version_numbers=(1,),
            confidence=1.0,
            rationale="Test-specific constraint.",
            detector_name="test",
        ),
    )

    evidence = await retriever.retrieve("document contents", top_k=2, constraints=constraints)

    assert len(evidence) == 1
    assert evidence[0].metadata["document_version_number"] == 1
    assert evidence[0].rank == 1


async def test_temporal_constraint_excludes_missing_and_out_of_range_dates() -> None:
    document_id = uuid.uuid4()
    evidence = [
        _evidence(document_id, 1, rank=1, score=0.99),
        _evidence(document_id, 2, rank=2, score=0.98),
        _evidence(document_id, 2, rank=3, score=0.97),
    ]
    evidence[0].metadata["published_at_epoch"] = datetime(2024, 5, 1, tzinfo=UTC).timestamp()
    evidence[1].metadata["published_at_epoch"] = datetime(2025, 5, 1, tzinfo=UTC).timestamp()
    retriever = VersionAwareRetriever(StubRetriever(evidence))
    constraint = DocumentDateConstraint(
        field=DocumentDateField.PUBLISHED_AT,
        date_range=DateRange(
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        original_expression="2024",
        rationale="Test publication range.",
        detector_name="test",
    )

    results = await retriever.retrieve(
        "retry policy",
        top_k=3,
        constraints=RetrievalConstraints(dates=(constraint,)),
    )

    assert len(results) == 1
    assert results[0].metadata["published_at_epoch"] == datetime(2024, 5, 1, tzinfo=UTC).timestamp()


async def test_latest_with_date_constraint_selects_latest_inside_range() -> None:
    document_id = uuid.uuid4()
    evidence = [
        _evidence(document_id, 3, rank=1, score=0.99),
        _evidence(document_id, 2, rank=2, score=0.95),
        _evidence(document_id, 1, rank=3, score=0.90),
    ]
    evidence[0].metadata["published_at_epoch"] = datetime(2026, 2, 1, tzinfo=UTC).timestamp()
    evidence[1].metadata["published_at_epoch"] = datetime(2025, 6, 1, tzinfo=UTC).timestamp()
    evidence[2].metadata["published_at_epoch"] = datetime(2025, 1, 1, tzinfo=UTC).timestamp()
    constraints = RetrievalConstraints(
        version=DocumentVersionConstraint(
            mode=VersionSelectionMode.LATEST,
            confidence=1.0,
            rationale="Latest inside date range.",
            detector_name="test",
        ),
        dates=(
            DocumentDateConstraint(
                field=DocumentDateField.PUBLISHED_AT,
                date_range=DateRange(
                    start=datetime(2025, 1, 1, tzinfo=UTC),
                    end=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                original_expression="2025",
                rationale="Test publication range.",
                detector_name="test",
            ),
        ),
    )

    results = await VersionAwareRetriever(StubRetriever(evidence)).retrieve(
        "latest policy published in 2025",
        top_k=3,
        constraints=constraints,
    )

    assert [item.metadata["document_version_number"] for item in results] == [2]
