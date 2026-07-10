from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from packages.rag_core.evaluation.models import EvidenceExpectation
from packages.rag_core.retrieval.models import EvidenceItem


def evidence_matches(expectation: EvidenceExpectation, evidence: EvidenceItem) -> bool:
    """Return whether one retrieved chunk satisfies every configured matcher."""

    if (
        expectation.qdrant_chunk_index_id is not None
        and str(evidence.qdrant_chunk_index_id) != expectation.qdrant_chunk_index_id
    ):
        return False
    if expectation.document_id is not None and str(evidence.document_id) != expectation.document_id:
        return False
    if (
        expectation.document_version_id is not None
        and str(evidence.document_version_id) != expectation.document_version_id
    ):
        return False
    if expectation.metadata and not _is_subset(expectation.metadata, evidence.metadata):
        return False

    normalized_text = _normalize_text(evidence.text)
    if any(_normalize_text(fragment) not in normalized_text for fragment in expectation.text_contains):
        return False
    return True


def maximum_expectation_matches(
    expectations: Sequence[EvidenceExpectation],
    evidence: Sequence[EvidenceItem],
) -> dict[int, int]:
    """Find a maximum one-to-one expectation-to-evidence matching.

    The returned mapping uses expectation indexes as keys and evidence indexes
    as values. One-to-one matching prevents a single broad chunk from inflating
    recall by satisfying several identical annotations.
    """

    candidates = [
        [evidence_index for evidence_index, item in enumerate(evidence) if evidence_matches(expectation, item)]
        for expectation in expectations
    ]
    evidence_to_expectation: dict[int, int] = {}

    def assign(expectation_index: int, visited: set[int]) -> bool:
        for evidence_index in candidates[expectation_index]:
            if evidence_index in visited:
                continue
            visited.add(evidence_index)
            previous = evidence_to_expectation.get(evidence_index)
            if previous is None or assign(previous, visited):
                evidence_to_expectation[evidence_index] = expectation_index
                return True
        return False

    for expectation_index in range(len(expectations)):
        assign(expectation_index, set())

    return {expectation_index: evidence_index for evidence_index, expectation_index in evidence_to_expectation.items()}


def first_relevant_rank(expectations: Sequence[EvidenceExpectation], evidence: Sequence[EvidenceItem]) -> int | None:
    for item in sorted(evidence, key=lambda candidate: candidate.rank):
        if any(evidence_matches(expectation, item) for expectation in expectations):
            return item.rank
    return None


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _is_subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict) or not _is_subset(expected_value, actual_value):
                return False
        elif actual_value != expected_value:
            return False
    return True
