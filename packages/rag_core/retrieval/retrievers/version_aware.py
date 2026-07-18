from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from packages.rag_core.documents import DocumentVersionConstraint, VersionSelectionMode
from packages.rag_core.query_understanding.versioning import detect_document_version_constraint
from packages.rag_core.query_understanding.temporal import (
    DocumentDateConstraint,
    DocumentDateField,
    detect_document_date_constraints,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import RetrievalBatch, Retriever, retrieve_batch_compatibly


class VersionAwareRetriever:
    """Apply explicit version and temporal semantics around a retrieval pipeline.

    Semantic ordering is preserved. Recency is never used as a general score
    boost; metadata is filtered only when the query explicitly requires it.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        candidate_multiplier: int = 4,
        max_candidates: int = 100,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        self._retriever = retriever
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates = max_candidates

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        return (await self.retrieve_with_metadata(question, top_k=top_k, constraints=constraints)).evidence

    async def retrieve_with_metadata(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> RetrievalBatch:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        effective = constraints or RetrievalConstraints(
            version=detect_document_version_constraint(question),
            dates=detect_document_date_constraints(question),
        )
        if not effective.active:
            batch = await retrieve_batch_compatibly(
                self._retriever,
                question,
                top_k=top_k,
                constraints=effective,
            )
            batch.metadata = {
                **batch.metadata,
                "version_aware": {
                    "constraint": effective.version.to_metadata(),
                    "constraints": effective.to_metadata(),
                    "candidate_count": len(batch.evidence),
                    "result_count": len(batch.evidence),
                    "recency_bias_applied": False,
                },
            }
            return batch

        candidate_k = max(top_k, min(top_k * self._candidate_multiplier, self._max_candidates))
        batch = await retrieve_batch_compatibly(
            self._retriever,
            question,
            top_k=candidate_k,
            constraints=effective,
        )
        selected = _apply_constraints(batch.evidence, effective)
        selected = [_with_rank(item, rank) for rank, item in enumerate(selected[:top_k], start=1)]
        return RetrievalBatch(
            evidence=selected,
            metadata={
                **batch.metadata,
                "version_aware": {
                    "constraint": effective.version.to_metadata(),
                    "constraints": effective.to_metadata(),
                    "candidate_top_k": candidate_k,
                    "candidate_count": len(batch.evidence),
                    "result_count": len(selected),
                    "recency_bias_applied": False,
                    "selection_applied": True,
                },
            },
        )


def _apply_constraints(
    evidence: list[EvidenceItem],
    constraints: RetrievalConstraints,
) -> list[EvidenceItem]:
    selected = [item for item in evidence if _matches_date_constraints(item, constraints.dates)]
    constraint = constraints.version
    mode = constraint.mode
    if mode is VersionSelectionMode.ALL:
        return selected
    if mode is VersionSelectionMode.SPECIFIC:
        allowed = set(constraint.version_numbers)
        return [item for item in selected if _version_number(item) in allowed]
    if mode is VersionSelectionMode.LATEST:
        return _keep_version_count_per_document(selected, count=1)
    if mode is VersionSelectionMode.PREVIOUS:
        return _keep_version_position_per_document(selected, position=2)
    if mode is VersionSelectionMode.LATEST_AND_PREVIOUS:
        return _keep_version_count_per_document(selected, count=2)
    return selected


def _matches_date_constraints(
    item: EvidenceItem,
    constraints: tuple[DocumentDateConstraint, ...],
) -> bool:
    for constraint in constraints:
        key = (
            "uploaded_at_epoch"
            if constraint.field is DocumentDateField.UPLOADED_AT
            else "published_at_epoch"
        )
        raw_value = item.metadata.get(key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False
        if constraint.date_range.start is not None and value < constraint.date_range.start.timestamp():
            return False
        if constraint.date_range.end is not None and value >= constraint.date_range.end.timestamp():
            return False
    return True


def _keep_version_position_per_document(
    evidence: list[EvidenceItem],
    *,
    position: int,
) -> list[EvidenceItem]:
    if position <= 0:
        raise ValueError("position must be positive.")
    versions_by_document: dict[str, set[int]] = defaultdict(set)
    for item in evidence:
        number = _version_number(item)
        if number is not None:
            versions_by_document[_document_key(item)].add(number)

    selected_versions = {}
    for key, numbers in versions_by_document.items():
        ordered = sorted(numbers, reverse=True)
        if len(ordered) >= position:
            selected_versions[key] = ordered[position - 1]

    return [
        item
        for item in evidence
        if (
            _version_number(item) is None
            or _version_number(item) == selected_versions.get(_document_key(item))
        )
    ]


def _keep_version_count_per_document(
    evidence: list[EvidenceItem],
    *,
    count: int,
) -> list[EvidenceItem]:
    versions_by_document: dict[str, set[int]] = defaultdict(set)
    for item in evidence:
        number = _version_number(item)
        if number is not None:
            versions_by_document[_document_key(item)].add(number)

    selected_versions = {
        key: set(sorted(numbers, reverse=True)[:count])
        for key, numbers in versions_by_document.items()
    }
    selected: list[EvidenceItem] = []
    for item in evidence:
        number = _version_number(item)
        if number is None:
            selected.append(item)
            continue
        if number in selected_versions.get(_document_key(item), set()):
            selected.append(item)
    return selected


def _document_key(item: EvidenceItem) -> str:
    if item.document_id is not None:
        return str(item.document_id)
    value = item.metadata.get("document_id") or item.metadata.get("document_title") or item.metadata.get("original_filename")
    return str(value or "unknown-document")


def _version_number(item: EvidenceItem) -> int | None:
    value: Any = item.metadata.get("document_version_number")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _with_rank(item: EvidenceItem, rank: int) -> EvidenceItem:
    metadata = dict(item.metadata)
    metadata["constraint_selection"] = {
        "selected": True,
        "document_version_number": _version_number(item),
    }
    return replace(item, rank=rank, metadata=metadata)
