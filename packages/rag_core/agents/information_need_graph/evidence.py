from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.retrieval.models import EvidenceItem


def evidence_key(item: EvidenceItem) -> str:
    if item.qdrant_chunk_index_id is not None:
        return f"qdrant_chunk:{item.qdrant_chunk_index_id}"
    normalized_text = " ".join(item.text.split()).lower()
    return "|".join(
        (
            "content",
            str(item.document_id) if item.document_id is not None else "",
            str(item.document_version_id) if item.document_version_id is not None else "",
            normalized_text,
        ),
    )


def merge_information_need_evidence(
    state: QueryState,
    evidence: list[EvidenceItem],
    *,
    information_need_id: str,
    attempt_number: int,
    query: str,
    max_items: int,
) -> tuple[tuple[str, ...], int]:
    """Merge one lookup into the query-wide evidence index with stable ranks."""

    if max_items <= 0:
        raise ValueError("max_items must be positive.")
    keys: list[str] = []
    unique_added = 0
    for item in evidence:
        key = evidence_key(item)
        existing = state.evidence_by_key.get(key)
        if existing is None:
            if len(state.evidence_by_key) >= max_items:
                continue
            aggregate_rank = max(
                state.next_evidence_rank,
                max((candidate.rank for candidate in state.evidence_by_key.values()), default=0) + 1,
            )
            existing = EvidenceItem(
                rank=aggregate_rank,
                text=item.text,
                score=item.score,
                qdrant_chunk_index_id=item.qdrant_chunk_index_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=dict(item.metadata),
            )
            state.next_evidence_rank = aggregate_rank + 1
            state.evidence_by_key[key] = existing
            unique_added += 1
        elif item.score is not None and (existing.score is None or item.score > existing.score):
            existing.score = item.score

        _annotate_lookup(
            existing,
            information_need_id=information_need_id,
            attempt_number=attempt_number,
            query=query,
        )
        if key not in keys:
            keys.append(key)

    _sync_retrieved_evidence(state)
    return tuple(keys), unique_added


def prune_information_need_evidence(
    state: QueryState,
    *,
    information_need_id: str,
    relevant_ranks: tuple[int, ...],
) -> None:
    """Retain only grader-approved evidence references for this information need."""

    execution = state.information_need_executions[information_need_id]
    relevant_rank_set = set(relevant_ranks)
    execution.evidence_keys = [
        key
        for key in execution.evidence_keys
        if state.evidence_by_key.get(key) is not None
        and state.evidence_by_key[key].rank in relevant_rank_set
    ]

    referenced = {
        key
        for item_execution in state.information_need_executions.values()
        for key in item_execution.evidence_keys
    }
    for key in tuple(state.evidence_by_key):
        if key not in referenced:
            del state.evidence_by_key[key]
    _sync_retrieved_evidence(state)


def evidence_for_information_need(state: QueryState, information_need_id: str) -> list[EvidenceItem]:
    execution = state.information_need_executions[information_need_id]
    return [
        state.evidence_by_key[key]
        for key in execution.evidence_keys
        if key in state.evidence_by_key
    ]


def _annotate_lookup(
    item: EvidenceItem,
    *,
    information_need_id: str,
    attempt_number: int,
    query: str,
) -> None:
    raw = item.metadata.setdefault("information_need_lookups", [])
    if not isinstance(raw, list):
        raw = []
        item.metadata["information_need_lookups"] = raw
    annotation = {
        "information_need_id": information_need_id,
        "attempt_number": attempt_number,
        "query": query,
    }
    if annotation not in raw:
        raw.append(annotation)


def _sync_retrieved_evidence(state: QueryState) -> None:
    state.retrieved_evidence = sorted(state.evidence_by_key.values(), key=lambda item: item.rank)
