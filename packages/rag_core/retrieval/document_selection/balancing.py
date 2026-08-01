from __future__ import annotations

import math
from collections import Counter

from packages.rag_core.documents.preferences import (
    DocumentPreference,
    document_reference_from_values,
)
from packages.rag_core.retrieval.document_selection.models import DocumentCandidateSelection
from packages.rag_core.retrieval.models import EvidenceItem


class DocumentBalancedCandidateSelector:
    """Apply soft primary preference and document quotas to a broad candidate pool."""

    name = "document_balanced_candidate_selector"

    def __init__(
        self,
        *,
        candidate_multiplier: int = 3,
        max_candidates: int | None = 60,
        primary_min_share: float = 0.6,
        primary_max_share: float = 0.8,
        secondary_max_share: float = 0.4,
        unpreferred_max_share: float = 0.6,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided.")
        for name, value in (
            ("primary_min_share", primary_min_share),
            ("primary_max_share", primary_max_share),
            ("secondary_max_share", secondary_max_share),
            ("unpreferred_max_share", unpreferred_max_share),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be greater than 0 and at most 1.")
        if primary_min_share > primary_max_share:
            raise ValueError("primary_min_share cannot exceed primary_max_share.")
        self._candidate_multiplier = candidate_multiplier
        self._max_candidates = max_candidates
        self._primary_min_share = primary_min_share
        self._primary_max_share = primary_max_share
        self._secondary_max_share = secondary_max_share
        self._unpreferred_max_share = unpreferred_max_share

    @property
    def candidate_multiplier(self) -> int:
        return self._candidate_multiplier

    @property
    def max_candidates(self) -> int | None:
        return self._max_candidates

    def select(
        self,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
        preference: DocumentPreference | None,
    ) -> DocumentCandidateSelection:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        ordered = sorted(evidence, key=lambda item: item.rank)
        keys = {item.rank: _document_key(item) for item in ordered}
        selected: list[EvidenceItem] = []
        selected_ranks: set[int] = set()
        counts: Counter[str] = Counter()
        quota_relaxed = False
        primary_key = preference.document.key if preference is not None else None
        primary_candidate_ranks: set[int] = set()
        primary_quota: int | None = None

        if preference is not None:
            primary_candidates = [item for item in ordered if preference.matches(
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                metadata=item.metadata,
            )]
            primary_candidate_ranks = {item.rank for item in primary_candidates}
            primary_min = min(top_k, max(1, math.ceil(top_k * self._primary_min_share)))
            primary_quota = primary_min
            primary_max = min(top_k, max(primary_min, math.ceil(top_k * self._primary_max_share)))
            secondary_max = max(1, math.ceil(top_k * self._secondary_max_share))
            self._append_candidates(
                selected,
                selected_ranks,
                counts,
                primary_candidates,
                limit=primary_min,
                per_document_limit=primary_max,
                key_by_rank=keys,
            )
            self._append_candidates(
                selected,
                selected_ranks,
                counts,
                ordered,
                limit=top_k,
                per_document_limit=secondary_max,
                key_by_rank=keys,
                skip_ranks=primary_candidate_ranks,
            )
            self._append_candidates(
                selected,
                selected_ranks,
                counts,
                primary_candidates,
                limit=top_k,
                per_document_limit=primary_max,
                key_by_rank=keys,
            )
        else:
            per_document_limit = max(1, math.ceil(top_k * self._unpreferred_max_share))
            self._append_candidates(
                selected,
                selected_ranks,
                counts,
                ordered,
                limit=top_k,
                per_document_limit=per_document_limit,
                key_by_rank=keys,
            )

        if len(selected) < min(top_k, len(ordered)):
            quota_relaxed = True
            self._append_candidates(
                selected,
                selected_ranks,
                counts,
                ordered,
                limit=top_k,
                per_document_limit=None,
                key_by_rank=keys,
            )

        reranked = tuple(
            _copy_with_balancing_metadata(
                item,
                rank=index,
                document_key=keys[item.rank],
                primary=preference is not None and preference.matches(
                    document_id=item.document_id,
                    document_version_id=item.document_version_id,
                    metadata=item.metadata,
                ),
                selector_name=self.name,
            )
            for index, item in enumerate(selected[:top_k], start=1)
        )
        selected_counts = Counter(
            str(item.metadata["document_balancing"]["document_key"])
            for item in reranked
        )
        return DocumentCandidateSelection(
            evidence=reranked,
            candidate_count=len(ordered),
            requested_top_k=top_k,
            selected_document_counts=tuple(sorted(selected_counts.items())),
            primary_document_key=primary_key,
            primary_selected_count=sum(
                bool(item.metadata["document_balancing"]["primary_document"]) for item in reranked
            ),
            quota_relaxed=quota_relaxed,
            selector_name=self.name,
            primary_document_quota=primary_quota,
        )

    @staticmethod
    def _append_candidates(
        selected: list[EvidenceItem],
        selected_ranks: set[int],
        counts: Counter[str],
        candidates: list[EvidenceItem],
        *,
        limit: int,
        per_document_limit: int | None,
        key_by_rank: dict[int, str],
        skip_ranks: set[int] | None = None,
    ) -> None:
        for item in candidates:
            if len(selected) >= limit:
                return
            if item.rank in selected_ranks:
                continue
            key = key_by_rank[item.rank]
            if skip_ranks is not None and item.rank in skip_ranks:
                continue
            if per_document_limit is not None and counts[key] >= per_document_limit:
                continue
            selected.append(item)
            selected_ranks.add(item.rank)
            counts[key] += 1


class PassthroughDocumentCandidateSelector:
    """Compatibility selector that only truncates and normalizes ranks."""

    name = "passthrough_document_candidate_selector"

    @property
    def candidate_multiplier(self) -> int:
        return 1

    @property
    def max_candidates(self) -> int | None:
        return None

    def select(
        self,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
        preference: DocumentPreference | None,
    ) -> DocumentCandidateSelection:
        del preference
        selected = tuple(
            _copy_with_balancing_metadata(
                item,
                rank=index,
                document_key=_document_key(item),
                primary=False,
                selector_name=self.name,
            )
            for index, item in enumerate(sorted(evidence, key=lambda candidate: candidate.rank)[:top_k], start=1)
        )
        counts = Counter(
            str(item.metadata["document_balancing"]["document_key"])
            for item in selected
        )
        return DocumentCandidateSelection(
            evidence=selected,
            candidate_count=len(evidence),
            requested_top_k=top_k,
            selected_document_counts=tuple(sorted(counts.items())),
            primary_document_key=None,
            primary_selected_count=0,
            quota_relaxed=False,
            selector_name=self.name,
            primary_document_quota=None,
        )


def _document_key(item: EvidenceItem) -> str:
    return document_reference_from_values(
        rank=item.rank,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata=item.metadata,
    ).key


def _copy_with_balancing_metadata(
    item: EvidenceItem,
    *,
    rank: int,
    document_key: str,
    primary: bool,
    selector_name: str,
) -> EvidenceItem:
    metadata = dict(item.metadata)
    metadata["document_balancing"] = {
        "selector": selector_name,
        "original_rank": item.rank,
        "document_key": document_key,
        "primary_document": primary,
    }
    return EvidenceItem(
        rank=rank,
        text=item.text,
        score=item.score,
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        subject_lane_id=item.subject_lane_id,
        subject_id=item.subject_id,
        subject_name=item.subject_name,
        metadata=metadata,
    )
