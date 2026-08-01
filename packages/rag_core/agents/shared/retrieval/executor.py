from __future__ import annotations

from dataclasses import replace
from math import ceil
from types import MappingProxyType
from typing import Mapping

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.document_scope import CoverageMode, DocumentScope
from packages.rag_core.agents.shared.retrieval.models import RetrievalPlanExecution
from packages.rag_core.agents.shared.retrieval.nodes.rerank import RerankNode
from packages.rag_core.agents.shared.retrieval.nodes.retrieve import RetrieveNode
from packages.rag_core.query_understanding.planning import RetrievalPlan
from packages.rag_core.retrieval.document_selection import (
    DocumentCandidateSelector,
    PassthroughDocumentCandidateSelector,
)
from packages.rag_core.retrieval.models import EvidenceItem


class RetrievalPlanExecutor:
    """Dispatch a typed plan, then apply document-aware candidate selection."""

    def __init__(
        self,
        executions: Mapping[str, RetrievalPlanExecution],
        *,
        candidate_selector: DocumentCandidateSelector | None = None,
        multi_document_fanout: int = 4,
        best_evidence_document_max_share: float = 0.6,
    ) -> None:
        if not executions:
            raise ValueError("At least one retrieval plan execution must be configured.")
        normalized: dict[str, RetrievalPlanExecution] = {}
        for pipeline_name, execution in executions.items():
            if pipeline_name != execution.pipeline_name:
                raise ValueError("Execution mapping keys must match execution.pipeline_name.")
            if pipeline_name in normalized:
                raise ValueError(f"Duplicate retrieval execution for {pipeline_name!r}.")
            normalized[pipeline_name] = execution
        self._executions = MappingProxyType(normalized)
        self._candidate_selector = candidate_selector or PassthroughDocumentCandidateSelector()
        if multi_document_fanout <= 0:
            raise ValueError("multi_document_fanout must be positive.")
        if not 0.0 < best_evidence_document_max_share <= 1.0:
            raise ValueError("best_evidence_document_max_share must be in (0, 1].")
        self._multi_document_fanout = multi_document_fanout
        self._best_evidence_document_max_share = best_evidence_document_max_share

    @property
    def available_pipeline_names(self) -> tuple[str, ...]:
        return tuple(self._executions)

    async def execute_lookup(
        self,
        *,
        plan: RetrievalPlan,
        query: str,
        top_k: int,
    ) -> tuple[list[EvidenceItem], dict[str, object]]:
        """Execute an isolated lookup without overwriting the parent QueryState."""

        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        execution = self._resolve_execution(plan)
        candidate_multiplier = max(
            execution.candidate_multiplier,
            self._candidate_selector.candidate_multiplier,
        )
        max_candidates = _lowest_candidate_limit(
            execution.max_candidates,
            self._candidate_selector.max_candidates,
        )
        coverage, lookup_state = await self._retrieve_for_coverage(
            execution=execution,
            plan=plan,
            query=query,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            max_candidates=max_candidates,
        )
        if execution.reranker is not None and lookup_state.retrieved_evidence:
            lookup_state.active_retrieval_top_k = len(lookup_state.retrieved_evidence)
            lookup_state = await RerankNode(execution.reranker)(lookup_state)
            lookup_state.active_retrieval_top_k = top_k
            reranking = lookup_state.metadata.get("reranking")
            if isinstance(reranking, dict):
                reranking["final_requested_top_k"] = top_k
                reranking["document_balancing_after_rerank"] = True

        candidates = lookup_state.retrieved_evidence
        if coverage["effective_mode"] == CoverageMode.BEST_EVIDENCE.value:
            candidates = _soft_document_cap_order(
                candidates,
                top_k=top_k,
                max_share=self._best_evidence_document_max_share,
            )
        selection = self._candidate_selector.select(
            candidates,
            top_k=top_k,
            preference=plan.preferred_document,
        )
        lookup_state.retrieved_evidence = [
            _with_lane_provenance(item, plan) for item in selection.evidence
        ]
        contributing_ids = tuple(
            dict.fromkeys(
                str(item.document_id)
                for item in lookup_state.retrieved_evidence
                if item.document_id is not None
            )
        )
        coverage["contributing_document_count"] = len(contributing_ids)
        coverage["contributing_document_ids"] = list(contributing_ids[:20])
        coverage["coverage_satisfied"] = None
        lookup_state.metadata["document_balancing"] = selection.to_metadata()
        metadata: dict[str, object] = {
            "selected_pipeline_name": execution.pipeline_name,
            "selected_pipeline_version": execution.pipeline_version,
            "strategy": plan.strategy.value,
            "query": query,
            "top_k": top_k,
            "reranking_applied": execution.reranker is not None,
            "retrieved_count": len(lookup_state.retrieved_evidence),
            "retrieval": lookup_state.metadata.get("retrieval", {}),
            "document_balancing": selection.to_metadata(),
            "preferred_document": (
                plan.preferred_document.to_metadata() if plan.preferred_document is not None else None
            ),
            "subject_lane": plan.subject_lane.to_metadata() if plan.subject_lane is not None else None,
            "coverage": coverage,
        }
        if "reranking" in lookup_state.metadata:
            metadata["reranking"] = lookup_state.metadata["reranking"]
        return lookup_state.retrieved_evidence, metadata

    async def _retrieve_for_coverage(
        self,
        *,
        execution: RetrievalPlanExecution,
        plan: RetrievalPlan,
        query: str,
        top_k: int,
        candidate_multiplier: int,
        max_candidates: int | None,
    ) -> tuple[dict[str, object], QueryState]:
        allowed_ids = plan.document_scope.allowed_document_ids
        effective_mode = plan.coverage_mode
        fallback_reason: str | None = None
        searched_ids: tuple = ()
        states: list[QueryState] = []

        if plan.coverage_mode is CoverageMode.MULTI_DOCUMENT:
            if plan.document_scope.is_global:
                effective_mode = CoverageMode.BEST_EVIDENCE
                fallback_reason = "global_scope_uses_best_evidence"
            elif not allowed_ids:
                searched_ids = ()
            else:
                searched_ids = tuple(sorted(allowed_ids, key=str))[: self._multi_document_fanout]
                if len(allowed_ids) > len(searched_ids):
                    fallback_reason = "multi_document_fanout_truncated"

        if effective_mode is CoverageMode.MULTI_DOCUMENT and searched_ids:
            for document_id in searched_ids:
                per_document_plan = replace(
                    plan,
                    document_scope=DocumentScope.strict_scope((document_id,)),
                    subject_lane=None,
                )
                states.append(
                    await _retrieve_once(
                        execution=execution,
                        plan=per_document_plan,
                        query=query,
                        top_k=top_k,
                        candidate_multiplier=candidate_multiplier,
                        max_candidates=max_candidates,
                    )
                )
            lookup_state = states[0]
            lookup_state.retrieved_evidence = _round_robin_evidence(
                [state.retrieved_evidence for state in states]
            )
            lookup_state.metadata["retrieval"] = {
                "strategy": "multi_document_per_document",
                "requested_top_k": top_k,
                "candidate_top_k": sum(
                    int(state.metadata.get("retrieval", {}).get("candidate_top_k", top_k))
                    for state in states
                    if isinstance(state.metadata.get("retrieval"), dict)
                ),
                "per_document_result_counts": {
                    str(document_id): len(state.retrieved_evidence)
                    for document_id, state in zip(searched_ids, states, strict=True)
                },
                "out_of_scope_rejected_count": sum(
                    int(state.metadata.get("retrieval", {}).get("out_of_scope_rejected_count", 0))
                    for state in states
                    if isinstance(state.metadata.get("retrieval"), dict)
                ),
            }
        else:
            lookup_state = await _retrieve_once(
                execution=execution,
                plan=plan,
                query=query,
                top_k=top_k,
                candidate_multiplier=candidate_multiplier,
                max_candidates=max_candidates,
            )
            if plan.document_scope.strict:
                searched_ids = allowed_ids

        raw_retrieval = lookup_state.metadata.get("retrieval")
        rejected_count = (
            int(raw_retrieval.get("out_of_scope_rejected_count", 0))
            if isinstance(raw_retrieval, dict)
            else 0
        )
        coverage: dict[str, object] = {
            "requested_mode": plan.coverage_mode.value,
            "effective_mode": effective_mode.value,
            "fallback_reason": fallback_reason,
            "lane_subject_id": (
                str(plan.subject_lane.subject_id) if plan.subject_lane is not None else None
            ),
            "lane_subject_name": (
                plan.subject_lane.subject_name if plan.subject_lane is not None else None
            ),
            "allowed_document_count": len(allowed_ids),
            "allowed_document_ids": [str(item) for item in allowed_ids[:20]],
            "searched_document_count": len(searched_ids),
            "searched_document_ids": [str(item) for item in searched_ids[:20]],
            "out_of_scope_rejected_count": rejected_count,
        }
        return coverage, lookup_state

    async def execute(self, state: QueryState) -> QueryState:
        plan = state.effective_retrieval_plan
        if plan is None:
            raise RuntimeError("Retrieval plan execution requires a plan first.")
        if len(state.subject_lanes) > 1:
            raise RuntimeError(
                "Multi-project comparisons require lane-bound information-need execution."
            )
        lane = state.subject_lanes[0] if state.subject_lanes else None
        effective_scope = lane.document_scope if lane is not None else state.document_scope
        evidence, metadata = await self.execute_lookup(
            plan=replace(
                plan,
                document_scope=effective_scope,
                subject_lane=lane,
                coverage_mode=state.coverage_mode,
            ),
            query=state.effective_retrieval_query,
            top_k=state.effective_retrieval_top_k,
        )
        state.retrieved_evidence = evidence
        state.metadata["retrieval_plan_execution"] = metadata
        return state

    def _resolve_execution(self, plan: RetrievalPlan) -> RetrievalPlanExecution:
        selected_name = plan.selected_pipeline_name
        try:
            execution = self._executions[selected_name]
        except KeyError as exc:
            available = ", ".join(sorted(self._executions))
            raise RuntimeError(
                f"Retrieval plan selected unavailable pipeline {selected_name!r}. Available: {available}.",
            ) from exc
        if execution.strategy is not plan.strategy:
            raise RuntimeError(
                f"Retrieval execution for {selected_name!r} is configured as {execution.strategy.value!r}, "
                f"but the plan selected {plan.strategy.value!r}.",
            )
        if (execution.reranker is not None) is not plan.requires_reranking:
            raise RuntimeError(
                f"Retrieval execution for {selected_name!r} does not match the plan's reranking requirement.",
            )
        return execution


def _lowest_candidate_limit(left: int | None, right: int | None) -> int | None:
    values = tuple(value for value in (left, right) if value is not None)
    return min(values) if values else None


async def _retrieve_once(
    *,
    execution: RetrievalPlanExecution,
    plan: RetrievalPlan,
    query: str,
    top_k: int,
    candidate_multiplier: int,
    max_candidates: int | None,
) -> QueryState:
    state = QueryState(
        question=query,
        top_k=top_k,
        document_scope=plan.document_scope,
        retrieval_plan=plan,
        active_retrieval_plan=plan,
        active_retrieval_query=query,
        active_retrieval_top_k=top_k,
    )
    return await RetrieveNode(
        execution.retriever,
        candidate_multiplier=candidate_multiplier,
        max_candidates=max_candidates,
    )(state)


def _round_robin_evidence(groups: list[list[EvidenceItem]]) -> list[EvidenceItem]:
    merged: list[EvidenceItem] = []
    seen: set[str] = set()
    offset = 0
    while True:
        added = False
        for group in groups:
            if offset >= len(group):
                continue
            item = group[offset]
            key = _evidence_identity(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)
            added = True
        if not added:
            break
        offset += 1
    return [replace(item, rank=index) for index, item in enumerate(merged, start=1)]


def _soft_document_cap_order(
    evidence: list[EvidenceItem],
    *,
    top_k: int,
    max_share: float,
) -> list[EvidenceItem]:
    ordered = sorted(evidence, key=lambda item: item.rank)
    cap = max(1, ceil(top_k * max_share))
    counts: dict[str, int] = {}
    preferred: list[EvidenceItem] = []
    deferred: list[EvidenceItem] = []
    for item in ordered:
        key = _document_identity(item)
        if counts.get(key, 0) < cap:
            counts[key] = counts.get(key, 0) + 1
            preferred.append(item)
        else:
            deferred.append(item)
    return [
        replace(item, rank=index)
        for index, item in enumerate((*preferred, *deferred), start=1)
    ]


def _with_lane_provenance(item: EvidenceItem, plan: RetrievalPlan) -> EvidenceItem:
    lane = plan.subject_lane
    if lane is None:
        return item
    metadata = dict(item.metadata)
    metadata["subject_lane"] = {
        "lane_id": lane.lane_id,
        "subject_id": str(lane.subject_id),
        "subject_name": lane.subject_name,
    }
    metadata["coverage_mode"] = plan.coverage_mode.value
    return replace(
        item,
        subject_lane_id=lane.lane_id,
        subject_id=lane.subject_id,
        subject_name=lane.subject_name,
        metadata=metadata,
    )


def _document_identity(item: EvidenceItem) -> str:
    if item.document_id is not None:
        return f"document:{item.document_id}"
    if item.document_version_id is not None:
        return f"version:{item.document_version_id}"
    return f"unknown:{item.rank}"


def _evidence_identity(item: EvidenceItem) -> str:
    if item.qdrant_chunk_index_id is not None:
        return f"chunk:{item.qdrant_chunk_index_id}"
    return "|".join(
        (
            _document_identity(item),
            " ".join(item.text.split()).casefold(),
        )
    )
