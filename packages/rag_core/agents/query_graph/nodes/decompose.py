from __future__ import annotations

from dataclasses import replace

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.document_scope import SubjectDocumentLane
from packages.rag_core.query_understanding.decomposition import InformationNeedDecomposer
from packages.rag_core.subjects.naming import normalize_subject_name

MAX_COMPARISON_INFORMATION_NEEDS = 12


class DecomposeInformationNeedsNode:
    """Graph node that extracts independently gradable answer requirements."""

    name = "decompose_information_needs"
    step_type = "query_decomposition"

    def __init__(self, decomposer: InformationNeedDecomposer) -> None:
        self._decomposer = decomposer

    async def __call__(self, state: QueryState) -> QueryState:
        decomposition = await self._decomposer.decompose(state.question)
        needs, lane_fallback_used = _bind_information_needs_to_lanes(
            decomposition.information_needs,
            lanes=state.subject_lanes,
            document_scope=state.document_scope,
            coverage_mode=state.coverage_mode,
            comparison_requested=state.comparison_requested,
        )
        decomposition = replace(
            decomposition,
            information_needs=needs,
            fallback_used=decomposition.fallback_used or lane_fallback_used,
        )
        state.information_need_decomposition = decomposition
        state.metadata["information_need_decomposition"] = decomposition.to_metadata()
        return state


def _bind_information_needs_to_lanes(
    information_needs,
    *,
    lanes: tuple[SubjectDocumentLane, ...],
    document_scope,
    coverage_mode,
    comparison_requested: bool,
):
    if not lanes:
        return (
            tuple(
                replace(
                    need,
                    document_scope=document_scope,
                    coverage_mode=coverage_mode,
                )
                for need in information_needs
            ),
            False,
        )
    if len(lanes) == 1:
        lane = lanes[0]
        return (
            tuple(
                replace(
                    need,
                    document_scope=lane.document_scope,
                    subject_lane=lane,
                    coverage_mode=coverage_mode,
                )
                for need in information_needs
            ),
            False,
        )
    if not comparison_requested:
        raise RuntimeError("Multiple subject lanes require explicit comparison intent.")

    by_lane: dict[str, list] = {lane.lane_id: [] for lane in lanes}
    fallback_used = False
    for need in information_needs:
        text = normalize_subject_name(
            " ".join((need.description, need.retrieval_query, need.subject_context))
        )
        matches = tuple(
            lane
            for lane in lanes
            if f" {normalize_subject_name(lane.subject_name)} " in f" {text} "
        )
        targets = matches if len(matches) == 1 else lanes
        fallback_used = fallback_used or len(matches) != 1
        for lane in targets:
            by_lane[lane.lane_id].append(_lane_need(need, lane, coverage_mode))

    for lane in lanes:
        if by_lane[lane.lane_id]:
            continue
        fallback_used = True
        by_lane[lane.lane_id].append(
            _lane_need(information_needs[0], lane, coverage_mode, fallback=True)
        )

    bounded: list = []
    offsets = {lane.lane_id: 0 for lane in lanes}
    while len(bounded) < MAX_COMPARISON_INFORMATION_NEEDS:
        added = False
        for lane in lanes:
            items = by_lane[lane.lane_id]
            offset = offsets[lane.lane_id]
            if offset >= len(items):
                continue
            bounded.append(items[offset])
            offsets[lane.lane_id] += 1
            added = True
            if len(bounded) >= MAX_COMPARISON_INFORMATION_NEEDS:
                break
        if not added:
            break
    return tuple(bounded), fallback_used


def _lane_need(need, lane: SubjectDocumentLane, coverage_mode, *, fallback: bool = False):
    suffix = lane.subject_id.hex[:8]
    label = f"Project: {lane.subject_name}"
    retrieval_query = need.retrieval_query
    if normalize_subject_name(lane.subject_name) not in normalize_subject_name(retrieval_query):
        retrieval_query = f"{retrieval_query} {lane.subject_name}"
    return replace(
        need,
        need_id=f"{need.need_id}__lane_{suffix}",
        description=f"{need.description} [{label}]",
        retrieval_query=retrieval_query,
        subject_context="; ".join(item for item in (need.subject_context, label) if item),
        document_scope=lane.document_scope,
        subject_lane=lane,
        coverage_mode=coverage_mode,
    )
