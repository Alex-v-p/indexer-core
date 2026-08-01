from __future__ import annotations

import uuid

from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution
from packages.rag_core.agents.information_need_graph.evidence import merge_information_need_evidence
from packages.rag_core.agents.query_graph.nodes.decompose import DecomposeInformationNeedsNode
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.document_scope import (
    CoverageMode,
    DocumentScope,
    ScopeResolutionSource,
    SubjectDocumentLane,
    SubjectScopeCatalogEntry,
    resolve_subject_scope,
)
from packages.rag_core.generation.citations import citation_from_evidence
from packages.rag_core.query_understanding.decomposition import (
    InformationNeed,
    InformationNeedDecomposition,
)
from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.subjects import SubjectKind


def test_comparison_intent_resolves_distinct_projects_but_alias_ambiguity_clarifies() -> None:
    daf = _project("DAF")
    internship = _project("Large Internship")
    duplicate_one = _project("One", aliases=("Shared",))
    duplicate_two = _project("Two", aliases=("Shared",))

    comparison = resolve_subject_scope(
        question="Compare DAF and Large Internship",
        requested_subject_ids=(),
        catalog=(daf, internship),
    )
    ambiguous = resolve_subject_scope(
        question="Compare Shared and DAF",
        requested_subject_ids=(),
        catalog=(daf, duplicate_one, duplicate_two),
    )

    assert comparison.source is ScopeResolutionSource.INFERRED_PROJECT
    assert comparison.comparison_requested is True
    assert [item.subject_id for item in comparison.lane_subjects] == [
        daf.subject_id,
        internship.subject_id,
    ]
    assert ambiguous.clarification_reason == "ambiguous_project_name"


async def test_model_omission_creates_one_strict_need_per_comparison_lane() -> None:
    first_document, second_document = uuid.uuid4(), uuid.uuid4()
    first = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope((first_document,))
    )
    second = SubjectDocumentLane(
        uuid.uuid4(),
        "Large Internship",
        DocumentScope.strict_scope((second_document,)),
    )
    state = QueryState(
        question="Compare DAF and Large Internship",
        document_scope=DocumentScope.strict_scope((first_document, second_document)),
        subject_lanes=(first, second),
        coverage_mode=CoverageMode.MULTI_DOCUMENT,
        comparison_requested=True,
    )

    await DecomposeInformationNeedsNode(_MergedDecomposer())(state)

    needs = state.information_need_decomposition.information_needs
    assert len(needs) == 2
    assert {need.subject_lane.subject_id for need in needs} == {
        first.subject_id,
        second.subject_id,
    }
    assert all(need.document_scope == need.subject_lane.document_scope for need in needs)
    assert all(need.coverage_mode is CoverageMode.MULTI_DOCUMENT for need in needs)


def test_lane_merge_rejects_other_project_and_citation_keeps_attribution() -> None:
    daf_document, internship_document = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope((daf_document,))
    )
    need = InformationNeed(
        "daf_need",
        "DAF status",
        "DAF status",
        document_scope=lane.document_scope,
        subject_lane=lane,
    )
    state = QueryState(
        question="Compare projects",
        document_scope=DocumentScope.strict_scope((daf_document, internship_document)),
    )
    state.information_need_executions = {
        need.need_id: InformationNeedExecution(information_need=need, max_attempts=1)
    }
    evidence = EvidenceItem(
        rank=1,
        text="DAF evidence",
        document_id=daf_document,
        subject_lane_id=lane.lane_id,
        subject_id=lane.subject_id,
        subject_name=lane.subject_name,
        metadata={
            "subject_lane": {
                "lane_id": lane.lane_id,
                "subject_id": str(lane.subject_id),
                "subject_name": lane.subject_name,
            }
        },
    )

    merge_information_need_evidence(
        state,
        [evidence, EvidenceItem(2, "Wrong side", document_id=internship_document)],
        information_need_id=need.need_id,
        attempt_number=1,
        query="DAF",
        max_items=5,
    )
    citation = citation_from_evidence(state.retrieved_evidence[0])

    assert [item.document_id for item in state.retrieved_evidence] == [daf_document]
    assert citation.subject_lane_id == lane.lane_id
    assert citation.subject_id == lane.subject_id
    assert citation.subject_name == "DAF"


class _MergedDecomposer:
    async def decompose(self, question: str) -> InformationNeedDecomposition:
        return InformationNeedDecomposition(
            information_needs=(
                InformationNeed("merged", "Compare delivery", "Compare both projects"),
            ),
            rationale="Model returned one merged comparison need.",
            decomposer_name="test",
        )


def _project(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> SubjectScopeCatalogEntry:
    return SubjectScopeCatalogEntry(
        subject_id=uuid.uuid4(),
        kind=SubjectKind.PROJECT,
        name=name,
        aliases=aliases,
    )
