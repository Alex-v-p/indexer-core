from __future__ import annotations

import uuid

import pytest

from packages.rag_core.document_scope import (
    DocumentScope,
    ScopeResolutionSource,
    SubjectScopeCatalogEntry,
    resolve_subject_scope,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints
from packages.rag_core.retrieval.retrievers.base import (
    UnsupportedStrictDocumentScopeError,
    retrieve_compatibly,
)
from packages.rag_core.subjects import SubjectKind


def test_subjectless_query_remains_global_but_strict_empty_is_distinct() -> None:
    result = resolve_subject_scope(
        question="How does retrieval work?",
        requested_subject_ids=(),
        catalog=(),
    )

    assert result.source is ScopeResolutionSource.GLOBAL
    assert result.hard_subject_ids == ()
    assert DocumentScope.global_scope().is_global is True
    assert DocumentScope.strict_scope(()).is_strict_empty is True


def test_explicit_topic_and_inferred_project_are_both_hard_filters() -> None:
    project = _subject("DAF", SubjectKind.PROJECT, aliases=("Data Access Framework",))
    topic = _subject("Architecture", SubjectKind.TOPIC)

    result = resolve_subject_scope(
        question="Explain architecture in the DAF rollout",
        requested_subject_ids=(topic.subject_id,),
        catalog=(project, topic),
    )

    assert result.source is ScopeResolutionSource.EXPLICIT_AND_INFERRED
    assert result.hard_subject_ids == (topic.subject_id, project.subject_id)


def test_inferred_non_project_subject_is_not_a_hard_scope() -> None:
    topic = _subject("Architecture", SubjectKind.TOPIC)

    result = resolve_subject_scope(
        question="Explain Architecture",
        requested_subject_ids=(),
        catalog=(topic,),
    )

    assert result.source is ScopeResolutionSource.GLOBAL
    assert result.hard_subject_ids == ()


def test_ambiguous_alias_and_unknown_named_project_require_clarification() -> None:
    first = _subject("DAF One", SubjectKind.PROJECT, aliases=("DAF",))
    second = _subject("DAF Two", SubjectKind.PROJECT, aliases=("DAF",))

    ambiguous = resolve_subject_scope(
        question="What changed in DAF?",
        requested_subject_ids=(),
        catalog=(first, second),
    )
    unknown = resolve_subject_scope(
        question="What changed in Project Zephyr?",
        requested_subject_ids=(),
        catalog=(first,),
    )

    assert ambiguous.clarification_reason == "ambiguous_project_name"
    assert {item.subject_id for item in ambiguous.ambiguity_candidates} == {
        first.subject_id,
        second.subject_id,
    }
    assert unknown.clarification_reason == "unknown_named_project"


@pytest.mark.parametrize(
    "question",
    (
        "Is there information about the DAF project?",
        'Is there information about the "DAF" project?',
        "Is there information about the Atlas Migration project?",
        'Is there information about project "DAF"?',
    ),
)
def test_unknown_postfix_quoted_titled_and_acronym_projects_require_clarification(
    question: str,
) -> None:
    result = resolve_subject_scope(
        question=question,
        requested_subject_ids=(),
        catalog=(_subject("Known Project", SubjectKind.PROJECT),),
    )

    assert result.source is ScopeResolutionSource.INFERRED_PROJECT
    assert result.clarification_reason == "unknown_named_project"


@pytest.mark.parametrize(
    "question",
    (
        "What is the project timeline?",
        "What is the project status?",
        "What is the Project Timeline?",
        "Summarize Software Project planning guidance.",
        "Summarize general project planning guidance.",
    ),
)
def test_generic_project_phrases_do_not_create_a_hard_scope(question: str) -> None:
    result = resolve_subject_scope(
        question=question,
        requested_subject_ids=(),
        catalog=(_subject("DAF", SubjectKind.PROJECT),),
    )

    assert result.source is ScopeResolutionSource.GLOBAL
    assert result.clarification_required is False


async def test_daf_scope_rejects_large_internship_evidence_before_merge() -> None:
    daf_document_id = uuid.uuid4()
    internship_document_id = uuid.uuid4()

    class LeakyRetriever:
        async def retrieve(self, question, *, top_k, constraints=None):
            return [
                EvidenceItem(rank=1, text="Large internship", document_id=internship_document_id),
                EvidenceItem(rank=2, text="DAF", document_id=daf_document_id),
            ]

    evidence = await retrieve_compatibly(
        LeakyRetriever(),
        "DAF",
        top_k=5,
        constraints=RetrievalConstraints(
            document_scope=DocumentScope.strict_scope((daf_document_id,)),
        ),
    )

    assert [item.document_id for item in evidence] == [daf_document_id]


async def test_strict_scope_fails_closed_for_legacy_retriever() -> None:
    class LegacyRetriever:
        async def retrieve(self, question, *, top_k):
            return []

    with pytest.raises(UnsupportedStrictDocumentScopeError):
        await retrieve_compatibly(
            LegacyRetriever(),
            "DAF",
            top_k=5,
            constraints=RetrievalConstraints(
                document_scope=DocumentScope.strict_scope((uuid.uuid4(),)),
            ),
        )


def _subject(
    name: str,
    kind: SubjectKind,
    *,
    aliases: tuple[str, ...] = (),
) -> SubjectScopeCatalogEntry:
    return SubjectScopeCatalogEntry(
        subject_id=uuid.uuid4(),
        kind=kind,
        name=name,
        aliases=aliases,
    )
