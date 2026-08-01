from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from packages.indexer_application.commands import (
    AddSubjectAliasCommand,
    AddSubjectAliasHandler,
    ArchiveSubjectAliasCommand,
    ArchiveSubjectAliasHandler,
    CreateSubjectCommand,
    CreateSubjectHandler,
    ArchivedSubjectMutationError,
    DocumentSubjectDecisionWriteConflict,
    ReviewDocumentSubjectSuggestionCommand,
    ReviewDocumentSubjectSuggestionHandler,
    SetDocumentSubjectDecisionCommand,
    SetDocumentSubjectDecisionHandler,
    UpdateSubjectCommand,
    UpdateSubjectHandler,
)
from packages.indexer_application.dto import (
    DocumentSubjectDecisionRecord,
    SubjectAliasRecord,
    SubjectNameMatchRecord,
    SubjectRecord,
)
from packages.indexer_application.ports import (
    SubjectCanonicalNameConflictError,
    SubjectDecisionConflictError,
)
from packages.indexer_application.queries import (
    ListDocumentSubjectSuggestionsHandler,
    ListDocumentSubjectSuggestionsQuery,
    ResolveSubjectNameHandler,
    ResolveSubjectNameQuery,
)
from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SubjectKind,
    SubjectNameMatchType,
)


def _subject(
    *,
    subject_id: uuid.UUID | None = None,
    name: str = "Orion",
    kind: SubjectKind = SubjectKind.PROJECT,
    archived: bool = False,
) -> SubjectRecord:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    return SubjectRecord(
        id=subject_id or uuid.uuid4(),
        kind=kind,
        name=name,
        normalized_name=name.casefold(),
        description=None,
        metadata={},
        created_at=now,
        updated_at=now,
        archived_at=now if archived else None,
    )


def _decision(
    *,
    document_id: uuid.UUID,
    subject_id: uuid.UUID,
    state: DecisionState,
    source: DecisionControlSource,
    revision: int,
) -> DocumentSubjectDecisionRecord:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    automatic = source is DecisionControlSource.AUTOMATIC
    return DocumentSubjectDecisionRecord(
        id=uuid.uuid4(),
        document_id=document_id,
        subject_id=subject_id,
        state=state,
        control_source=source,
        confidence=0.7 if automatic else None,
        confidence_band=ConfidenceBand.MEDIUM if automatic else None,
        rationale="Classifier suggestion." if automatic else "Operator decision.",
        classifier_version="v1" if automatic else None,
        policy_version="v1" if automatic else None,
        signals={"title": True} if automatic else {},
        classified_document_version_id=uuid.uuid4() if automatic else None,
        revision=revision,
        created_at=now,
        updated_at=now,
    )


class FakeDocuments:
    def __init__(self, existing_id: uuid.UUID) -> None:
        self.existing_id = existing_id

    async def get(self, document_id: uuid.UUID):
        return object() if document_id == self.existing_id else None


class FakeSubjects:
    def __init__(self, subject: SubjectRecord) -> None:
        self.subject = subject
        self.matches: list[SubjectNameMatchRecord] = []
        self.current: DocumentSubjectDecisionRecord | None = None
        self.writes = []
        self.raise_conflict = False

    async def resolve_name(self, name, **kwargs):
        self.resolve_call = (name, kwargs)
        return list(self.matches)

    async def create(self, **kwargs):
        self.create_call = kwargs
        return self.subject

    async def get(self, subject_id, *, include_archived=False):
        if subject_id != self.subject.id:
            return None
        if self.subject.archived_at is not None and not include_archived:
            return None
        return self.subject

    async def rename(self, **kwargs):
        self.rename_call = kwargs
        return self.subject

    async def archive(self, **kwargs):
        self.archive_call = kwargs
        return _subject(
            subject_id=self.subject.id,
            name=self.subject.name,
            kind=self.subject.kind,
            archived=True,
        )

    async def add_alias(self, **kwargs):
        self.add_alias_call = kwargs
        now = datetime(2026, 8, 1, tzinfo=UTC)
        return SubjectAliasRecord(
            id=uuid.uuid4(),
            subject_id=kwargs["subject_id"],
            name=kwargs["name"],
            normalized_name=kwargs["name"].casefold(),
            created_at=now,
        )

    async def archive_alias(self, **kwargs):
        self.archive_alias_call = kwargs
        now = datetime(2026, 8, 1, tzinfo=UTC)
        return SubjectAliasRecord(
            id=kwargs["alias_id"],
            subject_id=kwargs["subject_id"],
            name="Project Orion",
            normalized_name="project orion",
            created_at=now,
            archived_at=now,
        )

    async def get_decision(self, **kwargs):
        self.get_decision_call = kwargs
        return self.current

    async def write_decision(self, decision, *, expected_revision):
        self.writes.append((decision, expected_revision))
        if self.raise_conflict:
            raise SubjectDecisionConflictError("stale")
        return _decision(
            document_id=decision.document_id,
            subject_id=decision.subject_id,
            state=decision.state,
            source=decision.control_source,
            revision=max(1, expected_revision + 1),
        )

    async def list_document_decisions(self, **kwargs):
        self.list_call = kwargs
        return [self.current] if self.current is not None else []

    async def list_document_suggestions(self, **kwargs):
        self.suggestions_call = kwargs
        return [self.current] if self.current is not None else []


class FakeUnitOfWork:
    def __init__(self, *, document_id: uuid.UUID, subject: SubjectRecord) -> None:
        self.documents = FakeDocuments(document_id)
        self.subjects = FakeSubjects(subject)
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


@pytest.mark.asyncio
async def test_create_subject_checks_canonical_name_and_commits() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)

    result = await CreateSubjectHandler(uow=uow)(
        CreateSubjectCommand(kind=SubjectKind.PROJECT, name="Orion"),
    )

    assert result is subject
    assert uow.subjects.resolve_call == (
        "Orion",
        {"kind": SubjectKind.PROJECT, "include_archived": True},
    )
    assert uow.subjects.create_call["kind"] is SubjectKind.PROJECT
    assert uow.commit_calls == 1


@pytest.mark.asyncio
async def test_create_subject_rejects_duplicate_canonical_but_not_alias_collision() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)
    uow.subjects.matches = [
        SubjectNameMatchRecord(
            subject=_subject(name="Different"),
            match_type=SubjectNameMatchType.ALIAS,
        ),
    ]

    await CreateSubjectHandler(uow=uow)(
        CreateSubjectCommand(kind=SubjectKind.PROJECT, name="Orion"),
    )

    uow.subjects.matches = [
        SubjectNameMatchRecord(
            subject=_subject(name="Orion"),
            match_type=SubjectNameMatchType.CANONICAL,
        ),
    ]
    with pytest.raises(
        SubjectCanonicalNameConflictError,
        match="canonical name already exists",
    ):
        await CreateSubjectHandler(uow=uow)(
            CreateSubjectCommand(kind=SubjectKind.PROJECT, name="Orion"),
        )


@pytest.mark.asyncio
async def test_archived_subject_is_read_only_for_rename_and_decisions() -> None:
    document_id = uuid.uuid4()
    subject = _subject(archived=True)
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)

    with pytest.raises(ArchivedSubjectMutationError, match="Archived subjects are read-only"):
        await UpdateSubjectHandler(uow=uow)(
            UpdateSubjectCommand(subject_id=subject.id, name="New Orion"),
        )
    with pytest.raises(ArchivedSubjectMutationError, match="Archived subjects are read-only"):
        await SetDocumentSubjectDecisionHandler(uow=uow)(
            SetDocumentSubjectDecisionCommand(
                document_id=document_id,
                subject_id=subject.id,
                state=DecisionState.ASSIGNED,
                expected_revision=0,
            ),
        )

    assert uow.subjects.writes == []


@pytest.mark.asyncio
async def test_subject_can_be_renamed_and_archived_in_one_update() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)

    result = await UpdateSubjectHandler(uow=uow)(
        UpdateSubjectCommand(
            subject_id=subject.id,
            name="Orion Program",
            archive=True,
        ),
    )

    assert uow.subjects.rename_call == {
        "subject_id": subject.id,
        "name": "Orion Program",
    }
    assert uow.subjects.archive_call == {"subject_id": subject.id}
    assert result.archived_at is not None
    assert uow.commit_calls == 1


@pytest.mark.asyncio
async def test_alias_add_and_archive_use_subject_scoped_mutations() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    alias_id = uuid.uuid4()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)

    alias = await AddSubjectAliasHandler(uow=uow)(
        AddSubjectAliasCommand(subject_id=subject.id, name="Project Orion"),
    )
    archived = await ArchiveSubjectAliasHandler(uow=uow)(
        ArchiveSubjectAliasCommand(subject_id=subject.id, alias_id=alias_id),
    )

    assert alias.name == "Project Orion"
    assert uow.subjects.add_alias_call == {
        "subject_id": subject.id,
        "name": "Project Orion",
    }
    assert uow.subjects.archive_alias_call == {
        "subject_id": subject.id,
        "alias_id": alias_id,
    }
    assert archived.archived_at is not None
    assert uow.commit_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [DecisionState.ASSIGNED, DecisionState.REJECTED])
async def test_manual_assignment_and_rejection_use_manual_cas_decisions(
    state: DecisionState,
) -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)

    result = await SetDocumentSubjectDecisionHandler(uow=uow)(
        SetDocumentSubjectDecisionCommand(
            document_id=document_id,
            subject_id=subject.id,
            state=state,
            expected_revision=2,
            rationale="Operator choice.",
        ),
    )

    written, revision = uow.subjects.writes[0]
    assert written.state is state
    assert written.control_source is DecisionControlSource.MANUAL
    assert revision == 2
    assert result.state is state
    assert uow.commit_calls == 1


@pytest.mark.asyncio
async def test_manual_conflict_exposes_current_revision_context() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)
    current = _decision(
        document_id=document_id,
        subject_id=subject.id,
        state=DecisionState.ASSIGNED,
        source=DecisionControlSource.MANUAL,
        revision=4,
    )
    uow.subjects.current = current
    uow.subjects.raise_conflict = True

    with pytest.raises(DocumentSubjectDecisionWriteConflict) as raised:
        await SetDocumentSubjectDecisionHandler(uow=uow)(
            SetDocumentSubjectDecisionCommand(
                document_id=document_id,
                subject_id=subject.id,
                state=DecisionState.REJECTED,
                expected_revision=3,
            ),
        )

    assert raised.value.current is current
    assert raised.value.current.revision == 4
    assert uow.commit_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("accept", "expected_state"),
    [(True, DecisionState.ASSIGNED), (False, DecisionState.REJECTED)],
)
async def test_automatic_suggestion_review_becomes_authoritative_manual_decision(
    accept: bool,
    expected_state: DecisionState,
) -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)
    uow.subjects.current = _decision(
        document_id=document_id,
        subject_id=subject.id,
        state=DecisionState.SUGGESTED,
        source=DecisionControlSource.AUTOMATIC,
        revision=5,
    )

    result = await ReviewDocumentSubjectSuggestionHandler(uow=uow)(
        ReviewDocumentSubjectSuggestionCommand(
            document_id=document_id,
            subject_id=subject.id,
            accept=accept,
            expected_revision=5,
        ),
    )

    written, revision = uow.subjects.writes[0]
    assert written.state is expected_state
    assert written.control_source is DecisionControlSource.MANUAL
    assert revision == 5
    assert result.state is expected_state


@pytest.mark.asyncio
async def test_manual_decision_cannot_be_reviewed_as_an_automatic_suggestion() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)
    uow.subjects.current = _decision(
        document_id=document_id,
        subject_id=subject.id,
        state=DecisionState.ASSIGNED,
        source=DecisionControlSource.MANUAL,
        revision=6,
    )

    with pytest.raises(ValueError, match="Only an automatic suggestion"):
        await ReviewDocumentSubjectSuggestionHandler(uow=uow)(
            ReviewDocumentSubjectSuggestionCommand(
                document_id=document_id,
                subject_id=subject.id,
                accept=False,
                expected_revision=6,
            ),
        )

    assert uow.subjects.writes == []


@pytest.mark.asyncio
async def test_suggestion_listing_filters_to_suggested_state() -> None:
    document_id = uuid.uuid4()
    subject = _subject()
    uow = FakeUnitOfWork(document_id=document_id, subject=subject)
    uow.subjects.current = _decision(
        document_id=document_id,
        subject_id=subject.id,
        state=DecisionState.SUGGESTED,
        source=DecisionControlSource.AUTOMATIC,
        revision=1,
    )

    result = await ListDocumentSubjectSuggestionsHandler(uow=uow)(
        ListDocumentSubjectSuggestionsQuery(document_id=document_id),
    )

    assert len(result) == 1
    assert uow.subjects.suggestions_call == {"document_id": document_id}


@pytest.mark.asyncio
async def test_name_resolution_preserves_ambiguous_alias_matches() -> None:
    document_id = uuid.uuid4()
    uow = FakeUnitOfWork(document_id=document_id, subject=_subject())
    uow.subjects.matches = [
        SubjectNameMatchRecord(
            subject=_subject(name="One"),
            match_type=SubjectNameMatchType.ALIAS,
        ),
        SubjectNameMatchRecord(
            subject=_subject(name="Two"),
            match_type=SubjectNameMatchType.ALIAS,
        ),
    ]

    matches = await ResolveSubjectNameHandler(uow=uow)(
        ResolveSubjectNameQuery(name="shared"),
    )

    assert len(matches) == 2
    assert all(match.match_type is SubjectNameMatchType.ALIAS for match in matches)
