from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import CheckConstraint, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import configure_mappers

from packages.indexer_infrastructure.postgres import Base
from packages.indexer_infrastructure.postgres.models import (
    Document,
    DocumentSubjectDecision,
    QdrantChunkIndex,
    Subject,
    SubjectAlias,
)
from packages.indexer_infrastructure.postgres.repositories import (
    SqlAlchemySubjectRepository,
    to_document_subject_decision_record,
    to_subject_record,
)
from packages.indexer_application.ports import (
    SubjectCanonicalNameConflictError,
    SubjectDecisionConflictError,
)
from packages.indexer_infrastructure.postgres.repositories.subjects import (
    _automatic_write_is_allowed,
)
from packages.indexer_infrastructure.postgres.unit_of_work import SqlAlchemyUnitOfWork
from packages.rag_core.subjects import (
    DecisionControlSource,
    DecisionState,
    ConfidenceBand,
    SubjectKind,
    DocumentSubjectDecision as DomainDecision,
)


def test_subject_tables_and_relationships_are_registered() -> None:
    configure_mappers()

    assert {
        "subjects",
        "subject_aliases",
        "document_subject_decisions",
    }.issubset(Base.metadata.tables)
    assert Document.subject_decisions.property.passive_deletes is True


def test_subject_values_use_checked_strings_not_postgres_enums() -> None:
    assert isinstance(Subject.__table__.c.kind.type, String)
    assert isinstance(DocumentSubjectDecision.__table__.c.state.type, String)
    assert isinstance(
        DocumentSubjectDecision.__table__.c.control_source.type,
        String,
    )
    checks = {
        constraint.name
        for constraint in DocumentSubjectDecision.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert {
        "ck_document_subject_decisions_state",
        "ck_document_subject_decisions_control_source",
        "ck_document_subject_decisions_automatic_provenance",
        "ck_document_subject_decisions_manual_provenance",
        "ck_document_subject_decisions_signals",
    }.issubset(checks)


def test_alias_uniqueness_allows_cross_subject_collisions() -> None:
    unique_column_sets = {
        tuple(column.name for column in constraint.columns)
        for constraint in SubjectAlias.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }

    assert ("subject_id", "normalized_name") in unique_column_sets
    assert ("normalized_name",) not in unique_column_sets


def test_canonical_name_uniqueness_is_scoped_by_kind() -> None:
    unique_column_sets = {
        tuple(column.name for column in constraint.columns)
        for constraint in Subject.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }

    assert ("kind", "normalized_name") in unique_column_sets
    assert ("normalized_name",) not in unique_column_sets


def test_document_deletion_cascades_subject_decisions() -> None:
    foreign_keys = {
        (foreign_key.target_fullname, foreign_key.ondelete)
        for foreign_key in DocumentSubjectDecision.__table__.foreign_keys
    }

    assert ("documents.id", "CASCADE") in foreign_keys
    assert ("subjects.id", "CASCADE") in foreign_keys


def test_classified_version_is_retained_as_audit_uuid_without_delete_cascade() -> None:
    column = DocumentSubjectDecision.__table__.c.classified_document_version_id

    assert not column.foreign_keys
    assert column.nullable is True


def test_qdrant_registry_has_no_authoritative_subject_identity() -> None:
    assert "subject_id" not in QdrantChunkIndex.__table__.columns
    assert "subject_ids" not in QdrantChunkIndex.__table__.columns


def test_subject_and_decision_mappers_preserve_typed_records() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    subject_id = uuid.uuid4()
    subject = Subject(
        id=subject_id,
        kind=SubjectKind.PROJECT.value,
        name="Orion",
        normalized_name="orion",
        metadata_={"color": "blue"},
        created_at=now,
        updated_at=now,
    )
    subject.aliases = [
        SubjectAlias(
            id=uuid.uuid4(),
            subject_id=subject_id,
            name="Project Orion",
            normalized_name="project orion",
            created_at=now,
        ),
        SubjectAlias(
            id=uuid.uuid4(),
            subject_id=subject_id,
            name="Old Orion",
            normalized_name="old orion",
            created_at=now,
            archived_at=now,
        ),
    ]
    decision = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        subject_id=subject_id,
        state=DecisionState.REJECTED.value,
        control_source=DecisionControlSource.MANUAL.value,
        signals={},
        revision=3,
        created_at=now,
        updated_at=now,
    )

    subject_record = to_subject_record(subject)
    decision_record = to_document_subject_decision_record(decision)

    assert subject_record.kind is SubjectKind.PROJECT
    assert [alias.name for alias in subject_record.aliases] == ["Project Orion"]
    assert decision_record.state is DecisionState.REJECTED
    assert decision_record.is_membership is False
    assert decision_record.revision == 3


class RecordingSession:
    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def test_unit_of_work_exposes_subject_repository() -> None:
    session = RecordingSession()
    uow = SqlAlchemyUnitOfWork(session)  # type: ignore[arg-type]

    assert isinstance(uow.subjects, SqlAlchemySubjectRepository)
    assert uow.subjects._session is session


def test_automatic_write_guard_preserves_manual_assigned_and_rejected_rows() -> None:
    proposed = DomainDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.AUTOMATIC,
        confidence=0.7,
        confidence_band=ConfidenceBand.MEDIUM,
        rationale="Document evidence matches.",
        classifier_version="v1",
        policy_version="v1",
        classified_document_version_id=uuid.uuid4(),
    )
    statement = DocumentSubjectDecision.__table__.select().where(
        _automatic_write_is_allowed(proposed),
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "control_source" in sql
    assert "state" in sql
    assert "manual" in compiled.params.values()
    assert ["assigned", "rejected"] in compiled.params.values()


class ScalarResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class DecisionRecordingSession:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> ScalarResult:
        self.statements.append(statement)
        return ScalarResult(self.results.pop(0))

    async def flush(self) -> None:
        return None


class CollectionResult:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def scalars(self) -> CollectionResult:
        return self

    def all(self) -> list[Any]:
        return self.values


class CollectionRecordingSession:
    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> CollectionResult:
        self.statements.append(statement)
        return CollectionResult(self.values)


@pytest.mark.asyncio
async def test_automatic_decision_write_uses_guarded_atomic_upsert() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    document_id = uuid.uuid4()
    subject_id = uuid.uuid4()
    version_id = uuid.uuid4()
    proposed = DomainDecision(
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.AUTOMATIC,
        confidence=0.7,
        confidence_band=ConfidenceBand.MEDIUM,
        rationale="Document evidence matches.",
        classifier_version="v1",
        policy_version="v1",
        signals={"title_match": True},
        classified_document_version_id=version_id,
    )
    persisted = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.ASSIGNED.value,
        control_source=DecisionControlSource.AUTOMATIC.value,
        confidence=0.7,
        confidence_band=ConfidenceBand.MEDIUM.value,
        rationale="Document evidence matches.",
        classifier_version="v1",
        policy_version="v1",
        signals={"title_match": True},
        classified_document_version_id=version_id,
        revision=1,
        created_at=now,
        updated_at=now,
    )
    session = DecisionRecordingSession([object(), version_id, persisted])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    record = await repository.write_decision(proposed)

    assert record.revision == 1
    assert len(session.statements) == 3
    sql = str(session.statements[2].compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT" in sql
    assert "DO UPDATE SET" in sql
    assert "document_subject_decisions.revision +" in sql
    assert "control_source" in sql


@pytest.mark.asyncio
async def test_manual_decision_write_uses_expected_revision() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    document_id = uuid.uuid4()
    subject_id = uuid.uuid4()
    proposed = DomainDecision(
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.REJECTED,
        control_source=DecisionControlSource.MANUAL,
        rationale="Operator removed the assignment.",
    )
    persisted = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.REJECTED.value,
        control_source=DecisionControlSource.MANUAL.value,
        rationale="Operator removed the assignment.",
        signals={},
        revision=5,
        created_at=now,
        updated_at=now,
    )
    session = DecisionRecordingSession([object(), persisted])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    record = await repository.write_decision(proposed, expected_revision=4)

    assert record.revision == 5
    sql = str(session.statements[1].compile(dialect=postgresql.dialect()))
    assert sql.startswith("UPDATE document_subject_decisions")
    assert "document_subject_decisions.revision =" in sql
    assert "document_subject_decisions.revision +" in sql


@pytest.mark.asyncio
async def test_manual_decision_requires_expected_revision() -> None:
    proposed = DomainDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.MANUAL,
    )
    session = DecisionRecordingSession([])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="manual decisions require expected_revision"):
        await repository.write_decision(proposed)

    assert session.statements == []


@pytest.mark.asyncio
async def test_stale_manual_cas_raises_even_when_payload_matches_current_row() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    document_id = uuid.uuid4()
    subject_id = uuid.uuid4()
    proposed = DomainDecision(
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.MANUAL,
        rationale="Current operator decision.",
    )
    identical_current = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.ASSIGNED.value,
        control_source=DecisionControlSource.MANUAL.value,
        rationale="Current operator decision.",
        signals={},
        revision=4,
        created_at=now,
        updated_at=now,
    )
    # The UPDATE returning no row is authoritative. The repository must not
    # read and compare payloads to turn a stale explicit CAS into success.
    session = DecisionRecordingSession([object(), None, identical_current])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(SubjectDecisionConflictError, match="expected revision"):
        await repository.write_decision(proposed, expected_revision=3)

    assert len(session.statements) == 2
    assert session.results == [identical_current]


@pytest.mark.asyncio
async def test_stale_manual_cas_with_different_payload_raises_conflict() -> None:
    proposed = DomainDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.REJECTED,
        control_source=DecisionControlSource.MANUAL,
        rationale="A different operator decision.",
    )
    session = DecisionRecordingSession([object(), None])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(SubjectDecisionConflictError, match="expected revision"):
        await repository.write_decision(proposed, expected_revision=2)


@pytest.mark.asyncio
async def test_manual_revision_zero_conflicts_when_decision_already_exists() -> None:
    proposed = DomainDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.MANUAL,
    )
    session = DecisionRecordingSession([object(), None])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(SubjectDecisionConflictError, match="expected revision"):
        await repository.write_decision(proposed, expected_revision=0)


@pytest.mark.asyncio
async def test_decision_write_rejects_archived_subject() -> None:
    proposed = DomainDecision(
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.MANUAL,
    )
    # _require_subject filters archived_at, so an archived subject resolves to None.
    session = DecisionRecordingSession([None])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(LookupError, match="Active subject"):
        await repository.write_decision(proposed, expected_revision=0)

    assert len(session.statements) == 1
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "subjects.archived_at IS NULL" in sql
    assert "FOR UPDATE" in sql


@pytest.mark.asyncio
async def test_historical_decision_remains_readable_after_subject_is_archived() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    current = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.ASSIGNED.value,
        control_source=DecisionControlSource.MANUAL.value,
        signals={},
        revision=2,
        created_at=now,
        updated_at=now,
    )
    session = DecisionRecordingSession([current])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    record = await repository.get_decision(
        document_id=current.document_id,
        subject_id=current.subject_id,
    )

    assert record is not None
    assert record.revision == 2
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "subjects" not in sql
    assert "archived_at" not in sql


@pytest.mark.asyncio
async def test_archive_alias_locks_and_requires_active_subject() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    subject_id = uuid.uuid4()
    alias_id = uuid.uuid4()
    subject = Subject(
        id=subject_id,
        kind=SubjectKind.PROJECT.value,
        name="Orion",
        normalized_name="orion",
        metadata_={},
        created_at=now,
        updated_at=now,
    )
    alias = SubjectAlias(
        id=alias_id,
        subject_id=subject_id,
        name="Project Orion",
        normalized_name="project orion",
        created_at=now,
    )
    session = DecisionRecordingSession([subject, alias])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    result = await repository.archive_alias(subject_id=subject_id, alias_id=alias_id)

    assert result.archived_at is not None
    lock_sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "subjects.archived_at IS NULL" in lock_sql
    assert "FOR UPDATE" in lock_sql


@pytest.mark.asyncio
async def test_archive_alias_rejects_archived_subject_before_alias_mutation() -> None:
    session = DecisionRecordingSession([None])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(LookupError, match="Active subject"):
        await repository.archive_alias(
            subject_id=uuid.uuid4(),
            alias_id=uuid.uuid4(),
        )

    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_suggestion_query_filters_mixed_sources_and_archived_subjects() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    model = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        state=DecisionState.SUGGESTED.value,
        control_source=DecisionControlSource.AUTOMATIC.value,
        confidence=0.7,
        confidence_band=ConfidenceBand.MEDIUM.value,
        rationale="Classifier suggestion.",
        classifier_version="v1",
        policy_version="v1",
        signals={},
        classified_document_version_id=uuid.uuid4(),
        revision=1,
        created_at=now,
        updated_at=now,
    )
    session = CollectionRecordingSession([model])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    records = await repository.list_document_suggestions(
        document_id=model.document_id,
    )

    assert len(records) == 1
    compiled = session.statements[0].compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "subjects.archived_at IS NULL" in sql
    assert "control_source" in sql
    assert "state" in sql
    assert DecisionControlSource.AUTOMATIC.value in compiled.params.values()
    assert DecisionState.SUGGESTED.value in compiled.params.values()


@pytest.mark.asyncio
async def test_general_decision_history_does_not_hide_archived_or_manual_rows() -> None:
    session = CollectionRecordingSession([])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    records = await repository.list_document_decisions(
        document_id=uuid.uuid4(),
    )

    assert records == []
    compiled = session.statements[0].compile(dialect=postgresql.dialect())
    assert "subjects.archived_at IS NULL" not in str(compiled)
    assert DecisionControlSource.AUTOMATIC.value not in compiled.params.values()
    assert DecisionState.SUGGESTED.value not in compiled.params.values()


class ConstraintViolation(Exception):
    def __init__(self, constraint_name: str) -> None:
        super().__init__(constraint_name)
        self.constraint_name = constraint_name


class FailingFlushSession(DecisionRecordingSession):
    def __init__(self, *, error: IntegrityError, results: list[Any] | None = None) -> None:
        super().__init__(results or [])
        self.error = error
        self.added: list[Any] = []

    def add(self, model: Any) -> None:
        self.added.append(model)

    async def flush(self) -> None:
        raise self.error


class SuccessfulSubjectCreateSession:
    def __init__(self) -> None:
        self.added: Subject | None = None
        self.aliases_initialized_before_flush = False

    def add(self, model: Subject) -> None:
        self.added = model

    async def flush(self) -> None:
        assert self.added is not None
        self.aliases_initialized_before_flush = (
            "aliases" in self.added.__dict__ and self.added.aliases == []
        )
        self.added.id = uuid.uuid4()
        self.added.created_at = datetime(2026, 8, 1, tzinfo=UTC)
        self.added.updated_at = datetime(2026, 8, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_create_initializes_alias_collection_before_flush() -> None:
    session = SuccessfulSubjectCreateSession()
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    created = await repository.create(
        kind=SubjectKind.PROJECT,
        name="Orion",
        metadata={"fixture": True},
    )

    assert session.aliases_initialized_before_flush is True
    assert created.name == "Orion"
    assert created.aliases == ()
    assert created.metadata == {"fixture": True}


@pytest.mark.asyncio
async def test_atomic_create_or_get_uses_canonical_conflict_then_loaded_read() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    subject = Subject(
        id=uuid.uuid4(),
        kind=SubjectKind.PROJECT.value,
        name="Orion",
        normalized_name="orion",
        metadata_={"created_by": "automatic_subject_discovery"},
        created_at=now,
        updated_at=now,
    )
    subject.aliases = []
    session = DecisionRecordingSession([None, subject])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    resolved, created = await repository.create_or_get_canonical(
        kind=SubjectKind.PROJECT,
        name="Orion",
    )

    assert created is False
    assert resolved.id == subject.id
    insert_sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_subjects_kind_normalized_name DO NOTHING" in insert_sql
    assert "RETURNING subjects.id" in insert_sql
    assert "SELECT subjects" in str(session.statements[1])


class SubjectMutationSession(DecisionRecordingSession):
    def __init__(self, subject: Subject) -> None:
        super().__init__([subject])
        self.subject = subject
        self.original_updated_at = subject.updated_at

    async def flush(self) -> None:
        # Mimic SQLAlchemy expiring a server-updated timestamp when the
        # repository does not explicitly provide the new value.
        if self.subject.updated_at == self.original_updated_at:
            self.subject.__dict__.pop("updated_at", None)


@pytest.mark.asyncio
async def test_rename_and_archive_map_without_server_refresh_and_keep_aliases_loaded() -> None:
    now = datetime(2026, 7, 31, tzinfo=UTC)
    subject = Subject(
        id=uuid.uuid4(),
        kind=SubjectKind.PROJECT.value,
        name="Orion",
        normalized_name="orion",
        metadata_={},
        created_at=now,
        updated_at=now,
    )
    subject.aliases = [
        SubjectAlias(
            id=uuid.uuid4(),
            subject_id=subject.id,
            name="Project Orion",
            normalized_name="project orion",
            created_at=now,
        ),
    ]
    rename_session = SubjectMutationSession(subject)
    renamed = await SqlAlchemySubjectRepository(  # type: ignore[arg-type]
        rename_session,
    ).rename(subject_id=subject.id, name="Orion Program")

    assert renamed.updated_at > now
    assert [alias.name for alias in renamed.aliases] == ["Project Orion"]

    archive_session = SubjectMutationSession(subject)
    archived = await SqlAlchemySubjectRepository(  # type: ignore[arg-type]
        archive_session,
    ).archive(subject_id=subject.id)

    assert archived.archived_at is not None
    assert archived.updated_at >= renamed.updated_at
    assert [alias.name for alias in archived.aliases] == ["Project Orion"]


def _integrity_error(constraint_name: str) -> IntegrityError:
    return IntegrityError(
        "forced statement",
        {},
        ConstraintViolation(constraint_name),
    )


@pytest.mark.asyncio
async def test_create_translates_only_canonical_unique_constraint_race() -> None:
    session = FailingFlushSession(
        error=_integrity_error("uq_subjects_kind_normalized_name"),
    )
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(SubjectCanonicalNameConflictError, match="canonical name"):
        await repository.create(kind=SubjectKind.PROJECT, name="Orion")


@pytest.mark.asyncio
async def test_rename_translates_canonical_unique_constraint_race() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    subject = Subject(
        id=uuid.uuid4(),
        kind=SubjectKind.PROJECT.value,
        name="Orion",
        normalized_name="orion",
        metadata_={},
        created_at=now,
        updated_at=now,
    )
    subject.aliases = []
    session = FailingFlushSession(
        error=_integrity_error("uq_subjects_kind_normalized_name"),
        results=[subject],
    )
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(SubjectCanonicalNameConflictError, match="canonical name"):
        await repository.rename(subject_id=subject.id, name="Apollo")


@pytest.mark.asyncio
async def test_unrelated_integrity_failure_is_not_translated() -> None:
    error = _integrity_error("fk_unrelated_constraint")
    session = FailingFlushSession(error=error)
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    with pytest.raises(IntegrityError) as raised:
        await repository.create(kind=SubjectKind.PROJECT, name="Orion")

    assert raised.value is error


@pytest.mark.asyncio
async def test_automatic_identical_rerun_remains_idempotent_without_explicit_cas() -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    document_id = uuid.uuid4()
    subject_id = uuid.uuid4()
    version_id = uuid.uuid4()
    proposed = DomainDecision(
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.ASSIGNED,
        control_source=DecisionControlSource.AUTOMATIC,
        confidence=0.7,
        confidence_band=ConfidenceBand.MEDIUM,
        rationale="Document evidence matches.",
        classifier_version="v1",
        policy_version="v1",
        signals={"title_match": True},
        classified_document_version_id=version_id,
    )
    current = DocumentSubjectDecision(
        id=uuid.uuid4(),
        document_id=document_id,
        subject_id=subject_id,
        state=DecisionState.ASSIGNED.value,
        control_source=DecisionControlSource.AUTOMATIC.value,
        confidence=0.7,
        confidence_band=ConfidenceBand.MEDIUM.value,
        rationale="Document evidence matches.",
        classifier_version="v1",
        policy_version="v1",
        signals={"title_match": True},
        classified_document_version_id=version_id,
        revision=6,
        created_at=now,
        updated_at=now,
    )
    session = DecisionRecordingSession([object(), version_id, None, current])
    repository = SqlAlchemySubjectRepository(session)  # type: ignore[arg-type]

    record = await repository.write_decision(proposed)

    assert record.revision == 6
    assert len(session.statements) == 4
