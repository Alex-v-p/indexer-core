from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import CheckConstraint, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import configure_mappers

from packages.indexer_infrastructure.postgres import Base
from packages.indexer_infrastructure.postgres.models import (
    ContentGroup,
    ContentGroupAlias,
    DocumentContentGroupAssignment,
    DocumentType,
    DocumentTypeDecision,
)
from packages.indexer_infrastructure.postgres.repositories import (
    SqlAlchemyContentGroupRepository,
    SqlAlchemyDocumentTypeRepository,
    to_content_group_record,
    to_document_content_group_assignment_record,
    to_document_type_decision_record,
    to_document_type_record,
)
from packages.indexer_infrastructure.postgres.repositories.document_organization import (
    _automatic_assignment_write_is_allowed,
    _automatic_type_write_is_allowed,
)
from packages.indexer_infrastructure.postgres.unit_of_work import SqlAlchemyUnitOfWork
from packages.indexer_application.ports import ContentGroupInUseError
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentContentGroupAssignment as DomainAssignment,
    DocumentTypeDecision as DomainTypeDecision,
    DocumentTypeDecisionState,
)


def test_document_organization_tables_constraints_and_exactly_one_row_are_registered() -> None:
    configure_mappers()
    assert {
        "document_types", "document_type_decisions", "content_groups",
        "content_group_aliases", "document_content_group_assignments",
    }.issubset(Base.metadata.tables)
    assert list(DocumentContentGroupAssignment.__table__.primary_key.columns.keys()) == ["document_id"]
    assert isinstance(DocumentTypeDecision.__table__.c.state.type, String)
    checks = {
        constraint.name
        for constraint in DocumentContentGroupAssignment.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert {
        "ck_document_content_group_assignments_state_group",
        "ck_document_content_group_assignments_manual",
        "ck_document_content_group_assign_auto_resolved_provenance",
    }.issubset(checks)


def test_catalogue_uniqueness_and_delete_rules_are_explicit() -> None:
    type_unique = {tuple(column.name for column in constraint.columns) for constraint in DocumentType.__table__.constraints if constraint.__class__.__name__ == "UniqueConstraint"}
    group_unique = {tuple(column.name for column in constraint.columns) for constraint in ContentGroup.__table__.constraints if constraint.__class__.__name__ == "UniqueConstraint"}
    alias_unique = {tuple(column.name for column in constraint.columns) for constraint in ContentGroupAlias.__table__.constraints if constraint.__class__.__name__ == "UniqueConstraint"}
    assert ("key",) in type_unique
    assert ("normalized_name",) in group_unique
    assert ("normalized_name",) in alias_unique
    assert {(fk.target_fullname, fk.ondelete) for fk in DocumentContentGroupAssignment.__table__.foreign_keys} == {
        ("documents.id", "CASCADE"), ("content_groups.id", "RESTRICT")
    }


def test_mappers_round_trip_typed_records() -> None:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    type_id = uuid.uuid4()
    group_id = uuid.uuid4()
    document_id = uuid.uuid4()
    document_type = DocumentType(id=type_id, key="report", label="Report", metadata_={}, created_at=now, updated_at=now)
    type_decision = DocumentTypeDecision(id=uuid.uuid4(), document_id=document_id, document_type_id=type_id, state="assigned", source="manual", signals={}, revision=2, created_at=now, updated_at=now)
    group = ContentGroup(id=group_id, name="Factory Dashboard", normalized_name="factory dashboard", metadata_={}, created_at=now, updated_at=now)
    group.aliases = []
    assignment = DocumentContentGroupAssignment(document_id=document_id, content_group_id=group_id, state="assigned", source="manual", signals={}, revision=3, created_at=now, updated_at=now)

    assert to_document_type_record(document_type).key == "report"
    assert to_document_type_decision_record(type_decision).state is DocumentTypeDecisionState.ASSIGNED
    assert to_content_group_record(group).normalized_name == "factory dashboard"
    assert to_document_content_group_assignment_record(assignment).state is ContentGroupAssignmentState.ASSIGNED


def test_automatic_sql_guards_cannot_overwrite_manual_rows() -> None:
    type_write = DomainTypeDecision(document_id=uuid.uuid4(), document_type_id=uuid.uuid4(), state=DocumentTypeDecisionState.REJECTED, source=ClassificationSource.AUTOMATIC, confidence=0.2, confidence_band="low", rationale="Weak evidence.", classifier_version="v1", policy_version="v1", classified_document_version_id=uuid.uuid4())
    assignment_write = DomainAssignment(document_id=uuid.uuid4(), content_group_id=None, state=ContentGroupAssignmentState.PENDING, source=ClassificationSource.AUTOMATIC)
    for expression, table in (
        (_automatic_type_write_is_allowed(type_write), DocumentTypeDecision),
        (_automatic_assignment_write_is_allowed(assignment_write), DocumentContentGroupAssignment),
    ):
        compiled = table.__table__.select().where(expression).compile(dialect=postgresql.dialect())
        assert "manual" in compiled.params.values()
        assert "source" in str(compiled)


class RecordingSession:
    async def flush(self) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


def test_unit_of_work_exposes_separate_repositories() -> None:
    session = RecordingSession()
    uow = SqlAlchemyUnitOfWork(session)  # type: ignore[arg-type]
    assert isinstance(uow.document_types, SqlAlchemyDocumentTypeRepository)
    assert isinstance(uow.content_groups, SqlAlchemyContentGroupRepository)


class ScalarResult:
    def __init__(self, value): self.value = value
    def scalar_one_or_none(self): return self.value


class ArchiveSession:
    def __init__(self, group, document_id): self.results = [group, document_id]
    async def execute(self, statement):
        del statement
        return ScalarResult(self.results.pop(0))
    async def flush(self): ...


@pytest.mark.asyncio
async def test_content_group_archive_guard_rejects_active_references() -> None:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    group = ContentGroup(id=uuid.uuid4(), name="Factory Dashboard", normalized_name="factory dashboard", metadata_={}, created_at=now, updated_at=now)
    group.aliases = []
    repository = SqlAlchemyContentGroupRepository(ArchiveSession(group, uuid.uuid4()))  # type: ignore[arg-type]
    with pytest.raises(ContentGroupInUseError):
        await repository.archive(group.id)
