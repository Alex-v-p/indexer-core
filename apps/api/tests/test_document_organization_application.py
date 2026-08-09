from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from packages.indexer_application.commands import (
    CreateContentGroupCommand,
    CreateContentGroupHandler,
    CreateDocumentTypeCommand,
    CreateDocumentTypeHandler,
    DocumentOrganizationWriteConflict,
    ManualDocumentTypeDecisionInput,
    ReplaceDocumentTypesCommand,
    ReplaceDocumentTypesHandler,
    SetDocumentContentGroupCommand,
    SetDocumentContentGroupHandler,
    UpdateContentGroupCommand,
    UpdateContentGroupHandler,
    UpdateDocumentTypeCommand,
    UpdateDocumentTypeHandler,
)
from packages.indexer_application.dto import (
    ContentGroupRecord,
    DocumentContentGroupAssignmentRecord,
    DocumentTypeDecisionRecord,
    DocumentTypeRecord,
)
from packages.indexer_application.ports import DocumentOrganizationConflictError
from packages.indexer_application.queries import (
    GetDocumentOrganizationHandler,
    GetDocumentOrganizationQuery,
)
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentTypeDecisionState,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _document_type(*, archived: bool = False) -> DocumentTypeRecord:
    return DocumentTypeRecord(
        id=uuid.uuid4(),
        key="technical-design",
        label="Technical design",
        description=None,
        metadata={},
        archived_at=NOW if archived else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _group(*, archived: bool = False) -> ContentGroupRecord:
    return ContentGroupRecord(
        id=uuid.uuid4(),
        name="DAF",
        normalized_name="daf",
        description=None,
        metadata={},
        archived_at=NOW if archived else None,
        created_at=NOW,
        updated_at=NOW,
    )


def _decision(document_id, document_type_id, state, revision=1):
    return DocumentTypeDecisionRecord(
        id=uuid.uuid4(),
        document_id=document_id,
        document_type_id=document_type_id,
        state=state,
        source=ClassificationSource.MANUAL,
        confidence=None,
        confidence_band=None,
        rationale="curated",
        classifier_version=None,
        policy_version=None,
        signals={},
        classified_document_version_id=None,
        revision=revision,
        created_at=NOW,
        updated_at=NOW,
    )


def _assignment(document_id, group_id, *, state=ContentGroupAssignmentState.ASSIGNED, revision=1):
    return DocumentContentGroupAssignmentRecord(
        document_id=document_id,
        content_group_id=group_id,
        state=state,
        source=(
            ClassificationSource.MANUAL
            if state is ContentGroupAssignmentState.ASSIGNED
            else ClassificationSource.AUTOMATIC
        ),
        unresolved_reason=None,
        confidence=None,
        confidence_band=None,
        rationale=None,
        classifier_version=None,
        policy_version=None,
        signals={},
        summary_hash=None,
        classified_document_version_id=None,
        revision=revision,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeDocumentTypes:
    def __init__(self, value):
        self.value = value
        self.decisions = []
        self.conflict = False

    async def create(self, **kwargs):
        self.create_call = kwargs
        return self.value

    async def get(self, document_type_id, *, include_archived=False):
        return self.value if document_type_id == self.value.id else None

    async def update(self, document_type_id, **kwargs):
        self.update_call = (document_type_id, kwargs)
        return self.value

    async def archive(self, document_type_id):
        self.archive_call = document_type_id
        return self.value

    async def write_decision(self, decision, *, expected_revision):
        self.write_call = getattr(self, "write_call", []) + [(decision, expected_revision)]
        if self.conflict:
            raise DocumentOrganizationConflictError("stale")
        value = _decision(
            decision.document_id,
            decision.document_type_id,
            decision.state,
            expected_revision + 1,
        )
        self.decisions.append(value)
        return value

    async def list_document_decisions(self, **kwargs):
        return list(self.decisions)


class FakeContentGroups:
    def __init__(self, value):
        self.value = value
        self.assignment = None
        self.conflict = False

    async def create(self, **kwargs):
        self.create_call = kwargs
        return self.value

    async def get(self, content_group_id, *, include_archived=False):
        return self.value if content_group_id == self.value.id else None

    async def rename(self, **kwargs):
        self.rename_call = kwargs
        return self.value

    async def archive(self, content_group_id):
        self.archive_call = content_group_id
        return self.value

    async def get_assignment(self, document_id):
        return self.assignment

    async def write_assignment(self, assignment, *, expected_revision):
        self.write_call = (assignment, expected_revision)
        if self.conflict:
            raise DocumentOrganizationConflictError("stale")
        self.assignment = _assignment(
            assignment.document_id,
            assignment.content_group_id,
            revision=expected_revision + 1,
        )
        return self.assignment

    async def clear_assignment(self, *, document_id, expected_revision):
        self.clear_call = (document_id, expected_revision)
        if self.conflict:
            raise DocumentOrganizationConflictError("stale")
        self.assignment = _assignment(
            document_id,
            None,
            state=ContentGroupAssignmentState.PENDING,
            revision=expected_revision + 1,
        )
        return self.assignment


class FakeUow:
    def __init__(self, document_id, document_type, group):
        self.documents = SimpleNamespace(
            get=lambda value: _async_value(
                SimpleNamespace(
                    id=document_id,
                    metadata={
                        "organization_classification": {
                            "status": "pending",
                            "summary": "must not cross the API boundary",
                        }
                    },
                )
                if value == document_id
                else None
            )
        )
        self.document_types = FakeDocumentTypes(document_type)
        self.content_groups = FakeContentGroups(group)
        self.commits = 0
        self.rollbacks = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_catalogue_create_update_rename_and_archive_commit() -> None:
    document_id = uuid.uuid4()
    document_type = _document_type()
    group = _group()
    uow = FakeUow(document_id, document_type, group)

    await CreateDocumentTypeHandler(uow=uow)(CreateDocumentTypeCommand("technical-design", "Technical design"))
    await UpdateDocumentTypeHandler(uow=uow)(UpdateDocumentTypeCommand(document_type.id, label="Design", archive=True))
    await CreateContentGroupHandler(uow=uow)(CreateContentGroupCommand("DAF"))
    await UpdateContentGroupHandler(uow=uow)(UpdateContentGroupCommand(group.id, name="DAF project", archive=True))

    assert uow.document_types.update_call[1]["label"] == "Design"
    assert uow.document_types.archive_call == document_type.id
    assert uow.content_groups.rename_call["name"] == "DAF project"
    assert uow.content_groups.archive_call == group.id
    assert uow.commits == 4


@pytest.mark.asyncio
async def test_manual_type_batch_writes_explicit_assigned_and_rejected_with_cas() -> None:
    document_id = uuid.uuid4()
    first = _document_type()
    second = _document_type()
    uow = FakeUow(document_id, first, _group())
    original_get = uow.document_types.get

    async def get_type(value, *, include_archived=False):
        if value == second.id:
            return second
        return await original_get(value, include_archived=include_archived)

    uow.document_types.get = get_type
    result = await ReplaceDocumentTypesHandler(uow=uow)(
        ReplaceDocumentTypesCommand(
            document_id,
            (
                ManualDocumentTypeDecisionInput(first.id, DocumentTypeDecisionState.ASSIGNED, 0),
                ManualDocumentTypeDecisionInput(second.id, DocumentTypeDecisionState.REJECTED, 2),
            ),
        )
    )

    assert [item.state for item in result] == [
        DocumentTypeDecisionState.ASSIGNED,
        DocumentTypeDecisionState.REJECTED,
    ]
    assert [call[1] for call in uow.document_types.write_call] == [0, 2]
    assert all(call[0].source is ClassificationSource.MANUAL for call in uow.document_types.write_call)
    assert uow.commits == 1


@pytest.mark.asyncio
async def test_manual_group_set_and_clear_use_cas_and_clear_to_automatic_pending() -> None:
    document_id = uuid.uuid4()
    group = _group()
    uow = FakeUow(document_id, _document_type(), group)
    handler = SetDocumentContentGroupHandler(uow=uow)

    assigned = await handler(SetDocumentContentGroupCommand(document_id, group.id, 0, "curated"))
    cleared = await handler(SetDocumentContentGroupCommand(document_id, None, assigned.revision))

    assert assigned.source is ClassificationSource.MANUAL
    assert cleared.state is ContentGroupAssignmentState.PENDING
    assert cleared.source is ClassificationSource.AUTOMATIC
    assert cleared.content_group_id is None
    assert uow.content_groups.clear_call == (document_id, 1)


@pytest.mark.asyncio
async def test_manual_group_conflict_returns_current_revision_context() -> None:
    document_id = uuid.uuid4()
    group = _group()
    uow = FakeUow(document_id, _document_type(), group)
    uow.content_groups.assignment = _assignment(document_id, group.id, revision=4)
    uow.content_groups.conflict = True

    with pytest.raises(DocumentOrganizationWriteConflict) as caught:
        await SetDocumentContentGroupHandler(uow=uow)(
            SetDocumentContentGroupCommand(document_id, group.id, 3)
        )

    assert caught.value.current_assignment.revision == 4


@pytest.mark.asyncio
async def test_type_batch_conflict_rolls_back_before_loading_current_context() -> None:
    document_id = uuid.uuid4()
    first = _document_type()
    second = _document_type()
    persisted = _decision(
        document_id,
        first.id,
        DocumentTypeDecisionState.ASSIGNED,
        revision=2,
    )
    uow = FakeUow(document_id, first, _group())
    uow.document_types.decisions = [persisted]
    original_get = uow.document_types.get
    write_count = 0

    async def get_type(value, *, include_archived=False):
        if value == second.id:
            return second
        return await original_get(value, include_archived=include_archived)

    async def write_decision(value, *, expected_revision):
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise DocumentOrganizationConflictError("stale")
        uncommitted = _decision(
            value.document_id,
            value.document_type_id,
            value.state,
            revision=expected_revision + 1,
        )
        uow.document_types.decisions.append(uncommitted)
        return uncommitted

    async def rollback():
        uow.rollbacks += 1
        uow.document_types.decisions = [persisted]

    uow.document_types.get = get_type
    uow.document_types.write_decision = write_decision
    uow.rollback = rollback

    with pytest.raises(DocumentOrganizationWriteConflict) as caught:
        await ReplaceDocumentTypesHandler(uow=uow)(
            ReplaceDocumentTypesCommand(
                document_id,
                (
                    ManualDocumentTypeDecisionInput(
                        first.id,
                        DocumentTypeDecisionState.REJECTED,
                        2,
                    ),
                    ManualDocumentTypeDecisionInput(
                        second.id,
                        DocumentTypeDecisionState.ASSIGNED,
                        0,
                    ),
                ),
            )
        )

    assert uow.rollbacks == 1
    assert caught.value.current_type_decisions == (persisted,)


@pytest.mark.asyncio
async def test_document_organization_read_model_exposes_status_without_document_summary() -> None:
    document_id = uuid.uuid4()
    document_type = _document_type()
    group = _group()
    uow = FakeUow(document_id, document_type, group)
    uow.document_types.decisions = [
        _decision(document_id, document_type.id, DocumentTypeDecisionState.ASSIGNED)
    ]
    uow.content_groups.assignment = _assignment(document_id, group.id)

    view = await GetDocumentOrganizationHandler(uow=uow)(GetDocumentOrganizationQuery(document_id))

    assert view.type_decisions[0].document_type is document_type
    assert view.content_group is group
    assert view.status == {"status": "pending"}
    assert "summary" not in view.status
