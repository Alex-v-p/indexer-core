from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from packages.indexer_application.dto import DocumentVersionStatus
from packages.indexer_application.services.background_jobs import (
    ClassifyDocumentOrganizationJobHandler,
    DocumentOrganizationJobConfig,
)
from packages.indexer_application.services.background_jobs.document_organization import (
    _representatives,
)
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentOrganizationPolicy,
)


def _document(*, summary=None):
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    metadata = {}
    if summary is not None:
        metadata = {"hierarchical_retrieval": {"document_summary": summary}}
    version = SimpleNamespace(
        id=version_id,
        version_number=1,
        status=DocumentVersionStatus.READY,
        metadata=metadata,
    )
    return SimpleNamespace(
        id=document_id,
        title="Document",
        original_filename="document.md",
        versions=(version,),
    )


class Documents:
    def __init__(self, documents):
        self.items = {item.id: item for item in documents}
        self.statuses = []
        self.get_calls = []
    async def get(self, document_id):
        self.get_calls.append(document_id)
        return self.items.get(document_id)
    async def get_for_update(self, document_id):
        return self.items.get(document_id)
    async def set_organization_classification_status(self, **kwargs):
        self.statuses.append(kwargs)
        return True


class Types:
    async def list(self, **kwargs):
        del kwargs
        return []
    async def list_document_decisions(self, **kwargs):
        del kwargs
        return []


class Groups:
    def __init__(self, current=None):
        self.current = current
        self.writes = []
        self.assignments = []
    async def get_assignment(self, document_id):
        del document_id
        return self.current
    async def write_assignment(self, assignment):
        self.writes.append(assignment)
    async def list_assignments(self, **kwargs):
        del kwargs
        return self.assignments


class Uow:
    def __init__(self, document, current=None):
        self.documents = Documents([document])
        self.document_types = Types()
        self.content_groups = Groups(current)


async def _report(progress, stage):
    del progress, stage


@pytest.mark.asyncio
async def test_missing_root_summary_becomes_unresolved_without_fabricated_group() -> None:
    document = _document()
    uow = Uow(document)
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=None,
        group_provider=None,
    )
    result = await handler(
        {
            "document_id": str(document.id),
            "document_version_id": str(document.versions[0].id),
            "policy_version": "document-organization-policy/1.0",
        },
        _report,
        job_id=uuid.uuid4(),
    )

    assignment = uow.content_groups.writes[0]
    assert assignment.state is ContentGroupAssignmentState.UNRESOLVED
    assert assignment.content_group_id is None
    assert assignment.unresolved_reason == "no_summary"
    assert assignment.summary_hash is None
    assert "summary" not in str(assignment.signals).lower()
    assert result["content_group_state"] == "unresolved"


@pytest.mark.asyncio
async def test_manual_group_assignment_is_never_overwritten() -> None:
    document = _document()
    current = SimpleNamespace(source=ClassificationSource.MANUAL)
    uow = Uow(document, current)
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=None,
        group_provider=None,
    )
    await handler(
        {"document_id": str(document.id), "document_version_id": str(document.versions[0].id), "policy_version": "document-organization-policy/1.0"},
        _report,
        job_id=uuid.uuid4(),
    )
    assert uow.content_groups.writes == []


@pytest.mark.asyncio
async def test_representatives_exclude_current_document() -> None:
    current = _document(summary="Mobilab plan")
    other = _document(summary="Mobilab care platform delivery")
    uow = Uow(current)
    uow.documents.items[other.id] = other
    uow.content_groups.assignments = [
        SimpleNamespace(document_id=current.id),
        SimpleNamespace(document_id=other.id),
    ]
    group = SimpleNamespace(id=uuid.uuid4())
    candidates = await _representatives(
        uow,
        groups=[group],
        current_document_id=current.id,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
    )

    assert current.id not in uow.documents.get_calls
    assert candidates[0].document_count == 1
    assert "Mobilab care platform" in candidates[0].representative_content
