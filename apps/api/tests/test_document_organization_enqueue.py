from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from packages.indexer_application.commands import enqueue_document_organization_classification
from packages.indexer_application.commands import (
    EnqueueDocumentOrganizationClassificationCommand,
    EnqueueDocumentOrganizationClassificationHandler,
)
from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobStatus,
    BackgroundJobType,
    DocumentVersionIdentity,
    DocumentVersionStatus,
)
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
)


class Groups:
    def __init__(self, current=None) -> None:
        self.current = current
        self.writes = []
    async def get_assignment(self, document_id):
        del document_id
        return self.current
    async def write_assignment(self, value):
        self.writes.append(value)


class Jobs:
    def __init__(self) -> None: self.submission = None
    async def enqueue(self, submission):
        self.submission = submission
        now = datetime.now(UTC)
        return BackgroundJobRecord(uuid.uuid4(), submission.job_type, BackgroundJobStatus.QUEUED, 100, submission.payload, {}, 0, None, 0, submission.max_attempts, submission.dedupe_key, now, None, None, None, None, None, None, now, now)


class Documents:
    def __init__(self):
        self.status = None
        self.current = None
    async def set_organization_classification_status(self, **kwargs):
        self.status = kwargs
        return True
    async def get(self, document_id):
        if self.current is not None and self.current.id == document_id:
            return self.current
        return None
    async def list(self, *, limit, offset):
        del limit, offset
        return [self.current] if self.current is not None else []


class Uow:
    def __init__(self, current=None):
        self.content_groups = Groups(current)
        self.background_jobs = Jobs()
        self.documents = Documents()
        self.commits = 0
    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_enqueue_writes_pending_and_new_deduped_job() -> None:
    uow = Uow()
    document_id = uuid.uuid4()
    version = DocumentVersionIdentity(uuid.uuid4(), 2)
    job = await enqueue_document_organization_classification(
        uow=uow, document_id=document_id, version=version,
        policy_version="document-organization-policy/1.0", max_attempts=3,
    )

    pending = uow.content_groups.writes[0]
    assert pending.state is ContentGroupAssignmentState.PENDING
    assert pending.source is ClassificationSource.AUTOMATIC
    assert job.job_type is BackgroundJobType.CLASSIFY_DOCUMENT_ORGANIZATION
    assert uow.background_jobs.submission.dedupe_key.startswith("classify_document_organization:")
    assert uow.documents.status["status"]["status"] == "queued"


@pytest.mark.asyncio
async def test_enqueue_retains_manual_group_assignment() -> None:
    current = type("Current", (), {"source": ClassificationSource.MANUAL})()
    uow = Uow(current)
    await enqueue_document_organization_classification(
        uow=uow, document_id=uuid.uuid4(),
        version=DocumentVersionIdentity(uuid.uuid4(), 1),
        policy_version="document-organization-policy/1.0", max_attempts=3,
    )
    assert uow.content_groups.writes == []


@pytest.mark.asyncio
async def test_explicit_requeue_forces_current_document_while_backfill_skips_it() -> None:
    document_id = uuid.uuid4()
    version = type(
        "Version",
        (),
        {"id": uuid.uuid4(), "version_number": 1, "status": DocumentVersionStatus.READY},
    )()
    current = type(
        "Assignment",
        (),
        {
            "source": ClassificationSource.AUTOMATIC,
            "classified_document_version_id": version.id,
            "policy_version": "document-organization-policy/1.0",
            "classifier_version": "document-organization-classifier/1.0",
            "state": ContentGroupAssignmentState.ASSIGNED,
        },
    )()
    document = type("Document", (), {"id": document_id, "versions": (version,)})()

    backfill_uow = Uow(current)
    backfill_uow.documents.current = document
    backfill = await EnqueueDocumentOrganizationClassificationHandler(uow=backfill_uow)(
        EnqueueDocumentOrganizationClassificationCommand(limit=10),
    )
    assert backfill.jobs == ()
    assert backfill.skipped_document_ids == (document_id,)

    explicit_uow = Uow(current)
    explicit_uow.documents.current = document
    explicit = await EnqueueDocumentOrganizationClassificationHandler(uow=explicit_uow)(
        EnqueueDocumentOrganizationClassificationCommand(document_id=document_id),
    )
    assert len(explicit.jobs) == 1
    assert explicit.skipped_document_ids == ()
