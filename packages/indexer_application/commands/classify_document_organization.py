from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
    DocumentRecord,
    DocumentVersionIdentity,
    DocumentVersionRecord,
    DocumentVersionStatus,
)
from packages.indexer_application.ports import DocumentOrganizationConflictError, UnitOfWork
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentContentGroupAssignment,
    ORGANIZATION_CLASSIFIER_VERSION,
    ORGANIZATION_POLICY_VERSION,
)


@dataclass(frozen=True, slots=True)
class EnqueueDocumentOrganizationClassificationCommand:
    document_id: uuid.UUID | None = None
    limit: int = 500


@dataclass(frozen=True, slots=True)
class EnqueueDocumentOrganizationClassificationResult:
    jobs: tuple[BackgroundJobRecord, ...]
    skipped_document_ids: tuple[uuid.UUID, ...] = ()


class EnqueueDocumentOrganizationClassificationHandler:
    """Queue one document or a bounded backfill under the current policy."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        enabled: bool = True,
        policy_version: str = ORGANIZATION_POLICY_VERSION,
        classifier_version: str = ORGANIZATION_CLASSIFIER_VERSION,
        max_attempts: int = 3,
    ) -> None:
        self._uow = uow
        self._enabled = enabled
        self._policy_version = policy_version
        self._classifier_version = classifier_version
        self._max_attempts = max_attempts

    async def __call__(
        self,
        command: EnqueueDocumentOrganizationClassificationCommand,
    ) -> EnqueueDocumentOrganizationClassificationResult:
        if not self._enabled:
            raise ValueError("Automatic document organization classification is disabled.")
        if not 1 <= command.limit <= 5_000:
            raise ValueError("Organization backfill limit must be between 1 and 5000.")
        if command.document_id is not None:
            document = await self._uow.documents.get(command.document_id)
            if document is None:
                raise LookupError(f"Document {command.document_id} was not found.")
            documents = [document]
        else:
            documents = await _list_documents(self._uow, command.limit)
        is_backfill = command.document_id is None
        jobs: list[BackgroundJobRecord] = []
        skipped: list[uuid.UUID] = []
        for document in documents:
            version = latest_ready_version(document)
            if version is None:
                skipped.append(document.id)
                continue
            current = await self._uow.content_groups.get_assignment(document.id)
            if (
                is_backfill
                and current is not None
                and current.source is ClassificationSource.AUTOMATIC
                and current.classified_document_version_id == version.id
                and current.policy_version == self._policy_version
                and current.classifier_version == self._classifier_version
                and current.state is not ContentGroupAssignmentState.PENDING
            ):
                skipped.append(document.id)
                continue
            jobs.append(
                await enqueue_document_organization_classification(
                    uow=self._uow,
                    document_id=document.id,
                    version=version,
                    policy_version=self._policy_version,
                    max_attempts=self._max_attempts,
                )
            )
        await self._uow.commit()
        return EnqueueDocumentOrganizationClassificationResult(tuple(jobs), tuple(skipped))


async def enqueue_document_organization_classification(
    *,
    uow: UnitOfWork,
    document_id: uuid.UUID,
    version: DocumentVersionRecord | DocumentVersionIdentity,
    policy_version: str,
    max_attempts: int,
) -> BackgroundJobRecord:
    current = await uow.content_groups.get_assignment(document_id)
    if current is None or current.source is not ClassificationSource.MANUAL:
        try:
            await uow.content_groups.write_assignment(
                DocumentContentGroupAssignment(
                    document_id=document_id,
                    content_group_id=None,
                    state=ContentGroupAssignmentState.PENDING,
                    source=ClassificationSource.AUTOMATIC,
                )
            )
        except DocumentOrganizationConflictError:
            # A concurrent manual assignment is authoritative; the job may still
            # classify types but must not replace that group assignment.
            pass
    job = await uow.background_jobs.enqueue(
        BackgroundJobSubmission(
            job_type=BackgroundJobType.CLASSIFY_DOCUMENT_ORGANIZATION,
            payload={
                "document_id": str(document_id),
                "document_version_id": str(version.id),
                "document_version_number": version.version_number,
                "policy_version": policy_version,
            },
            max_attempts=max_attempts,
            dedupe_key=f"classify_document_organization:{document_id}:{version.id}:{policy_version}",
        )
    )
    await uow.documents.set_organization_classification_status(
        document_id=document_id,
        status={
            "status": job.status.value,
            "job_id": str(job.id),
            "document_version_id": str(version.id),
            "policy_version": policy_version,
        },
    )
    return job


def latest_ready_version(document: DocumentRecord) -> DocumentVersionRecord | None:
    return max(
        (version for version in document.versions if version.status is DocumentVersionStatus.READY),
        key=lambda version: version.version_number,
        default=None,
    )


async def _list_documents(uow: UnitOfWork, limit: int) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    offset = 0
    while len(documents) < limit:
        batch = await uow.documents.list(limit=min(100, limit - len(documents)), offset=offset)
        documents.extend(batch)
        if len(batch) < 100:
            break
        offset += len(batch)
    return documents
