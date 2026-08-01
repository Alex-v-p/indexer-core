from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
    DocumentRecord,
    DocumentVersionRecord,
    DocumentVersionIdentity,
    DocumentVersionStatus,
)
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class EnqueueDocumentSubjectClassificationCommand:
    document_id: uuid.UUID | None = None
    limit: int = 500


@dataclass(frozen=True, slots=True)
class EnqueueDocumentSubjectClassificationResult:
    jobs: tuple[BackgroundJobRecord, ...]
    skipped_document_ids: tuple[uuid.UUID, ...] = ()


class EnqueueDocumentSubjectClassificationHandler:
    def __init__(
        self,
        *,
        uow: UnitOfWork,
        enabled: bool,
        policy_version: str,
        max_attempts: int = 3,
    ) -> None:
        self._uow = uow
        self._enabled = enabled
        self._policy_version = policy_version
        self._max_attempts = max_attempts

    async def __call__(
        self,
        command: EnqueueDocumentSubjectClassificationCommand,
    ) -> EnqueueDocumentSubjectClassificationResult:
        if not self._enabled:
            raise ValueError("Automatic subject classification is disabled.")
        if command.limit <= 0 or command.limit > 5_000:
            raise ValueError("Classification backfill limit must be between 1 and 5000.")
        if command.document_id is not None:
            document = await self._uow.documents.get(command.document_id)
            if document is None:
                raise LookupError(f"Document {command.document_id} was not found.")
            documents = [document]
        else:
            documents = await _list_documents(self._uow, limit=command.limit)

        jobs: list[BackgroundJobRecord] = []
        skipped: list[uuid.UUID] = []
        for document in documents:
            version = _latest_ready_version(document)
            if version is None:
                skipped.append(document.id)
                continue
            jobs.append(
                await enqueue_document_subject_classification(
                    uow=self._uow,
                    document_id=document.id,
                    version=version,
                    policy_version=self._policy_version,
                    max_attempts=self._max_attempts,
                ),
            )
        await self._uow.commit()
        return EnqueueDocumentSubjectClassificationResult(
            jobs=tuple(jobs),
            skipped_document_ids=tuple(skipped),
        )


async def enqueue_document_subject_classification(
    *,
    uow: UnitOfWork,
    document_id: uuid.UUID,
    version: DocumentVersionRecord | DocumentVersionIdentity,
    policy_version: str,
    max_attempts: int,
) -> BackgroundJobRecord:
    job = await uow.background_jobs.enqueue(
        BackgroundJobSubmission(
            job_type=BackgroundJobType.CLASSIFY_DOCUMENT_SUBJECTS,
            payload={
                "document_id": str(document_id),
                "document_version_id": str(version.id),
                "document_version_number": version.version_number,
                "policy_version": policy_version,
            },
            max_attempts=max_attempts,
            dedupe_key=(
                f"classify_document_subjects:{document_id}:{version.id}:{policy_version}"
            ),
        ),
    )
    await uow.documents.set_subject_classification_status(
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
    return _latest_ready_version(document)


def _latest_ready_version(document: DocumentRecord) -> DocumentVersionRecord | None:
    ready = [
        version
        for version in document.versions
        if version.status is DocumentVersionStatus.READY
    ]
    return max(ready, key=lambda version: version.version_number, default=None)


async def _list_documents(uow: UnitOfWork, *, limit: int) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    offset = 0
    while len(documents) < limit:
        batch = await uow.documents.list(
            limit=min(100, limit - len(documents)),
            offset=offset,
        )
        documents.extend(batch)
        if len(batch) < 100:
            break
        offset += len(batch)
    return documents
