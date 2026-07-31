from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
)
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class DocumentVersionDeletionTarget:
    document_id: uuid.UUID
    version_id: uuid.UUID


@dataclass(frozen=True, slots=True)
class EnqueueDocumentVersionDeletionCommand:
    targets: tuple[DocumentVersionDeletionTarget, ...]


class EnqueueDocumentVersionDeletionHandler:
    """Queue ordered removal of one or more explicit document versions."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        max_attempts: int = 3,
        max_targets: int = 100,
    ) -> None:
        self._uow = uow
        self._max_attempts = max_attempts
        self._max_targets = max_targets

    async def __call__(
        self,
        command: EnqueueDocumentVersionDeletionCommand,
    ) -> BackgroundJobRecord:
        targets = _unique_targets(command.targets)
        if not targets:
            raise ValueError("At least one document version must be selected for deletion.")
        if len(targets) > self._max_targets:
            raise ValueError(
                f"At most {self._max_targets} document versions can be deleted in one operation."
            )

        documents = {}
        payload_targets: list[dict[str, object]] = []
        for target in targets:
            document = documents.get(target.document_id)
            if document is None:
                document = await self._uow.documents.get(target.document_id)
                if document is None:
                    raise LookupError(f"Document {target.document_id} was not found.")
                documents[target.document_id] = document

            version = next((item for item in document.versions if item.id == target.version_id), None)
            if version is None:
                raise LookupError(
                    f"Document version {target.version_id} was not found on document {target.document_id}."
                )
            payload_targets.append(
                {
                    "document_id": str(document.id),
                    "document_title": document.title,
                    "document_version_id": str(version.id),
                    "document_version_number": version.version_number,
                }
            )

        for document_id in documents:
            if await self._uow.background_jobs.has_active_for_document(document_id=document_id):
                raise ValueError(
                    "One or more selected documents still have active background work. "
                    "Wait for it to finish before deleting versions."
                )

        job = await self._uow.background_jobs.enqueue(
            BackgroundJobSubmission(
                job_type=BackgroundJobType.DELETE_DOCUMENT_VERSIONS,
                payload={"targets": payload_targets},
                max_attempts=self._max_attempts,
                dedupe_key=_dedupe_key(targets),
            ),
        )
        await self._uow.commit()
        return job


def _unique_targets(
    targets: tuple[DocumentVersionDeletionTarget, ...],
) -> tuple[DocumentVersionDeletionTarget, ...]:
    unique: dict[tuple[uuid.UUID, uuid.UUID], DocumentVersionDeletionTarget] = {}
    for target in targets:
        unique.setdefault((target.document_id, target.version_id), target)
    return tuple(unique.values())


def _dedupe_key(targets: tuple[DocumentVersionDeletionTarget, ...]) -> str:
    identity = "|".join(
        sorted(f"{target.document_id}:{target.version_id}" for target in targets)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"delete_document_versions:{digest}"
