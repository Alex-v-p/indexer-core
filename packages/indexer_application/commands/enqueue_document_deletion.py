from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
)
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class EnqueueDocumentDeletionCommand:
    document_id: uuid.UUID


class EnqueueDocumentDeletionHandler:
    """Queue complete removal of one document and all of its versions."""

    def __init__(self, *, uow: UnitOfWork, max_attempts: int = 3) -> None:
        self._uow = uow
        self._max_attempts = max_attempts

    async def __call__(
        self,
        command: EnqueueDocumentDeletionCommand,
    ) -> BackgroundJobRecord | None:
        document = await self._uow.documents.get(command.document_id)
        if document is None:
            return None

        has_conflicting_work = await self._uow.background_jobs.has_active_for_document(
            document_id=document.id,
            exclude_job_types=(BackgroundJobType.DELETE_DOCUMENT,),
        )
        if has_conflicting_work:
            raise ValueError(
                "The document still has active background work. Wait for it to finish before deleting it."
            )

        job = await self._uow.background_jobs.enqueue(
            BackgroundJobSubmission(
                job_type=BackgroundJobType.DELETE_DOCUMENT,
                payload={
                    "document_id": str(document.id),
                    "document_title": document.title,
                },
                max_attempts=self._max_attempts,
                dedupe_key=f"delete_document:{document.id}",
            ),
        )
        await self._uow.commit()
        return job
