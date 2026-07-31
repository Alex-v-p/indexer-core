from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobSubmission,
    BackgroundJobType,
    DocumentVersionStatus,
)
from packages.indexer_application.ports import UnitOfWork


@dataclass(frozen=True, slots=True)
class EnqueueDocumentMaintenanceCommand:
    document_id: uuid.UUID
    contextualization_only: bool = False


@dataclass(frozen=True, slots=True)
class EnqueueEvaluationCommand:
    dataset_path: str
    pipeline_name: str | None = None
    top_k: int | None = None
    repetitions: int = 1


class EnqueueDocumentMaintenanceHandler:
    def __init__(self, *, uow: UnitOfWork, max_attempts: int = 3) -> None:
        self._uow = uow
        self._max_attempts = max_attempts

    async def __call__(self, command: EnqueueDocumentMaintenanceCommand) -> BackgroundJobRecord | None:
        document = await self._uow.documents.get(command.document_id)
        if document is None:
            return None
        ready_versions = [
            version for version in document.versions if version.status is DocumentVersionStatus.READY
        ]
        if not ready_versions:
            raise ValueError("The document has no ready version to process.")
        version = max(ready_versions, key=lambda item: item.version_number)
        job_type = (
            BackgroundJobType.CONTEXTUALIZE_DOCUMENT
            if command.contextualization_only
            else BackgroundJobType.REBUILD_DOCUMENT_INDEX
        )
        job = await self._uow.background_jobs.enqueue(
            BackgroundJobSubmission(
                job_type=job_type,
                payload={
                    "document_id": str(document.id),
                    "document_version_id": str(version.id),
                    "document_version_number": version.version_number,
                },
                max_attempts=self._max_attempts,
                dedupe_key=f"{job_type.value}:{version.id}",
            ),
        )
        await self._uow.commit()
        return job


class EnqueueEvaluationHandler:
    def __init__(self, *, uow: UnitOfWork, max_attempts: int = 2) -> None:
        self._uow = uow
        self._max_attempts = max_attempts

    async def __call__(self, command: EnqueueEvaluationCommand) -> BackgroundJobRecord:
        dataset_path = command.dataset_path.strip()
        if not dataset_path:
            raise ValueError("dataset_path must not be empty.")
        if command.top_k is not None and command.top_k <= 0:
            raise ValueError("top_k must be positive when provided.")
        if command.repetitions <= 0:
            raise ValueError("repetitions must be positive.")
        job = await self._uow.background_jobs.enqueue(
            BackgroundJobSubmission(
                job_type=BackgroundJobType.RUN_EVALUATION,
                payload={
                    "dataset_path": dataset_path,
                    "pipeline_name": command.pipeline_name,
                    "top_k": command.top_k,
                    "repetitions": command.repetitions,
                },
                max_attempts=self._max_attempts,
            ),
        )
        await self._uow.commit()
        return job
