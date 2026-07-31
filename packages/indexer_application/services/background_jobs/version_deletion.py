from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import DocumentRecord, DocumentVersionRecord, DocumentVersionStatus
from packages.indexer_application.ports import (
    CacheInvalidator,
    DocumentObjectStore,
    DocumentVersionIndexActivator,
    DocumentVersionIndexCleaner,
    UnitOfWork,
)
from packages.indexer_application.services.background_jobs.execution import ProgressReporter
from packages.indexer_application.services.background_jobs.payloads import stored_document_from_record


@dataclass(frozen=True, slots=True)
class _DeletionTarget:
    document_id: uuid.UUID
    version_id: uuid.UUID


class DeleteDocumentVersionsJobHandler:
    """Idempotently remove explicit source versions from every owned persistence system."""

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        object_store: DocumentObjectStore,
        document_index: DocumentVersionIndexCleaner,
        version_index: DocumentVersionIndexActivator,
        keyword_cache: CacheInvalidator,
    ) -> None:
        self._uow = uow
        self._object_store = object_store
        self._document_index = document_index
        self._version_index = version_index
        self._keyword_cache = keyword_cache

    async def __call__(
        self,
        payload: dict[str, object],
        report: ProgressReporter,
    ) -> dict[str, object]:
        targets = _targets(payload)
        deleted_versions: list[dict[str, object]] = []
        already_deleted: list[dict[str, str]] = []
        deleted_document_ids: set[str] = set()
        promoted_versions: list[dict[str, object]] = []

        documents = {
            document_id: await self._uow.documents.get(document_id)
            for document_id in dict.fromkeys(target.document_id for target in targets)
        }
        total = len(targets)
        for index, target in enumerate(targets):
            base = index / total
            span = 1 / total
            await report(_progress(base, span, 0.05), "loading_document_version_for_deletion")
            document = documents.get(target.document_id)
            if document is None:
                already_deleted.append(_target_result(target))
                continue
            version = _find_version(document, target.version_id)
            if version is None:
                already_deleted.append(_target_result(target))
                continue

            reference = stored_document_from_record(document=document, version=version)
            await report(_progress(base, span, 0.22), "removing_version_vectors")
            await self._document_index.delete_document_version(
                document_id=document.id,
                version_id=version.id,
            )

            await report(_progress(base, span, 0.45), "removing_version_source_file")
            await self._object_store.delete(reference)

            await report(_progress(base, span, 0.66), "updating_document_version_records")
            outcome = await self._uow.documents.delete_version(
                document_id=document.id,
                version_id=version.id,
            )
            if outcome is None or not outcome.deleted:
                already_deleted.append(_target_result(target))
                continue

            if outcome.document_deleted:
                deleted_document_ids.add(str(document.id))
            elif outcome.promoted_version_id is not None:
                await report(_progress(base, span, 0.82), "promoting_remaining_document_version")
                await self._version_index.activate_document_version(
                    document_id=document.id,
                    version_id=outcome.promoted_version_id,
                )
                promoted_versions.append(
                    {
                        "document_id": str(document.id),
                        "document_version_id": str(outcome.promoted_version_id),
                        "document_version_number": outcome.promoted_version_number,
                    }
                )

            deleted_versions.append(
                {
                    "document_id": str(document.id),
                    "document_title": document.title,
                    "document_version_id": str(version.id),
                    "document_version_number": version.version_number,
                    "storage_uri": reference.storage_uri,
                }
            )

        await report(0.96, "invalidating_keyword_index")
        self._keyword_cache.invalidate()
        await report(0.99, "document_version_deletion_complete")
        return {
            "requested_version_count": total,
            "deleted_version_count": len(deleted_versions),
            "already_deleted_version_count": len(already_deleted),
            "deleted_versions": deleted_versions,
            "already_deleted_versions": already_deleted,
            "deleted_document_ids": sorted(deleted_document_ids),
            "promoted_versions": promoted_versions,
        }


def _targets(payload: dict[str, object]) -> tuple[_DeletionTarget, ...]:
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Deletion job targets must be a non-empty list.")
    parsed: list[_DeletionTarget] = []
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise ValueError("Each deletion target must be an object.")
        parsed.append(
            _DeletionTarget(
                document_id=uuid.UUID(_required_string(raw, "document_id")),
                version_id=uuid.UUID(_required_string(raw, "document_version_id")),
            )
        )
    return tuple(parsed)


def _find_version(document: DocumentRecord, version_id: uuid.UUID) -> DocumentVersionRecord | None:
    return next((version for version in document.versions if version.id == version_id), None)


def _target_result(target: _DeletionTarget) -> dict[str, str]:
    return {
        "document_id": str(target.document_id),
        "document_version_id": str(target.version_id),
    }


def _progress(base: float, span: float, within_target: float) -> float:
    return min(0.94, 0.02 + (base + span * within_target) * 0.92)


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
