from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from packages.indexer_application.ports import DocumentVersionIndexActivator, UnitOfWork
from packages.indexer_application.services.ingestion.contextualize import ContextualizedDocumentContent
from packages.indexer_application.services.ingestion.index import IndexedDocument
from packages.indexer_application.services.ingestion.parse import ParsedDocumentContent
from packages.indexer_application.services.ingestion.prepare import PreparedDocument


@dataclass(frozen=True, slots=True)
class ActivateDocumentInput:
    prepared: PreparedDocument
    parsed: ParsedDocumentContent
    contextualized: ContextualizedDocumentContent
    indexed: IndexedDocument


async def activate_document(
    *,
    request: ActivateDocumentInput,
    uow: UnitOfWork,
    version_index: DocumentVersionIndexActivator,
) -> None:
    """Activate the indexed version and commit its ready application state."""

    prepared = request.prepared
    parsed = request.parsed
    contextualized = request.contextualized
    indexed = request.indexed
    version = prepared.version

    await version_index.activate_document_version(
        document_id=prepared.document_id,
        version_id=version.id,
    )
    await uow.documents.set_version_parser_metadata(
        version_id=version.id,
        parser_name=parsed.parsed_document.parser_name,
        parser_version=parsed.parsed_document.parser_version,
        metadata={
            **parsed.parsed_document.metadata,
            "chunk_count": len(parsed.chunks),
            "contextualization": contextualized.contextualization_metadata,
            "hierarchical_retrieval": indexed.hierarchical_retrieval_metadata,
            "document_version_number": version.version_number,
            "uploaded_at": _isoformat(version.uploaded_at),
            "published_at": _isoformat(version.published_at),
            "version_detection": {
                "method": prepared.version_detection_method,
                "matched_existing_document": prepared.matched_existing_document,
            },
        },
    )
    await uow.documents.mark_ready(
        document_id=prepared.document_id,
        version_id=version.id,
        document_metadata={
            "chunk_count": len(parsed.chunks),
            "parser_name": parsed.parsed_document.parser_name,
            "parser_version": parsed.parsed_document.parser_version,
            "contextualization": contextualized.contextualization_metadata,
            "hierarchical_retrieval": indexed.hierarchical_retrieval_metadata,
            "latest_version_id": str(version.id),
            "latest_version_number": version.version_number,
            "version_detection_method": prepared.version_detection_method,
        },
        stored_file=prepared.stored_document,
        version_number=version.version_number,
    )
    await uow.commit()


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
