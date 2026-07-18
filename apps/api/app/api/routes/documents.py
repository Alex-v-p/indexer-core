from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.composition import (
    build_chunk_contextualizer,
    build_document_ingestion_config,
    build_document_object_store,
    build_embedding_provider,
    build_keyword_cache_invalidator,
    build_vector_store,
)
from app.core.config import Settings, get_settings
from app.dependencies.database import get_unit_of_work
from app.schemas.documents import (
    ChunkIndexResponse,
    DocumentDetailResponse,
    DocumentSummaryResponse,
    DocumentVersionResponse,
)
from packages.indexer_application.dto import DocumentRecord
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services import IngestionError, get_document, ingest_uploaded_document, list_documents
from packages.rag_core.documents import UnsupportedDocumentTypeError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentDetailResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    detect_existing_versions: bool = Form(default=True),
    published_at: date | None = Form(default=None),
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> DocumentDetailResponse:
    """Upload, parse, chunk, and index a source document."""

    try:
        document = await ingest_uploaded_document(
            uow=uow,
            config=build_document_ingestion_config(settings),
            upload=file,
            object_store=build_document_object_store(settings),
            embedding_provider=build_embedding_provider(settings),
            vector_index=build_vector_store(settings),
            keyword_cache=build_keyword_cache_invalidator(settings),
            contextualizer=build_chunk_contextualizer(settings),
            title=title,
            detect_existing_versions=detect_existing_versions,
            published_at=published_at,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return to_document_detail_response(document)


@router.post(
    "/{document_id}/versions",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_version(
    document_id: uuid.UUID,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    published_at: date | None = Form(default=None),
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> DocumentDetailResponse:
    """Upload a new version for an existing logical document."""

    try:
        document = await ingest_uploaded_document(
            uow=uow,
            config=build_document_ingestion_config(settings),
            upload=file,
            object_store=build_document_object_store(settings),
            embedding_provider=build_embedding_provider(settings),
            vector_index=build_vector_store(settings),
            keyword_cache=build_keyword_cache_invalidator(settings),
            contextualizer=build_chunk_contextualizer(settings),
            title=title,
            version_of_document_id=document_id,
            detect_existing_versions=False,
            published_at=published_at,
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except IngestionError as exc:
        detail = str(exc)
        response_status = status.HTTP_404_NOT_FOUND if "was not found" in detail else status.HTTP_422_UNPROCESSABLE_ENTITY
        raise HTTPException(status_code=response_status, detail=detail) from exc

    return to_document_detail_response(document)


@router.get("", response_model=list[DocumentSummaryResponse])
async def read_documents(
    limit: int = 50,
    offset: int = 0,
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> list[DocumentSummaryResponse]:
    documents = await list_documents(uow=uow, limit=min(limit, 100), offset=max(offset, 0))
    return [to_document_summary_response(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def read_document(
    document_id: uuid.UUID,
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> DocumentDetailResponse:
    document = await get_document(uow=uow, document_id=document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return to_document_detail_response(document)


def to_document_summary_response(document: DocumentRecord) -> DocumentSummaryResponse:
    chunk_count = len(document.chunk_indexes)
    metadata = document.metadata
    return DocumentSummaryResponse(
        id=document.id,
        title=document.title,
        original_filename=document.original_filename,
        content_type=document.content_type,
        storage_uri=document.storage_uri,
        size_bytes=document.size_bytes,
        checksum_sha256=document.checksum_sha256,
        status=document.status.value,
        chunk_count=chunk_count or int(metadata.get("chunk_count", 0) or 0),
        metadata=metadata,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def to_document_detail_response(document: DocumentRecord) -> DocumentDetailResponse:
    summary = to_document_summary_response(document)
    ready_version_numbers = [
        version.version_number
        for version in document.versions
        if version.status.value == "ready"
    ]
    latest_version_number = max(
        ready_version_numbers or [version.version_number for version in document.versions],
        default=None,
    )
    return DocumentDetailResponse(
        **summary.model_dump(),
        versions=[
            DocumentVersionResponse(
                id=version.id,
                version_number=version.version_number,
                storage_uri=version.storage_uri,
                content_type=version.content_type,
                checksum_sha256=version.checksum_sha256,
                parser_name=version.parser_name,
                parser_version=version.parser_version,
                status=version.status.value,
                is_latest=version.version_number == latest_version_number,
                uploaded_at=version.created_at,
                published_at=version.published_at,
                metadata=version.metadata,
                created_at=version.created_at,
                updated_at=version.updated_at,
            )
            for version in document.versions
        ],
        chunks=[
            ChunkIndexResponse(
                id=chunk.id,
                ordinal=chunk.ordinal,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                source_page_start=chunk.source_page_start,
                source_page_end=chunk.source_page_end,
                section_title=chunk.section_title,
                qdrant_collection=chunk.qdrant_collection,
                qdrant_point_id=chunk.qdrant_point_id,
                metadata=chunk.metadata,
                created_at=chunk.created_at,
            )
            for chunk in document.chunk_indexes
        ],
    )
