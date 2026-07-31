from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status

from app.dependencies.application import (
    get_document_handler,
    get_enqueue_document_version_deletion_handler,
    get_enqueue_document_maintenance_handler,
    get_list_documents_handler,
    get_submit_document_ingestion_handler,
)
from app.schemas.documents import (
    BatchDocumentUploadResponse,
    BatchDocumentVersionDeletionRequest,
    ChunkIndexResponse,
    DocumentDetailResponse,
    DocumentSummaryResponse,
    DocumentVersionResponse,
    QueuedDocumentUploadItemResponse,
    QueuedDocumentVersionDeletionResponse,
    RejectedDocumentUploadResponse,
)
from packages.indexer_application.commands import (
    DocumentVersionDeletionTarget,
    EnqueueDocumentVersionDeletionCommand,
    EnqueueDocumentVersionDeletionHandler,
    EnqueueDocumentMaintenanceCommand,
    EnqueueDocumentMaintenanceHandler,
    SubmitDocumentIngestionCommand,
    SubmitDocumentIngestionHandler,
)
from packages.indexer_application.dto import DocumentRecord
from packages.indexer_application.services.ingestion import IngestionError
from packages.indexer_application.queries import (
    GetDocumentHandler,
    GetDocumentQuery,
    ListDocumentsHandler,
    ListDocumentsQuery,
)
from packages.rag_core.documents import UnsupportedDocumentTypeError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentDetailResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    detect_existing_versions: bool = Form(default=True),
    published_at: date | None = Form(default=None),
    handler: SubmitDocumentIngestionHandler = Depends(get_submit_document_ingestion_handler),
) -> DocumentDetailResponse:
    """Persist an upload and enqueue parsing, contextualization, and indexing."""

    try:
        result = await handler(
            SubmitDocumentIngestionCommand(
                upload=file,
                title=title,
                detect_existing_versions=detect_existing_versions,
                published_at=published_at,
            ),
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    response.headers["Location"] = str(request.url_for("get_background_job", job_id=str(result.job.id)))
    response.headers["X-Background-Job-ID"] = str(result.job.id)
    return to_document_detail_response(result.document)


@router.post(
    "/{document_id}/versions",
    response_model=DocumentDetailResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_version(
    document_id: uuid.UUID,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    published_at: date | None = Form(default=None),
    handler: SubmitDocumentIngestionHandler = Depends(get_submit_document_ingestion_handler),
) -> DocumentDetailResponse:
    """Persist a new source version and enqueue its ingestion."""

    try:
        result = await handler(
            SubmitDocumentIngestionCommand(
                upload=file,
                title=title,
                version_of_document_id=document_id,
                detect_existing_versions=False,
                published_at=published_at,
            ),
        )
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except IngestionError as exc:
        detail = str(exc)
        response_status = status.HTTP_404_NOT_FOUND if "was not found" in detail else status.HTTP_422_UNPROCESSABLE_ENTITY
        raise HTTPException(status_code=response_status, detail=detail) from exc

    response.headers["Location"] = str(request.url_for("get_background_job", job_id=str(result.job.id)))
    response.headers["X-Background-Job-ID"] = str(result.job.id)
    return to_document_detail_response(result.document)


@router.post(
    "/batch",
    response_model=BatchDocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
    title: str | None = Form(default=None),
    detect_existing_versions: bool = Form(default=True),
    version_of_document_id: uuid.UUID | None = Form(default=None),
    published_at: date | None = Form(default=None),
    handler: SubmitDocumentIngestionHandler = Depends(get_submit_document_ingestion_handler),
) -> BatchDocumentUploadResponse:
    """Persist and independently enqueue up to 50 uploaded source files."""

    if not files:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No files were uploaded.")
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="At most 50 files can be uploaded in one batch.",
        )

    accepted: list[QueuedDocumentUploadItemResponse] = []
    rejected: list[RejectedDocumentUploadResponse] = []
    for file in files:
        filename = file.filename or "document"
        try:
            result = await handler(
                SubmitDocumentIngestionCommand(
                    upload=file,
                    title=title if len(files) == 1 else None,
                    version_of_document_id=version_of_document_id,
                    detect_existing_versions=(
                        False if version_of_document_id is not None else detect_existing_versions
                    ),
                    published_at=published_at,
                )
            )
        except (UnsupportedDocumentTypeError, IngestionError) as exc:
            rejected.append(RejectedDocumentUploadResponse(filename=filename, detail=str(exc)))
            continue
        accepted.append(
            QueuedDocumentUploadItemResponse(
                filename=filename,
                document=to_document_detail_response(result.document),
                job_id=result.job.id,
            )
        )

    return BatchDocumentUploadResponse(accepted=accepted, rejected=rejected)


@router.delete(
    "/{document_id}/versions/{version_id}",
    response_model=QueuedDocumentVersionDeletionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_document_version(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    request: Request,
    response: Response,
    handler: EnqueueDocumentVersionDeletionHandler = Depends(
        get_enqueue_document_version_deletion_handler
    ),
) -> QueuedDocumentVersionDeletionResponse:
    try:
        job = await handler(
            EnqueueDocumentVersionDeletionCommand(
                targets=(
                    DocumentVersionDeletionTarget(
                        document_id=document_id,
                        version_id=version_id,
                    ),
                )
            )
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    response.headers["Location"] = str(request.url_for("get_background_job", job_id=str(job.id)))
    response.headers["X-Background-Job-ID"] = str(job.id)
    return QueuedDocumentVersionDeletionResponse(
        job_id=job.id,
        status=job.status.value,
        target_count=1,
    )


@router.post(
    "/versions/batch-delete",
    response_model=QueuedDocumentVersionDeletionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_document_versions_batch(
    body: BatchDocumentVersionDeletionRequest,
    request: Request,
    response: Response,
    handler: EnqueueDocumentVersionDeletionHandler = Depends(
        get_enqueue_document_version_deletion_handler
    ),
) -> QueuedDocumentVersionDeletionResponse:
    targets = tuple(
        DocumentVersionDeletionTarget(
            document_id=target.document_id,
            version_id=target.document_version_id,
        )
        for target in body.targets
    )
    try:
        job = await handler(EnqueueDocumentVersionDeletionCommand(targets=targets))
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    response.headers["Location"] = str(request.url_for("get_background_job", job_id=str(job.id)))
    response.headers["X-Background-Job-ID"] = str(job.id)
    return QueuedDocumentVersionDeletionResponse(
        job_id=job.id,
        status=job.status.value,
        target_count=len(targets),
    )


@router.post(
    "/{document_id}/rebuild-index",
    response_model=dict[str, str],
    status_code=status.HTTP_202_ACCEPTED,
)
async def rebuild_document_index(
    document_id: uuid.UUID,
    request: Request,
    response: Response,
    handler: EnqueueDocumentMaintenanceHandler = Depends(get_enqueue_document_maintenance_handler),
) -> dict[str, str]:
    try:
        job = await handler(EnqueueDocumentMaintenanceCommand(document_id=document_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    response.headers["Location"] = str(request.url_for("get_background_job", job_id=str(job.id)))
    return {"job_id": str(job.id), "status": job.status.value}


@router.post(
    "/{document_id}/contextualize",
    response_model=dict[str, str],
    status_code=status.HTTP_202_ACCEPTED,
)
async def contextualize_document(
    document_id: uuid.UUID,
    request: Request,
    response: Response,
    handler: EnqueueDocumentMaintenanceHandler = Depends(get_enqueue_document_maintenance_handler),
) -> dict[str, str]:
    try:
        job = await handler(
            EnqueueDocumentMaintenanceCommand(
                document_id=document_id,
                contextualization_only=True,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    response.headers["Location"] = str(request.url_for("get_background_job", job_id=str(job.id)))
    return {"job_id": str(job.id), "status": job.status.value}


@router.get("", response_model=list[DocumentSummaryResponse])
async def read_documents(
    limit: int = 50,
    offset: int = 0,
    handler: ListDocumentsHandler = Depends(get_list_documents_handler),
) -> list[DocumentSummaryResponse]:
    documents = await handler(
        ListDocumentsQuery(limit=min(limit, 100), offset=max(offset, 0)),
    )
    return [to_document_summary_response(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def read_document(
    document_id: uuid.UUID,
    handler: GetDocumentHandler = Depends(get_document_handler),
) -> DocumentDetailResponse:
    document = await handler(GetDocumentQuery(document_id=document_id))
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
