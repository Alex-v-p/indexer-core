from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies.application import (
    get_add_content_group_alias_handler,
    get_archive_content_group_alias_handler,
    get_content_group_handler,
    get_content_groups_handler,
    get_create_content_group_handler,
    get_create_document_type_handler,
    get_document_organization_handler,
    get_document_type_handler,
    get_document_types_handler,
    get_enqueue_document_organization_classification_handler,
    get_replace_document_types_handler,
    get_set_document_content_group_handler,
    get_update_content_group_handler,
    get_update_document_type_handler,
)
from app.schemas.document_organization import (
    AddContentGroupAliasRequest,
    ContentGroupAliasResponse,
    ContentGroupResponse,
    CreateContentGroupRequest,
    CreateDocumentTypeRequest,
    DocumentContentGroupAssignmentResponse,
    DocumentOrganizationConflictErrorResponse,
    DocumentOrganizationConflictResponse,
    DocumentOrganizationResponse,
    DocumentTypeDecisionResponse,
    DocumentTypeDecisionViewResponse,
    DocumentTypeResponse,
    OrganizationEnqueueResponse,
    ReplaceDocumentTypesRequest,
    ReplaceDocumentTypesResponse,
    SetDocumentContentGroupRequest,
    UpdateContentGroupRequest,
    UpdateDocumentTypeRequest,
)
from packages.indexer_application.commands import (
    AddContentGroupAliasCommand,
    AddContentGroupAliasHandler,
    ArchivedDocumentOrganizationMutationError,
    ArchiveContentGroupAliasCommand,
    ArchiveContentGroupAliasHandler,
    CreateContentGroupCommand,
    CreateContentGroupHandler,
    CreateDocumentTypeCommand,
    CreateDocumentTypeHandler,
    DocumentOrganizationWriteConflict,
    EnqueueDocumentOrganizationClassificationCommand,
    EnqueueDocumentOrganizationClassificationHandler,
    ManualDocumentTypeDecisionInput,
    ReplaceDocumentTypesCommand,
    ReplaceDocumentTypesHandler,
    SetDocumentContentGroupCommand,
    SetDocumentContentGroupHandler,
    UpdateContentGroupCommand,
    UpdateContentGroupHandler,
    UpdateDocumentTypeCommand,
    UpdateDocumentTypeHandler,
)
from packages.indexer_application.dto import (
    ContentGroupAliasRecord,
    ContentGroupRecord,
    DocumentContentGroupAssignmentRecord,
    DocumentTypeDecisionRecord,
    DocumentTypeRecord,
)
from packages.indexer_application.ports import (
    ContentGroupInUseError,
    ContentGroupNameConflictError,
    DocumentTypeKeyConflictError,
)
from packages.indexer_application.queries import (
    GetContentGroupHandler,
    GetContentGroupQuery,
    GetDocumentOrganizationHandler,
    GetDocumentOrganizationQuery,
    GetDocumentTypeHandler,
    GetDocumentTypeQuery,
    ListContentGroupsHandler,
    ListContentGroupsQuery,
    ListDocumentTypesHandler,
    ListDocumentTypesQuery,
)
from packages.rag_core.document_organization import (
    DocumentTypeDecisionState,
    InvalidContentGroupNameError,
    InvalidDocumentTypeKeyError,
)

router = APIRouter(tags=["document-organization"])


@router.post("/document-types", response_model=DocumentTypeResponse, status_code=201)
async def create_document_type(
    body: CreateDocumentTypeRequest,
    handler: CreateDocumentTypeHandler = Depends(get_create_document_type_handler),
) -> DocumentTypeResponse:
    try:
        value = await handler(CreateDocumentTypeCommand(**body.model_dump()))
    except DocumentTypeKeyConflictError as exc:
        _conflict(exc)
    except (InvalidDocumentTypeKeyError, ValueError) as exc:
        _unprocessable(exc)
    return _type_response(value)


@router.get("/document-types", response_model=list[DocumentTypeResponse])
async def list_document_types(
    include_archived: bool = False,
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    handler: ListDocumentTypesHandler = Depends(get_document_types_handler),
) -> list[DocumentTypeResponse]:
    values = await handler(ListDocumentTypesQuery(include_archived, limit, offset))
    return [_type_response(value) for value in values]


@router.get("/document-types/{document_type_id}", response_model=DocumentTypeResponse)
async def get_document_type(
    document_type_id: uuid.UUID,
    include_archived: bool = False,
    handler: GetDocumentTypeHandler = Depends(get_document_type_handler),
) -> DocumentTypeResponse:
    value = await handler(GetDocumentTypeQuery(document_type_id, include_archived))
    if value is None:
        raise HTTPException(status_code=404, detail="Document type not found.")
    return _type_response(value)


@router.patch("/document-types/{document_type_id}", response_model=DocumentTypeResponse)
async def update_document_type(
    document_type_id: uuid.UUID,
    body: UpdateDocumentTypeRequest,
    handler: UpdateDocumentTypeHandler = Depends(get_update_document_type_handler),
) -> DocumentTypeResponse:
    try:
        value = await handler(UpdateDocumentTypeCommand(document_type_id, **body.model_dump()))
    except LookupError as exc:
        _not_found(exc)
    except (DocumentTypeKeyConflictError, ArchivedDocumentOrganizationMutationError) as exc:
        _conflict(exc)
    except (InvalidDocumentTypeKeyError, ValueError) as exc:
        _unprocessable(exc)
    return _type_response(value)


@router.post("/content-groups", response_model=ContentGroupResponse, status_code=201)
async def create_content_group(
    body: CreateContentGroupRequest,
    handler: CreateContentGroupHandler = Depends(get_create_content_group_handler),
) -> ContentGroupResponse:
    try:
        value = await handler(CreateContentGroupCommand(**body.model_dump()))
    except ContentGroupNameConflictError as exc:
        _conflict(exc)
    except (InvalidContentGroupNameError, ValueError) as exc:
        _unprocessable(exc)
    return _group_response(value)


@router.get("/content-groups", response_model=list[ContentGroupResponse])
async def list_content_groups(
    include_archived: bool = False,
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    handler: ListContentGroupsHandler = Depends(get_content_groups_handler),
) -> list[ContentGroupResponse]:
    values = await handler(ListContentGroupsQuery(include_archived, limit, offset))
    return [_group_response(value) for value in values]


@router.get("/content-groups/{content_group_id}", response_model=ContentGroupResponse)
async def get_content_group(
    content_group_id: uuid.UUID,
    include_archived: bool = False,
    handler: GetContentGroupHandler = Depends(get_content_group_handler),
) -> ContentGroupResponse:
    value = await handler(GetContentGroupQuery(content_group_id, include_archived))
    if value is None:
        raise HTTPException(status_code=404, detail="Content group not found.")
    return _group_response(value)


@router.patch("/content-groups/{content_group_id}", response_model=ContentGroupResponse)
async def update_content_group(
    content_group_id: uuid.UUID,
    body: UpdateContentGroupRequest,
    handler: UpdateContentGroupHandler = Depends(get_update_content_group_handler),
) -> ContentGroupResponse:
    try:
        value = await handler(UpdateContentGroupCommand(content_group_id, **body.model_dump()))
    except LookupError as exc:
        _not_found(exc)
    except (
        ContentGroupNameConflictError,
        ContentGroupInUseError,
        ArchivedDocumentOrganizationMutationError,
    ) as exc:
        _conflict(exc)
    except (InvalidContentGroupNameError, ValueError) as exc:
        _unprocessable(exc)
    return _group_response(value)


@router.post(
    "/content-groups/{content_group_id}/aliases",
    response_model=ContentGroupAliasResponse,
    status_code=201,
)
async def add_content_group_alias(
    content_group_id: uuid.UUID,
    body: AddContentGroupAliasRequest,
    handler: AddContentGroupAliasHandler = Depends(get_add_content_group_alias_handler),
) -> ContentGroupAliasResponse:
    try:
        value = await handler(AddContentGroupAliasCommand(content_group_id, body.name))
    except LookupError as exc:
        _not_found(exc)
    except (ContentGroupNameConflictError, ArchivedDocumentOrganizationMutationError) as exc:
        _conflict(exc)
    except (InvalidContentGroupNameError, ValueError) as exc:
        _unprocessable(exc)
    return _alias_response(value)


@router.delete(
    "/content-groups/{content_group_id}/aliases/{alias_id}",
    response_model=ContentGroupAliasResponse,
)
async def archive_content_group_alias(
    content_group_id: uuid.UUID,
    alias_id: uuid.UUID,
    handler: ArchiveContentGroupAliasHandler = Depends(get_archive_content_group_alias_handler),
) -> ContentGroupAliasResponse:
    try:
        value = await handler(ArchiveContentGroupAliasCommand(content_group_id, alias_id))
    except LookupError as exc:
        _not_found(exc)
    except ArchivedDocumentOrganizationMutationError as exc:
        _conflict(exc)
    except ValueError as exc:
        _unprocessable(exc)
    return _alias_response(value)


@router.get(
    "/documents/{document_id}/organization",
    response_model=DocumentOrganizationResponse,
)
async def get_document_organization(
    document_id: uuid.UUID,
    handler: GetDocumentOrganizationHandler = Depends(get_document_organization_handler),
) -> DocumentOrganizationResponse:
    try:
        view = await handler(GetDocumentOrganizationQuery(document_id))
    except LookupError as exc:
        _not_found(exc)
    return DocumentOrganizationResponse(
        document_id=view.document_id,
        type_decisions=[
            DocumentTypeDecisionViewResponse(
                decision=_decision_response(item.decision),
                document_type=_type_response(item.document_type),
            )
            for item in view.type_decisions
        ],
        content_group_assignment=_assignment_response(view.content_group_assignment),
        content_group=_group_response(view.content_group) if view.content_group else None,
        status=view.status,
    )


@router.put(
    "/documents/{document_id}/organization/types",
    response_model=ReplaceDocumentTypesResponse,
    responses={409: {"model": DocumentOrganizationConflictErrorResponse}},
)
async def replace_document_types(
    document_id: uuid.UUID,
    body: ReplaceDocumentTypesRequest,
    handler: ReplaceDocumentTypesHandler = Depends(get_replace_document_types_handler),
) -> ReplaceDocumentTypesResponse:
    try:
        values = await handler(
            ReplaceDocumentTypesCommand(
                document_id=document_id,
                decisions=tuple(
                    ManualDocumentTypeDecisionInput(
                        document_type_id=item.document_type_id,
                        state=DocumentTypeDecisionState(item.state),
                        expected_revision=item.expected_revision,
                        rationale=item.rationale,
                    )
                    for item in body.decisions
                ),
            )
        )
    except DocumentOrganizationWriteConflict as exc:
        _write_conflict(exc)
    except LookupError as exc:
        _not_found(exc)
    except ArchivedDocumentOrganizationMutationError as exc:
        _conflict(exc)
    except ValueError as exc:
        _unprocessable(exc)
    return ReplaceDocumentTypesResponse(decisions=[_decision_response(value) for value in values])


@router.put(
    "/documents/{document_id}/organization/content-group",
    response_model=DocumentContentGroupAssignmentResponse | None,
    responses={409: {"model": DocumentOrganizationConflictErrorResponse}},
)
async def set_document_content_group(
    document_id: uuid.UUID,
    body: SetDocumentContentGroupRequest,
    handler: SetDocumentContentGroupHandler = Depends(get_set_document_content_group_handler),
) -> DocumentContentGroupAssignmentResponse | None:
    try:
        value = await handler(
            SetDocumentContentGroupCommand(
                document_id=document_id,
                content_group_id=body.content_group_id,
                expected_revision=body.expected_revision,
                rationale=body.rationale,
            )
        )
    except DocumentOrganizationWriteConflict as exc:
        _write_conflict(exc)
    except LookupError as exc:
        _not_found(exc)
    except ArchivedDocumentOrganizationMutationError as exc:
        _conflict(exc)
    except ValueError as exc:
        _unprocessable(exc)
    return _assignment_response(value)


@router.post(
    "/documents/{document_id}/organization/requeue",
    response_model=OrganizationEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def requeue_document_organization(
    document_id: uuid.UUID,
    handler: EnqueueDocumentOrganizationClassificationHandler = Depends(
        get_enqueue_document_organization_classification_handler
    ),
) -> OrganizationEnqueueResponse:
    try:
        result = await handler(EnqueueDocumentOrganizationClassificationCommand(document_id))
    except LookupError as exc:
        _not_found(exc)
    except ValueError as exc:
        _unprocessable(exc)
    return OrganizationEnqueueResponse(
        job_ids=[job.id for job in result.jobs],
        skipped_document_ids=list(result.skipped_document_ids),
    )


@router.post(
    "/document-organization/backfill",
    response_model=OrganizationEnqueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def backfill_document_organization(
    limit: int = Query(default=500, ge=1, le=5000),
    handler: EnqueueDocumentOrganizationClassificationHandler = Depends(
        get_enqueue_document_organization_classification_handler
    ),
) -> OrganizationEnqueueResponse:
    try:
        result = await handler(EnqueueDocumentOrganizationClassificationCommand(limit=limit))
    except ValueError as exc:
        _unprocessable(exc)
    return OrganizationEnqueueResponse(
        job_ids=[job.id for job in result.jobs],
        skipped_document_ids=list(result.skipped_document_ids),
    )


def _type_response(value: DocumentTypeRecord) -> DocumentTypeResponse:
    return DocumentTypeResponse(**asdict(value))


def _alias_response(value: ContentGroupAliasRecord) -> ContentGroupAliasResponse:
    return ContentGroupAliasResponse(**asdict(value))


def _group_response(value: ContentGroupRecord) -> ContentGroupResponse:
    return ContentGroupResponse(
        id=value.id,
        name=value.name,
        normalized_name=value.normalized_name,
        description=value.description,
        metadata=value.metadata,
        archived_at=value.archived_at,
        created_at=value.created_at,
        updated_at=value.updated_at,
        aliases=[_alias_response(alias) for alias in value.aliases],
    )


def _decision_response(value: DocumentTypeDecisionRecord) -> DocumentTypeDecisionResponse:
    return DocumentTypeDecisionResponse(**asdict(value))


def _assignment_response(
    value: DocumentContentGroupAssignmentRecord | None,
) -> DocumentContentGroupAssignmentResponse | None:
    if value is None:
        return None
    return DocumentContentGroupAssignmentResponse(**asdict(value))


def _write_conflict(exc: DocumentOrganizationWriteConflict) -> None:
    detail = DocumentOrganizationConflictResponse(
        message=str(exc),
        current_type_decisions=[_decision_response(value) for value in exc.current_type_decisions],
        current_assignment=_assignment_response(exc.current_assignment),
    )
    raise HTTPException(status_code=409, detail=detail.model_dump(mode="json")) from exc


def _not_found(exc: Exception) -> None:
    raise HTTPException(status_code=404, detail=str(exc)) from exc


def _conflict(exc: Exception) -> None:
    raise HTTPException(status_code=409, detail=str(exc)) from exc


def _unprocessable(exc: Exception) -> None:
    raise HTTPException(status_code=422, detail=str(exc)) from exc
