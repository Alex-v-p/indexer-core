from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies.application import (
    get_add_subject_alias_handler,
    get_archive_subject_alias_handler,
    get_create_subject_handler,
    get_document_subject_decisions_handler,
    get_document_subject_suggestion_review_handler,
    get_document_subject_suggestions_handler,
    get_document_subject_write_handler,
    get_subject_handler,
    get_subject_name_resolution_handler,
    get_subjects_handler,
    get_update_subject_handler,
)
from app.schemas.subjects import (
    AddSubjectAliasRequest,
    CreateSubjectRequest,
    DocumentSubjectDecisionResponse,
    ReviewSubjectSuggestionRequest,
    SetDocumentSubjectDecisionRequest,
    SubjectAliasResponse,
    SubjectDecisionConflictResponse,
    SubjectDecisionConflictErrorResponse,
    SubjectNameMatchResponse,
    SubjectNameResolutionResponse,
    SubjectResponse,
    UpdateSubjectRequest,
)
from packages.indexer_application.commands import (
    AddSubjectAliasCommand,
    AddSubjectAliasHandler,
    ArchivedSubjectMutationError,
    ArchiveSubjectAliasCommand,
    ArchiveSubjectAliasHandler,
    CreateSubjectCommand,
    CreateSubjectHandler,
    DocumentSubjectDecisionWriteConflict,
    ReviewDocumentSubjectSuggestionCommand,
    ReviewDocumentSubjectSuggestionHandler,
    SetDocumentSubjectDecisionCommand,
    SetDocumentSubjectDecisionHandler,
    UpdateSubjectCommand,
    UpdateSubjectHandler,
)
from packages.indexer_application.dto import (
    DocumentSubjectDecisionRecord,
    SubjectAliasRecord,
    SubjectRecord,
)
from packages.indexer_application.queries import (
    GetSubjectHandler,
    GetSubjectQuery,
    ListDocumentSubjectDecisionsHandler,
    ListDocumentSubjectDecisionsQuery,
    ListDocumentSubjectSuggestionsHandler,
    ListDocumentSubjectSuggestionsQuery,
    ListSubjectsHandler,
    ListSubjectsQuery,
    ResolveSubjectNameHandler,
    ResolveSubjectNameQuery,
)
from packages.indexer_application.ports import SubjectCanonicalNameConflictError
from packages.rag_core.subjects import (
    DecisionState,
    InvalidSubjectNameError,
    SubjectKind,
    SubjectName,
)

router = APIRouter(tags=["subjects"])


@router.post(
    "/subjects",
    response_model=SubjectResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_subject(
    body: CreateSubjectRequest,
    handler: CreateSubjectHandler = Depends(get_create_subject_handler),
) -> SubjectResponse:
    try:
        subject = await handler(
            CreateSubjectCommand(
                kind=body.kind,
                name=body.name,
                description=body.description,
                metadata=body.metadata,
            ),
        )
    except SubjectCanonicalNameConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (InvalidSubjectNameError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _to_subject_response(subject)


@router.get("/subjects", response_model=list[SubjectResponse])
async def list_subjects(
    kind: SubjectKind | None = None,
    include_archived: bool = False,
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    handler: ListSubjectsHandler = Depends(get_subjects_handler),
) -> list[SubjectResponse]:
    subjects = await handler(
        ListSubjectsQuery(
            kind=kind,
            include_archived=include_archived,
            limit=limit,
            offset=offset,
        ),
    )
    return [_to_subject_response(subject) for subject in subjects]


@router.get("/subjects/resolve", response_model=SubjectNameResolutionResponse)
async def resolve_subject_name(
    name: str = Query(min_length=1, max_length=255),
    kind: SubjectKind | None = None,
    include_archived: bool = False,
    handler: ResolveSubjectNameHandler = Depends(
        get_subject_name_resolution_handler,
    ),
) -> SubjectNameResolutionResponse:
    try:
        normalized_name = SubjectName.from_value(name).normalized
        matches = await handler(
            ResolveSubjectNameQuery(
                name=name,
                kind=kind,
                include_archived=include_archived,
            ),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return SubjectNameResolutionResponse(
        normalized_name=normalized_name,
        ambiguous=len(matches) > 1,
        matches=[
            SubjectNameMatchResponse(
                subject=_to_subject_response(match.subject),
                match_type=match.match_type,
            )
            for match in matches
        ],
    )


@router.get("/subjects/{subject_id}", response_model=SubjectResponse)
async def get_subject(
    subject_id: uuid.UUID,
    include_archived: bool = False,
    handler: GetSubjectHandler = Depends(get_subject_handler),
) -> SubjectResponse:
    subject = await handler(
        GetSubjectQuery(
            subject_id=subject_id,
            include_archived=include_archived,
        ),
    )
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subject not found.",
        )
    return _to_subject_response(subject)


@router.patch("/subjects/{subject_id}", response_model=SubjectResponse)
async def update_subject(
    subject_id: uuid.UUID,
    body: UpdateSubjectRequest,
    handler: UpdateSubjectHandler = Depends(get_update_subject_handler),
) -> SubjectResponse:
    try:
        subject = await handler(
            UpdateSubjectCommand(
                subject_id=subject_id,
                name=body.name,
                archive=body.archive,
            ),
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (SubjectCanonicalNameConflictError, ArchivedSubjectMutationError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (InvalidSubjectNameError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _to_subject_response(subject)


@router.post(
    "/subjects/{subject_id}/aliases",
    response_model=SubjectAliasResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_subject_alias(
    subject_id: uuid.UUID,
    body: AddSubjectAliasRequest,
    handler: AddSubjectAliasHandler = Depends(get_add_subject_alias_handler),
) -> SubjectAliasResponse:
    try:
        alias = await handler(AddSubjectAliasCommand(subject_id=subject_id, name=body.name))
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ArchivedSubjectMutationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (InvalidSubjectNameError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _to_alias_response(alias)


@router.delete(
    "/subjects/{subject_id}/aliases/{alias_id}",
    response_model=SubjectAliasResponse,
)
async def archive_subject_alias(
    subject_id: uuid.UUID,
    alias_id: uuid.UUID,
    handler: ArchiveSubjectAliasHandler = Depends(get_archive_subject_alias_handler),
) -> SubjectAliasResponse:
    try:
        alias = await handler(
            ArchiveSubjectAliasCommand(subject_id=subject_id, alias_id=alias_id),
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ArchivedSubjectMutationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return _to_alias_response(alias)


@router.get(
    "/documents/{document_id}/subjects",
    response_model=list[DocumentSubjectDecisionResponse],
)
async def list_document_subject_decisions(
    document_id: uuid.UUID,
    handler: ListDocumentSubjectDecisionsHandler = Depends(
        get_document_subject_decisions_handler,
    ),
) -> list[DocumentSubjectDecisionResponse]:
    try:
        decisions = await handler(
            ListDocumentSubjectDecisionsQuery(document_id=document_id),
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return [_to_decision_response(decision) for decision in decisions]


@router.put(
    "/documents/{document_id}/subjects/{subject_id}",
    response_model=DocumentSubjectDecisionResponse,
    responses={
        status.HTTP_409_CONFLICT: {
            "model": SubjectDecisionConflictErrorResponse,
            "description": "The expected decision revision did not match.",
        },
    },
)
async def set_document_subject_decision(
    document_id: uuid.UUID,
    subject_id: uuid.UUID,
    body: SetDocumentSubjectDecisionRequest,
    handler: SetDocumentSubjectDecisionHandler = Depends(
        get_document_subject_write_handler,
    ),
) -> DocumentSubjectDecisionResponse:
    try:
        decision = await handler(
            SetDocumentSubjectDecisionCommand(
                document_id=document_id,
                subject_id=subject_id,
                state=DecisionState(body.state),
                expected_revision=body.expected_revision,
                rationale=body.rationale,
            ),
        )
    except DocumentSubjectDecisionWriteConflict as exc:
        _raise_decision_conflict(exc)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ArchivedSubjectMutationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _to_decision_response(decision)


@router.get(
    "/documents/{document_id}/subject-suggestions",
    response_model=list[DocumentSubjectDecisionResponse],
)
async def list_document_subject_suggestions(
    document_id: uuid.UUID,
    handler: ListDocumentSubjectSuggestionsHandler = Depends(
        get_document_subject_suggestions_handler,
    ),
) -> list[DocumentSubjectDecisionResponse]:
    try:
        decisions = await handler(
            ListDocumentSubjectSuggestionsQuery(document_id=document_id),
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return [_to_decision_response(decision) for decision in decisions]


@router.post(
    "/documents/{document_id}/subject-suggestions/{subject_id}/review",
    response_model=DocumentSubjectDecisionResponse,
    responses={
        status.HTTP_409_CONFLICT: {
            "model": SubjectDecisionConflictErrorResponse,
            "description": "The suggestion changed or cannot be reviewed.",
        },
    },
)
async def review_document_subject_suggestion(
    document_id: uuid.UUID,
    subject_id: uuid.UUID,
    body: ReviewSubjectSuggestionRequest,
    handler: ReviewDocumentSubjectSuggestionHandler = Depends(
        get_document_subject_suggestion_review_handler,
    ),
) -> DocumentSubjectDecisionResponse:
    try:
        decision = await handler(
            ReviewDocumentSubjectSuggestionCommand(
                document_id=document_id,
                subject_id=subject_id,
                accept=body.decision == "accept",
                expected_revision=body.expected_revision,
                rationale=body.rationale,
            ),
        )
    except DocumentSubjectDecisionWriteConflict as exc:
        _raise_decision_conflict(exc)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ArchivedSubjectMutationError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _to_decision_response(decision)


def _raise_decision_conflict(exc: DocumentSubjectDecisionWriteConflict) -> None:
    detail = SubjectDecisionConflictResponse(
        message=str(exc),
        current=(
            _to_decision_response(exc.current)
            if exc.current is not None
            else None
        ),
    )
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=detail.model_dump(mode="json"),
    ) from exc


def _to_subject_response(subject: SubjectRecord) -> SubjectResponse:
    return SubjectResponse(
        id=subject.id,
        kind=subject.kind,
        name=subject.name,
        normalized_name=subject.normalized_name,
        description=subject.description,
        metadata=subject.metadata,
        created_at=subject.created_at,
        updated_at=subject.updated_at,
        archived_at=subject.archived_at,
        aliases=[_to_alias_response(alias) for alias in subject.aliases],
    )


def _to_alias_response(alias: SubjectAliasRecord) -> SubjectAliasResponse:
    return SubjectAliasResponse(
        id=alias.id,
        subject_id=alias.subject_id,
        name=alias.name,
        normalized_name=alias.normalized_name,
        created_at=alias.created_at,
        archived_at=alias.archived_at,
    )


def _to_decision_response(
    decision: DocumentSubjectDecisionRecord,
) -> DocumentSubjectDecisionResponse:
    return DocumentSubjectDecisionResponse(
        id=decision.id,
        document_id=decision.document_id,
        subject_id=decision.subject_id,
        state=decision.state,
        control_source=decision.control_source,
        confidence=decision.confidence,
        confidence_band=decision.confidence_band,
        rationale=decision.rationale,
        classifier_version=decision.classifier_version,
        policy_version=decision.policy_version,
        signals=decision.signals,
        classified_document_version_id=decision.classified_document_version_id,
        revision=decision.revision,
        created_at=decision.created_at,
        updated_at=decision.updated_at,
    )
