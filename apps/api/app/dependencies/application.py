from __future__ import annotations

from fastapi import Depends

from app.composition import build_document_object_store
from app.core.config import Settings, get_settings
from app.dependencies.database import get_unit_of_work
from app.dependencies.query_runtime import get_query_pipeline_registry
from packages.indexer_application.commands import (
    AddContentGroupAliasHandler,
    AddSubjectAliasHandler,
    ArchiveContentGroupAliasHandler,
    ArchiveSubjectAliasHandler,
    CreateContentGroupHandler,
    CreateDocumentTypeHandler,
    CreateSubjectHandler,
    EnqueueDocumentVersionDeletionHandler,
    EnqueueDocumentMaintenanceHandler,
    EnqueueEvaluationHandler,
    EnqueueDocumentSubjectClassificationHandler,
    EnqueueDocumentOrganizationClassificationHandler,
    SubmitQueryHandler,
    SubmitDocumentIngestionHandler,
    ReviewDocumentSubjectSuggestionHandler,
    SetDocumentSubjectDecisionHandler,
    ReplaceDocumentTypesHandler,
    SetDocumentContentGroupHandler,
    UpdateSubjectHandler,
    UpdateContentGroupHandler,
    UpdateDocumentTypeHandler,
)
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.queries import (
    GetContentGroupHandler,
    GetBackgroundJobHandler,
    GetDocumentHandler,
    GetDocumentOrganizationHandler,
    GetDocumentTypeHandler,
    GetQueryRunHandler,
    ListBackgroundJobsHandler,
    ListDocumentsHandler,
    ListContentGroupsHandler,
    ListDocumentTypesHandler,
    GetSubjectHandler,
    ListDocumentSubjectDecisionsHandler,
    ListDocumentSubjectSuggestionsHandler,
    ListSubjectsHandler,
    ResolveSubjectNameHandler,
)
from packages.rag_core.pipelines import PipelineRegistry
from packages.rag_core.subjects import POLICY_VERSION
from packages.rag_core.document_organization import ORGANIZATION_POLICY_VERSION


def get_submit_document_ingestion_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> SubmitDocumentIngestionHandler:
    return SubmitDocumentIngestionHandler(
        uow=uow,
        object_store=build_document_object_store(settings),
        max_attempts=settings.background_job_ingestion_max_attempts,
    )


def get_enqueue_document_version_deletion_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueDocumentVersionDeletionHandler:
    return EnqueueDocumentVersionDeletionHandler(
        uow=uow,
        max_attempts=settings.background_job_deletion_max_attempts,
    )


def get_enqueue_document_maintenance_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueDocumentMaintenanceHandler:
    return EnqueueDocumentMaintenanceHandler(
        uow=uow,
        max_attempts=settings.background_job_maintenance_max_attempts,
    )


def get_enqueue_document_subject_classification_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueDocumentSubjectClassificationHandler:
    return EnqueueDocumentSubjectClassificationHandler(
        uow=uow,
        enabled=settings.subject_classification_enabled,
        policy_version=settings.subject_classification_policy_version or POLICY_VERSION,
        max_attempts=settings.background_job_subject_classification_max_attempts,
    )


def get_enqueue_document_organization_classification_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueDocumentOrganizationClassificationHandler:
    return EnqueueDocumentOrganizationClassificationHandler(
        uow=uow,
        enabled=settings.organization_classification_enabled,
        policy_version=(
            settings.organization_classification_policy_version
            or ORGANIZATION_POLICY_VERSION
        ),
        max_attempts=settings.background_job_organization_classification_max_attempts,
    )


def get_enqueue_evaluation_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueEvaluationHandler:
    return EnqueueEvaluationHandler(
        uow=uow,
        max_attempts=settings.background_job_evaluation_max_attempts,
    )


def get_background_job_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetBackgroundJobHandler:
    return GetBackgroundJobHandler(uow=uow)


def get_list_background_jobs_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListBackgroundJobsHandler:
    return ListBackgroundJobsHandler(uow=uow)


def get_submit_query_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    pipeline_registry: PipelineRegistry = Depends(get_query_pipeline_registry),
    settings: Settings = Depends(get_settings),
) -> SubmitQueryHandler:
    return SubmitQueryHandler(
        uow=uow,
        pipeline_registry=pipeline_registry,
        max_attempts=settings.background_job_query_max_attempts,
        priority=settings.background_job_query_priority,
    )


def get_document_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetDocumentHandler:
    return GetDocumentHandler(uow=uow)


def get_list_documents_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListDocumentsHandler:
    return ListDocumentsHandler(uow=uow)


def get_query_run_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetQueryRunHandler:
    return GetQueryRunHandler(uow=uow)


def get_create_subject_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> CreateSubjectHandler:
    return CreateSubjectHandler(uow=uow)


def get_update_subject_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> UpdateSubjectHandler:
    return UpdateSubjectHandler(uow=uow)


def get_add_subject_alias_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> AddSubjectAliasHandler:
    return AddSubjectAliasHandler(uow=uow)


def get_archive_subject_alias_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ArchiveSubjectAliasHandler:
    return ArchiveSubjectAliasHandler(uow=uow)


def get_subjects_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListSubjectsHandler:
    return ListSubjectsHandler(uow=uow)


def get_subject_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetSubjectHandler:
    return GetSubjectHandler(uow=uow)


def get_subject_name_resolution_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ResolveSubjectNameHandler:
    return ResolveSubjectNameHandler(uow=uow)


def get_document_subject_decisions_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListDocumentSubjectDecisionsHandler:
    return ListDocumentSubjectDecisionsHandler(uow=uow)


def get_document_subject_suggestions_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListDocumentSubjectSuggestionsHandler:
    return ListDocumentSubjectSuggestionsHandler(uow=uow)


def get_document_subject_write_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> SetDocumentSubjectDecisionHandler:
    return SetDocumentSubjectDecisionHandler(uow=uow)


def get_document_subject_suggestion_review_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ReviewDocumentSubjectSuggestionHandler:
    return ReviewDocumentSubjectSuggestionHandler(uow=uow)


def get_create_document_type_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> CreateDocumentTypeHandler:
    return CreateDocumentTypeHandler(uow=uow)


def get_update_document_type_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> UpdateDocumentTypeHandler:
    return UpdateDocumentTypeHandler(uow=uow)


def get_document_types_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListDocumentTypesHandler:
    return ListDocumentTypesHandler(uow=uow)


def get_document_type_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetDocumentTypeHandler:
    return GetDocumentTypeHandler(uow=uow)


def get_create_content_group_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> CreateContentGroupHandler:
    return CreateContentGroupHandler(uow=uow)


def get_update_content_group_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> UpdateContentGroupHandler:
    return UpdateContentGroupHandler(uow=uow)


def get_add_content_group_alias_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> AddContentGroupAliasHandler:
    return AddContentGroupAliasHandler(uow=uow)


def get_archive_content_group_alias_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ArchiveContentGroupAliasHandler:
    return ArchiveContentGroupAliasHandler(uow=uow)


def get_content_groups_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListContentGroupsHandler:
    return ListContentGroupsHandler(uow=uow)


def get_content_group_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetContentGroupHandler:
    return GetContentGroupHandler(uow=uow)


def get_document_organization_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetDocumentOrganizationHandler:
    return GetDocumentOrganizationHandler(uow=uow)


def get_replace_document_types_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ReplaceDocumentTypesHandler:
    return ReplaceDocumentTypesHandler(uow=uow)


def get_set_document_content_group_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> SetDocumentContentGroupHandler:
    return SetDocumentContentGroupHandler(uow=uow)
