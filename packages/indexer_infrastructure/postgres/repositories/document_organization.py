from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, not_, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from packages.indexer_application.dto import (
    ContentGroupAliasRecord,
    ContentGroupNameMatchRecord,
    ContentGroupRecord,
    DocumentContentGroupAssignmentRecord,
    DocumentTypeDecisionRecord,
    DocumentTypeRecord,
)
from packages.indexer_application.ports import (
    ContentGroupInUseError,
    ContentGroupNameConflictError,
    DocumentOrganizationConflictError,
    DocumentTypeKeyConflictError,
)
from packages.indexer_infrastructure.postgres.models import (
    ContentGroup,
    ContentGroupAlias,
    DocumentContentGroupAssignment as ContentGroupAssignmentModel,
    DocumentType,
    DocumentTypeDecision as DocumentTypeDecisionModel,
    DocumentVersion,
)
from packages.indexer_infrastructure.postgres.repositories.mappers import (
    to_content_group_record,
    to_document_content_group_assignment_record,
    to_document_type_decision_record,
    to_document_type_record,
)
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    ContentGroupName,
    ContentGroupNameMatchType,
    DocumentContentGroupAssignment,
    DocumentTypeDecision,
    DocumentTypeDecisionState,
    DocumentTypeKey,
)


class SqlAlchemyDocumentTypeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        key: str,
        label: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentTypeRecord:
        model = DocumentType(
            key=DocumentTypeKey.from_value(key).value,
            label=_required_label(label),
            description=_optional_description(description),
            metadata_=dict(metadata or {}),
        )
        self._session.add(model)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            if _constraint_name(exc) == "uq_document_types_key":
                raise DocumentTypeKeyConflictError(
                    "A document type with that key already exists."
                ) from exc
            raise
        return to_document_type_record(model)

    async def get(
        self,
        document_type_id: uuid.UUID,
        *,
        include_archived: bool = False,
    ) -> DocumentTypeRecord | None:
        statement = select(DocumentType).where(DocumentType.id == document_type_id)
        if not include_archived:
            statement = statement.where(DocumentType.archived_at.is_(None))
        model = (await self._session.execute(statement)).scalar_one_or_none()
        return to_document_type_record(model) if model is not None else None

    async def update(
        self,
        document_type_id: uuid.UUID,
        *,
        label: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentTypeRecord:
        if label is None and description is None and metadata is None:
            raise ValueError("At least one document type update must be provided.")
        model = await self._require_type(document_type_id)
        if label is not None:
            model.label = _required_label(label)
        if description is not None:
            model.description = _optional_description(description)
        if metadata is not None:
            model.metadata_ = dict(metadata)
        model.updated_at = datetime.now(UTC)
        await self._session.flush()
        return to_document_type_record(model)

    async def get_by_key(
        self,
        key: str,
        *,
        include_archived: bool = False,
    ) -> DocumentTypeRecord | None:
        resolved_key = DocumentTypeKey.from_value(key).value
        statement = select(DocumentType).where(DocumentType.key == resolved_key)
        if not include_archived:
            statement = statement.where(DocumentType.archived_at.is_(None))
        model = (await self._session.execute(statement)).scalar_one_or_none()
        return to_document_type_record(model) if model is not None else None

    async def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentTypeRecord]:
        _validate_page(limit, offset)
        statement = select(DocumentType)
        if not include_archived:
            statement = statement.where(DocumentType.archived_at.is_(None))
        result = await self._session.execute(
            statement.order_by(DocumentType.key, DocumentType.id).offset(offset).limit(limit)
        )
        return [to_document_type_record(model) for model in result.scalars().all()]

    async def archive(self, document_type_id: uuid.UUID) -> DocumentTypeRecord:
        model = await self._require_type(document_type_id)
        if model.archived_at is None:
            model.archived_at = datetime.now(UTC)
            model.updated_at = model.archived_at
            await self._session.flush()
        return to_document_type_record(model)

    async def get_decision(
        self, *, document_id: uuid.UUID, document_type_id: uuid.UUID
    ) -> DocumentTypeDecisionRecord | None:
        model = (
            await self._session.execute(
                select(DocumentTypeDecisionModel).where(
                    DocumentTypeDecisionModel.document_id == document_id,
                    DocumentTypeDecisionModel.document_type_id == document_type_id,
                )
            )
        ).scalar_one_or_none()
        return to_document_type_decision_record(model) if model is not None else None

    async def list_document_decisions(
        self,
        *,
        document_id: uuid.UUID,
        states: tuple[DocumentTypeDecisionState, ...] | None = None,
    ) -> list[DocumentTypeDecisionRecord]:
        statement = select(DocumentTypeDecisionModel).where(
            DocumentTypeDecisionModel.document_id == document_id
        )
        if states is not None:
            if not states:
                return []
            statement = statement.where(
                DocumentTypeDecisionModel.state.in_(tuple(DocumentTypeDecisionState(state).value for state in states))
            )
        result = await self._session.execute(
            statement.order_by(DocumentTypeDecisionModel.document_type_id)
        )
        return [to_document_type_decision_record(model) for model in result.scalars().all()]

    async def write_decision(
        self,
        decision: DocumentTypeDecision,
        *,
        expected_revision: int | None = None,
    ) -> DocumentTypeDecisionRecord:
        _validate_expected_revision(decision.source, expected_revision)
        await self._require_type(decision.document_type_id)
        if decision.classified_document_version_id is not None:
            await _require_version(self._session, decision.document_id, decision.classified_document_version_id)
        if expected_revision is None:
            model = await self._blind_upsert(decision)
            if model is None:
                current = await self.get_decision(
                    document_id=decision.document_id,
                    document_type_id=decision.document_type_id,
                )
                if current is not None and _same_type_decision(current, decision):
                    return current
                raise DocumentOrganizationConflictError(
                    "Automatic document type decision cannot overwrite a manual decision."
                )
        elif expected_revision == 0:
            model = await self._insert(decision)
        else:
            model = await self._update(decision, expected_revision)
        if model is None:
            raise DocumentOrganizationConflictError(
                "Document type decision did not match the expected revision."
            )
        return to_document_type_decision_record(model)

    async def _blind_upsert(self, decision: DocumentTypeDecision) -> DocumentTypeDecisionModel | None:
        values = _type_decision_values(decision)
        insert = postgresql_insert(DocumentTypeDecisionModel).values(id=uuid.uuid4(), revision=1, **values)
        excluded = insert.excluded
        different = _columns_differ(DocumentTypeDecisionModel, excluded, tuple(key for key in values if key not in {"document_id", "document_type_id"}))
        statement = insert.on_conflict_do_update(
            constraint="uq_document_type_decisions_document_type",
            set_={
                **{key: getattr(excluded, key) for key in values if key not in {"document_id", "document_type_id"}},
                "revision": DocumentTypeDecisionModel.revision + 1,
                "updated_at": func.now(),
            },
            where=and_(_automatic_type_write_is_allowed(decision), different),
        ).returning(DocumentTypeDecisionModel)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _insert(self, decision: DocumentTypeDecision) -> DocumentTypeDecisionModel | None:
        statement = postgresql_insert(DocumentTypeDecisionModel).values(
            id=uuid.uuid4(), revision=1, **_type_decision_values(decision)
        ).on_conflict_do_nothing(
            constraint="uq_document_type_decisions_document_type"
        ).returning(DocumentTypeDecisionModel)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _update(self, decision: DocumentTypeDecision, revision: int) -> DocumentTypeDecisionModel | None:
        values = _type_decision_values(decision)
        statement = update(DocumentTypeDecisionModel).where(
            DocumentTypeDecisionModel.document_id == decision.document_id,
            DocumentTypeDecisionModel.document_type_id == decision.document_type_id,
            DocumentTypeDecisionModel.revision == revision,
            _automatic_type_write_is_allowed(decision),
        ).values(
            **{key: value for key, value in values.items() if key not in {"document_id", "document_type_id"}},
            revision=DocumentTypeDecisionModel.revision + 1,
            updated_at=func.now(),
        ).returning(DocumentTypeDecisionModel)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _require_type(self, document_type_id: uuid.UUID) -> DocumentType:
        model = (
            await self._session.execute(
                select(DocumentType).where(
                    DocumentType.id == document_type_id,
                    DocumentType.archived_at.is_(None),
                ).with_for_update()
            )
        ).scalar_one_or_none()
        if model is None:
            raise LookupError(f"Active document type {document_type_id} was not found.")
        return model


class SqlAlchemyContentGroupRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def acquire_publish_lock(self) -> None:
        """Serialize the short catalogue recheck and automatic group publish step."""

        await self._session.execute(
            select(func.pg_advisory_xact_lock(func.hashtext("document-organization-publish")))
        )

    async def create(
        self,
        *,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContentGroupRecord:
        resolved = ContentGroupName.from_value(name)
        await self._lock_name(resolved.normalized)
        await self._ensure_name_available(resolved.normalized)
        model = ContentGroup(
            name=resolved.value,
            normalized_name=resolved.normalized,
            description=_optional_description(description),
            metadata_=dict(metadata or {}),
            aliases=[],
        )
        self._session.add(model)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            _raise_group_name_conflict(exc)
        return to_content_group_record(model)

    async def create_or_get_canonical(
        self,
        *,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[ContentGroupRecord, bool]:
        resolved = ContentGroupName.from_value(name)
        await self._lock_name(resolved.normalized)
        alias_owner = (
            await self._session.execute(
                select(ContentGroupAlias.content_group_id).where(
                    ContentGroupAlias.normalized_name == resolved.normalized
                )
            )
        ).scalar_one_or_none()
        if alias_owner is not None:
            existing = await self.get(alias_owner, include_archived=True)
            if existing is None or existing.archived_at is not None:
                raise ContentGroupNameConflictError(
                    "An archived content group alias owns that normalized name."
                )
            return existing, False
        content_group_id = uuid.uuid4()
        insert = postgresql_insert(ContentGroup).values(
            id=content_group_id,
            name=resolved.value,
            normalized_name=resolved.normalized,
            description=_optional_description(description),
            metadata_=dict(metadata or {}),
        ).on_conflict_do_nothing(
            constraint="uq_content_groups_normalized_name"
        ).returning(ContentGroup.id)
        created_id = (await self._session.execute(insert)).scalar_one_or_none()
        result = await self._session.execute(
            select(ContentGroup).where(
                ContentGroup.id == created_id
                if created_id is not None
                else ContentGroup.normalized_name == resolved.normalized
            ).options(selectinload(ContentGroup.aliases))
        )
        model = result.scalar_one_or_none()
        if model is None:
            raise RuntimeError("Content group create-or-get did not resolve a row.")
        if model.archived_at is not None:
            raise ContentGroupNameConflictError(
                "An archived content group owns that normalized name."
            )
        return to_content_group_record(model), created_id is not None

    async def get(
        self, content_group_id: uuid.UUID, *, include_archived: bool = False
    ) -> ContentGroupRecord | None:
        statement = select(ContentGroup).where(ContentGroup.id == content_group_id).options(selectinload(ContentGroup.aliases))
        if not include_archived:
            statement = statement.where(ContentGroup.archived_at.is_(None))
        model = (await self._session.execute(statement)).scalar_one_or_none()
        return to_content_group_record(model) if model is not None else None

    async def list(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ContentGroupRecord]:
        _validate_page(limit, offset)
        statement = select(ContentGroup).options(selectinload(ContentGroup.aliases))
        if not include_archived:
            statement = statement.where(ContentGroup.archived_at.is_(None))
        result = await self._session.execute(
            statement.order_by(ContentGroup.normalized_name, ContentGroup.id).offset(offset).limit(limit)
        )
        return [to_content_group_record(model) for model in result.scalars().unique().all()]

    async def rename(self, *, content_group_id: uuid.UUID, name: str) -> ContentGroupRecord:
        resolved = ContentGroupName.from_value(name)
        await self._lock_name(resolved.normalized)
        model = await self._require_group(content_group_id, include_aliases=True)
        if resolved.normalized != model.normalized_name:
            await self._ensure_name_available(resolved.normalized, owner_id=content_group_id)
        model.name = resolved.value
        model.normalized_name = resolved.normalized
        now = datetime.now(UTC)
        model.updated_at = now
        for alias in model.aliases:
            if alias.normalized_name == resolved.normalized:
                alias.archived_at = now
        try:
            await self._session.flush()
        except IntegrityError as exc:
            _raise_group_name_conflict(exc)
        return to_content_group_record(model)

    async def archive(self, content_group_id: uuid.UUID) -> ContentGroupRecord:
        model = await self._require_group(content_group_id, include_aliases=True)
        active_reference = (
            await self._session.execute(
                select(ContentGroupAssignmentModel.document_id).where(
                    ContentGroupAssignmentModel.content_group_id == content_group_id,
                    ContentGroupAssignmentModel.state.in_((
                        ContentGroupAssignmentState.SUGGESTED.value,
                        ContentGroupAssignmentState.ASSIGNED.value,
                    )),
                ).limit(1)
            )
        ).scalar_one_or_none()
        if active_reference is not None:
            raise ContentGroupInUseError(
                "Content group cannot be archived while documents actively reference it."
            )
        if model.archived_at is None:
            model.archived_at = datetime.now(UTC)
            model.updated_at = model.archived_at
            await self._session.flush()
        return to_content_group_record(model)

    async def add_alias(self, *, content_group_id: uuid.UUID, name: str) -> ContentGroupAliasRecord:
        resolved = ContentGroupName.from_value(name)
        await self._lock_name(resolved.normalized)
        group = await self._require_group(content_group_id)
        if group.normalized_name == resolved.normalized:
            raise ValueError("content group alias must differ from its canonical name.")
        await self._ensure_name_available(resolved.normalized, owner_id=content_group_id)
        result = await self._session.execute(
            select(ContentGroupAlias).where(
                ContentGroupAlias.content_group_id == content_group_id,
                ContentGroupAlias.normalized_name == resolved.normalized,
            )
        )
        alias = result.scalar_one_or_none()
        if alias is None:
            alias = ContentGroupAlias(
                content_group_id=content_group_id,
                name=resolved.value,
                normalized_name=resolved.normalized,
            )
            self._session.add(alias)
        else:
            alias.name = resolved.value
            alias.archived_at = None
        try:
            await self._session.flush()
        except IntegrityError as exc:
            _raise_group_name_conflict(exc)
        return _to_group_alias(alias)

    async def archive_alias(self, *, content_group_id: uuid.UUID, alias_id: uuid.UUID) -> ContentGroupAliasRecord:
        await self._require_group(content_group_id)
        alias = (
            await self._session.execute(
                select(ContentGroupAlias).where(
                    ContentGroupAlias.id == alias_id,
                    ContentGroupAlias.content_group_id == content_group_id,
                )
            )
        ).scalar_one_or_none()
        if alias is None:
            raise LookupError(f"Content group alias {alias_id} was not found.")
        if alias.archived_at is None:
            alias.archived_at = datetime.now(UTC)
            await self._session.flush()
        return _to_group_alias(alias)

    async def resolve_name(self, name: str, *, include_archived: bool = False) -> list[ContentGroupNameMatchRecord]:
        normalized = ContentGroupName.from_value(name).normalized
        statement = select(ContentGroup).outerjoin(
            ContentGroupAlias, ContentGroupAlias.content_group_id == ContentGroup.id
        ).where(
            or_(
                ContentGroup.normalized_name == normalized,
                and_(ContentGroupAlias.normalized_name == normalized, ContentGroupAlias.archived_at.is_(None)),
            )
        ).options(selectinload(ContentGroup.aliases))
        if not include_archived:
            statement = statement.where(ContentGroup.archived_at.is_(None))
        result = await self._session.execute(statement)
        models = result.scalars().unique().all()
        return [
            ContentGroupNameMatchRecord(
                content_group=to_content_group_record(model),
                match_type=(ContentGroupNameMatchType.CANONICAL if model.normalized_name == normalized else ContentGroupNameMatchType.ALIAS),
            )
            for model in sorted(models, key=lambda item: (item.normalized_name != normalized, str(item.id)))
        ]

    async def get_assignment(self, document_id: uuid.UUID) -> DocumentContentGroupAssignmentRecord | None:
        model = (
            await self._session.execute(
                select(ContentGroupAssignmentModel).where(ContentGroupAssignmentModel.document_id == document_id)
            )
        ).scalar_one_or_none()
        return to_document_content_group_assignment_record(model) if model is not None else None

    async def list_assignments(
        self,
        *,
        content_group_id: uuid.UUID | None = None,
        states: tuple[ContentGroupAssignmentState, ...] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentContentGroupAssignmentRecord]:
        _validate_page(limit, offset)
        statement = select(ContentGroupAssignmentModel)
        if content_group_id is not None:
            statement = statement.where(ContentGroupAssignmentModel.content_group_id == content_group_id)
        if states is not None:
            if not states:
                return []
            statement = statement.where(
                ContentGroupAssignmentModel.state.in_(tuple(ContentGroupAssignmentState(state).value for state in states))
            )
        result = await self._session.execute(
            statement.order_by(ContentGroupAssignmentModel.document_id).offset(offset).limit(limit)
        )
        return [to_document_content_group_assignment_record(model) for model in result.scalars().all()]

    async def write_assignment(
        self,
        assignment: DocumentContentGroupAssignment,
        *,
        expected_revision: int | None = None,
    ) -> DocumentContentGroupAssignmentRecord:
        _validate_expected_revision(assignment.source, expected_revision)
        if assignment.content_group_id is not None:
            await self._require_group(assignment.content_group_id)
        if assignment.classified_document_version_id is not None:
            await _require_version(self._session, assignment.document_id, assignment.classified_document_version_id)
        if expected_revision is None:
            model = await self._blind_upsert_assignment(assignment)
            if model is None:
                current = await self.get_assignment(assignment.document_id)
                if current is not None and _same_assignment(current, assignment):
                    return current
                raise DocumentOrganizationConflictError(
                    "Automatic content group assignment cannot overwrite a manual assignment."
                )
        elif expected_revision == 0:
            model = await self._insert_assignment(assignment)
        else:
            model = await self._update_assignment(assignment, expected_revision)
        if model is None:
            raise DocumentOrganizationConflictError(
                "Content group assignment did not match the expected revision."
            )
        return to_document_content_group_assignment_record(model)

    async def clear_assignment(
        self,
        *,
        document_id: uuid.UUID,
        expected_revision: int,
    ) -> DocumentContentGroupAssignmentRecord:
        if expected_revision <= 0:
            raise ValueError("Clearing a content group requires a positive expected_revision.")
        statement = update(ContentGroupAssignmentModel).where(
            ContentGroupAssignmentModel.document_id == document_id,
            ContentGroupAssignmentModel.revision == expected_revision,
        ).values(
            content_group_id=None,
            state=ContentGroupAssignmentState.PENDING.value,
            source=ClassificationSource.AUTOMATIC.value,
            unresolved_reason=None,
            confidence=None,
            confidence_band=None,
            rationale=None,
            classifier_version=None,
            policy_version=None,
            signals={},
            summary_hash=None,
            classified_document_version_id=None,
            revision=ContentGroupAssignmentModel.revision + 1,
            updated_at=func.now(),
        ).returning(ContentGroupAssignmentModel)
        model = (await self._session.execute(statement)).scalar_one_or_none()
        if model is None:
            raise DocumentOrganizationConflictError(
                "Content group assignment did not match the expected revision."
            )
        return to_document_content_group_assignment_record(model)

    async def _blind_upsert_assignment(self, assignment: DocumentContentGroupAssignment) -> ContentGroupAssignmentModel | None:
        values = _assignment_values(assignment)
        insert = postgresql_insert(ContentGroupAssignmentModel).values(revision=1, **values)
        excluded = insert.excluded
        different = _columns_differ(ContentGroupAssignmentModel, excluded, tuple(key for key in values if key != "document_id"))
        statement = insert.on_conflict_do_update(
            index_elements=[ContentGroupAssignmentModel.document_id],
            set_={
                **{key: getattr(excluded, key) for key in values if key != "document_id"},
                "revision": ContentGroupAssignmentModel.revision + 1,
                "updated_at": func.now(),
            },
            where=and_(_automatic_assignment_write_is_allowed(assignment), different),
        ).returning(ContentGroupAssignmentModel)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _insert_assignment(self, assignment: DocumentContentGroupAssignment) -> ContentGroupAssignmentModel | None:
        statement = postgresql_insert(ContentGroupAssignmentModel).values(
            revision=1, **_assignment_values(assignment)
        ).on_conflict_do_nothing(
            index_elements=[ContentGroupAssignmentModel.document_id]
        ).returning(ContentGroupAssignmentModel)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _update_assignment(self, assignment: DocumentContentGroupAssignment, revision: int) -> ContentGroupAssignmentModel | None:
        values = _assignment_values(assignment)
        statement = update(ContentGroupAssignmentModel).where(
            ContentGroupAssignmentModel.document_id == assignment.document_id,
            ContentGroupAssignmentModel.revision == revision,
            _automatic_assignment_write_is_allowed(assignment),
        ).values(
            **{key: value for key, value in values.items() if key != "document_id"},
            revision=ContentGroupAssignmentModel.revision + 1,
            updated_at=func.now(),
        ).returning(ContentGroupAssignmentModel)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def _lock_name(self, normalized: str) -> None:
        # Cross-table canonical/alias uniqueness is serialized by normalized name.
        await self._session.execute(select(func.pg_advisory_xact_lock(func.hashtext(normalized))))

    async def _ensure_name_available(self, normalized: str, *, owner_id: uuid.UUID | None = None) -> None:
        canonical_owner = (
            await self._session.execute(
                select(ContentGroup.id).where(ContentGroup.normalized_name == normalized)
            )
        ).scalar_one_or_none()
        alias_owner = (
            await self._session.execute(
                select(ContentGroupAlias.content_group_id).where(ContentGroupAlias.normalized_name == normalized)
            )
        ).scalar_one_or_none()
        owners = {owner for owner in (canonical_owner, alias_owner) if owner is not None}
        if owners - ({owner_id} if owner_id is not None else set()):
            raise ContentGroupNameConflictError(
                "A content group canonical name or alias already owns that normalized name."
            )

    async def _require_group(self, content_group_id: uuid.UUID, *, include_aliases: bool = False) -> ContentGroup:
        statement = select(ContentGroup).where(
            ContentGroup.id == content_group_id,
            ContentGroup.archived_at.is_(None),
        ).with_for_update()
        if include_aliases:
            statement = statement.options(selectinload(ContentGroup.aliases))
        model = (await self._session.execute(statement)).scalar_one_or_none()
        if model is None:
            raise LookupError(f"Active content group {content_group_id} was not found.")
        return model


def _type_decision_values(decision: DocumentTypeDecision) -> dict[str, Any]:
    return {
        "document_id": decision.document_id,
        "document_type_id": decision.document_type_id,
        "state": decision.state.value,
        "source": decision.source.value,
        "confidence": decision.confidence,
        "confidence_band": decision.confidence_band.value if decision.confidence_band else None,
        "rationale": decision.rationale,
        "classifier_version": decision.classifier_version,
        "policy_version": decision.policy_version,
        "signals": dict(decision.signals),
        "classified_document_version_id": decision.classified_document_version_id,
    }


def _assignment_values(assignment: DocumentContentGroupAssignment) -> dict[str, Any]:
    return {
        "document_id": assignment.document_id,
        "content_group_id": assignment.content_group_id,
        "state": assignment.state.value,
        "source": assignment.source.value,
        "unresolved_reason": assignment.unresolved_reason,
        "confidence": assignment.confidence,
        "confidence_band": assignment.confidence_band.value if assignment.confidence_band else None,
        "rationale": assignment.rationale,
        "classifier_version": assignment.classifier_version,
        "policy_version": assignment.policy_version,
        "signals": dict(assignment.signals),
        "summary_hash": assignment.summary_hash,
        "classified_document_version_id": assignment.classified_document_version_id,
    }


def _columns_differ(model: type[Any], excluded: Any, names: tuple[str, ...]) -> Any:
    return or_(*(getattr(model, name).is_distinct_from(getattr(excluded, name)) for name in names))


def _automatic_type_write_is_allowed(decision: DocumentTypeDecision) -> Any:
    if decision.source is ClassificationSource.MANUAL:
        return True
    return DocumentTypeDecisionModel.source != ClassificationSource.MANUAL.value


def _automatic_assignment_write_is_allowed(
    assignment: DocumentContentGroupAssignment,
) -> Any:
    if assignment.source is ClassificationSource.MANUAL:
        return True
    return ContentGroupAssignmentModel.source != ClassificationSource.MANUAL.value


def _same_type_decision(current: DocumentTypeDecisionRecord, proposed: DocumentTypeDecision) -> bool:
    return all(
        getattr(current, name) == getattr(proposed, name)
        for name in (
            "state", "source", "confidence", "confidence_band", "rationale",
            "classifier_version", "policy_version", "classified_document_version_id",
        )
    ) and current.signals == dict(proposed.signals)


def _same_assignment(current: DocumentContentGroupAssignmentRecord, proposed: DocumentContentGroupAssignment) -> bool:
    return all(
        getattr(current, name) == getattr(proposed, name)
        for name in (
            "content_group_id", "state", "source", "unresolved_reason", "confidence",
            "confidence_band", "rationale", "classifier_version", "policy_version",
            "summary_hash", "classified_document_version_id",
        )
    ) and current.signals == dict(proposed.signals)


async def _require_version(session: AsyncSession, document_id: uuid.UUID, version_id: uuid.UUID) -> None:
    found = (
        await session.execute(
            select(DocumentVersion.id).where(
                DocumentVersion.id == version_id,
                DocumentVersion.document_id == document_id,
            )
        )
    ).scalar_one_or_none()
    if found is None:
        raise ValueError("classified_document_version_id must belong to the document.")


def _validate_expected_revision(source: ClassificationSource, revision: int | None) -> None:
    if revision is not None and revision < 0:
        raise ValueError("expected_revision must not be negative.")
    if source is ClassificationSource.MANUAL and revision is None:
        raise ValueError("manual writes require expected_revision.")


def _validate_page(limit: int, offset: int) -> None:
    if limit <= 0 or offset < 0:
        raise ValueError("limit must be positive and offset must not be negative.")


def _required_label(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned or len(cleaned) > 120:
        raise ValueError("document type label must be non-blank and no longer than 120 characters.")
    return cleaned


def _optional_description(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    if len(cleaned) > 2_000:
        raise ValueError("description must not exceed 2000 characters.")
    return cleaned


def _to_group_alias(alias: ContentGroupAlias) -> ContentGroupAliasRecord:
    return ContentGroupAliasRecord(
        id=alias.id,
        content_group_id=alias.content_group_id,
        name=alias.name,
        normalized_name=alias.normalized_name,
        archived_at=alias.archived_at,
        created_at=alias.created_at,
    )


def _raise_group_name_conflict(error: IntegrityError) -> None:
    if _constraint_name(error) in {
        "uq_content_groups_normalized_name",
        "uq_content_group_aliases_normalized_name",
    }:
        raise ContentGroupNameConflictError(
            "A content group canonical name or alias already owns that normalized name."
        ) from error
    raise error


def _constraint_name(error: IntegrityError) -> str | None:
    candidate: object | None = error.orig
    visited: set[int] = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        direct = getattr(candidate, "constraint_name", None)
        if isinstance(direct, str):
            return direct
        diag = getattr(candidate, "diag", None)
        diagnosed = getattr(diag, "constraint_name", None)
        if isinstance(diagnosed, str):
            return diagnosed
        candidate = getattr(candidate, "__cause__", None) or getattr(candidate, "__context__", None)
    return None
