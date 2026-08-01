from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, case, func, not_, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from packages.indexer_application.dto import (
    DocumentSubjectDecisionRecord,
    SubjectAliasRecord,
    SubjectNameMatchRecord,
    SubjectRecord,
)
from packages.indexer_application.ports.repositories import (
    SubjectCanonicalNameConflictError,
    SubjectDecisionConflictError,
)
from packages.indexer_infrastructure.postgres.models import (
    DocumentSubjectDecision as DocumentSubjectDecisionModel,
    DocumentVersion,
    Subject,
    SubjectAlias,
)
from packages.indexer_infrastructure.postgres.repositories.mappers import (
    to_document_subject_decision_record,
    to_subject_record,
)
from packages.rag_core.subjects import (
    DecisionControlSource,
    DecisionState,
    DocumentSubjectDecision,
    SubjectKind,
    SubjectName,
    SubjectNameMatchType,
)


class SqlAlchemySubjectRepository:
    """PostgreSQL adapter for subjects, aliases, and current decisions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        kind: SubjectKind,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubjectRecord:
        subject_name = SubjectName.from_value(name)
        subject = Subject(
            kind=SubjectKind(kind).value,
            name=subject_name.value,
            normalized_name=subject_name.normalized,
            description=_optional_description(description),
            metadata_=dict(metadata or {}),
            aliases=[],
        )
        self._session.add(subject)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            _raise_canonical_name_conflict(exc, kind=SubjectKind(kind))
        return to_subject_record(subject)

    async def get(
        self,
        subject_id: uuid.UUID,
        *,
        include_archived: bool = False,
    ) -> SubjectRecord | None:
        statement = (
            select(Subject)
            .where(Subject.id == subject_id)
            .options(selectinload(Subject.aliases))
        )
        if not include_archived:
            statement = statement.where(Subject.archived_at.is_(None))
        result = await self._session.execute(statement)
        subject = result.scalar_one_or_none()
        return to_subject_record(subject) if subject is not None else None

    async def list(
        self,
        *,
        kind: SubjectKind | None = None,
        include_archived: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SubjectRecord]:
        if limit <= 0 or offset < 0:
            raise ValueError("limit must be positive and offset must not be negative.")
        statement = select(Subject).options(selectinload(Subject.aliases))
        if kind is not None:
            statement = statement.where(Subject.kind == SubjectKind(kind).value)
        if not include_archived:
            statement = statement.where(Subject.archived_at.is_(None))
        statement = (
            statement.order_by(_kind_order(), Subject.normalized_name, Subject.id)
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return [to_subject_record(item) for item in result.scalars().unique().all()]

    async def rename(self, *, subject_id: uuid.UUID, name: str) -> SubjectRecord:
        subject_name = SubjectName.from_value(name)
        subject = await self._require_subject(subject_id, include_aliases=True)
        subject.name = subject_name.value
        subject.normalized_name = subject_name.normalized
        now = datetime.now(UTC)
        for alias in subject.aliases:
            if alias.normalized_name == subject_name.normalized:
                alias.archived_at = now
        try:
            await self._session.flush()
        except IntegrityError as exc:
            _raise_canonical_name_conflict(
                exc,
                kind=SubjectKind(subject.kind),
            )
        return to_subject_record(subject)

    async def archive(self, *, subject_id: uuid.UUID) -> SubjectRecord:
        subject = await self._require_subject(subject_id, include_aliases=True)
        if subject.archived_at is None:
            subject.archived_at = datetime.now(UTC)
            await self._session.flush()
        return to_subject_record(subject)

    async def add_alias(
        self,
        *,
        subject_id: uuid.UUID,
        name: str,
    ) -> SubjectAliasRecord:
        alias_name = SubjectName.from_value(name)
        subject = await self._require_subject(subject_id)
        if subject.normalized_name == alias_name.normalized:
            raise ValueError("subject alias must differ from its canonical name.")
        result = await self._session.execute(
            select(SubjectAlias).where(
                SubjectAlias.subject_id == subject_id,
                SubjectAlias.normalized_name == alias_name.normalized,
            ),
        )
        alias = result.scalar_one_or_none()
        if alias is None:
            alias = SubjectAlias(
                subject_id=subject_id,
                name=alias_name.value,
                normalized_name=alias_name.normalized,
            )
            self._session.add(alias)
        else:
            alias.name = alias_name.value
            alias.archived_at = None
        await self._session.flush()
        return _to_alias_record(alias)

    async def archive_alias(
        self,
        *,
        subject_id: uuid.UUID,
        alias_id: uuid.UUID,
    ) -> SubjectAliasRecord:
        await self._require_subject(subject_id)
        result = await self._session.execute(
            select(SubjectAlias).where(
                SubjectAlias.id == alias_id,
                SubjectAlias.subject_id == subject_id,
            ),
        )
        alias = result.scalar_one_or_none()
        if alias is None:
            raise LookupError(f"Subject alias {alias_id} was not found.")
        if alias.archived_at is None:
            alias.archived_at = datetime.now(UTC)
            await self._session.flush()
        return _to_alias_record(alias)

    async def resolve_name(
        self,
        name: str,
        *,
        kind: SubjectKind | None = None,
        include_archived: bool = False,
    ) -> list[SubjectNameMatchRecord]:
        normalized = SubjectName.from_value(name).normalized
        statement = (
            select(Subject)
            .outerjoin(SubjectAlias, SubjectAlias.subject_id == Subject.id)
            .where(
                or_(
                    Subject.normalized_name == normalized,
                    and_(
                        SubjectAlias.normalized_name == normalized,
                        SubjectAlias.archived_at.is_(None),
                    ),
                ),
            )
            .options(selectinload(Subject.aliases))
        )
        if kind is not None:
            statement = statement.where(Subject.kind == SubjectKind(kind).value)
        if not include_archived:
            statement = statement.where(Subject.archived_at.is_(None))
        result = await self._session.execute(statement)
        subjects = list(result.scalars().unique().all())
        subjects.sort(
            key=lambda item: (
                item.normalized_name != normalized,
                _kind_rank(item.kind),
                item.normalized_name,
                str(item.id),
            ),
        )
        return [
            SubjectNameMatchRecord(
                subject=to_subject_record(item),
                match_type=(
                    SubjectNameMatchType.CANONICAL
                    if item.normalized_name == normalized
                    else SubjectNameMatchType.ALIAS
                ),
            )
            for item in subjects
        ]

    async def get_decision(
        self,
        *,
        document_id: uuid.UUID,
        subject_id: uuid.UUID,
    ) -> DocumentSubjectDecisionRecord | None:
        result = await self._session.execute(
            select(DocumentSubjectDecisionModel).where(
                DocumentSubjectDecisionModel.document_id == document_id,
                DocumentSubjectDecisionModel.subject_id == subject_id,
            ),
        )
        model = result.scalar_one_or_none()
        return (
            to_document_subject_decision_record(model)
            if model is not None
            else None
        )

    async def list_document_decisions(
        self,
        *,
        document_id: uuid.UUID,
        states: tuple[DecisionState, ...] | None = None,
    ) -> list[DocumentSubjectDecisionRecord]:
        statement = (
            select(DocumentSubjectDecisionModel)
            .join(Subject, Subject.id == DocumentSubjectDecisionModel.subject_id)
            .where(DocumentSubjectDecisionModel.document_id == document_id)
        )
        if states is not None:
            if not states:
                return []
            statement = statement.where(
                DocumentSubjectDecisionModel.state.in_(
                    tuple(DecisionState(state).value for state in states),
                ),
            )
        statement = statement.order_by(
            _kind_order(),
            Subject.normalized_name,
            DocumentSubjectDecisionModel.id,
        )
        result = await self._session.execute(statement)
        return [
            to_document_subject_decision_record(item)
            for item in result.scalars().all()
        ]

    async def list_document_suggestions(
        self,
        *,
        document_id: uuid.UUID,
    ) -> list[DocumentSubjectDecisionRecord]:
        statement = (
            select(DocumentSubjectDecisionModel)
            .join(Subject, Subject.id == DocumentSubjectDecisionModel.subject_id)
            .where(
                DocumentSubjectDecisionModel.document_id == document_id,
                DocumentSubjectDecisionModel.state == DecisionState.SUGGESTED.value,
                DocumentSubjectDecisionModel.control_source
                == DecisionControlSource.AUTOMATIC.value,
                Subject.archived_at.is_(None),
            )
            .order_by(
                _kind_order(),
                Subject.normalized_name,
                DocumentSubjectDecisionModel.id,
            )
        )
        result = await self._session.execute(statement)
        return [
            to_document_subject_decision_record(item)
            for item in result.scalars().all()
        ]

    async def write_decision(
        self,
        decision: DocumentSubjectDecision,
        *,
        expected_revision: int | None = None,
    ) -> DocumentSubjectDecisionRecord:
        if expected_revision is not None and expected_revision < 0:
            raise ValueError("expected_revision must not be negative.")
        if (
            decision.control_source is DecisionControlSource.MANUAL
            and expected_revision is None
        ):
            raise ValueError("manual decisions require expected_revision.")
        # Archived subjects remain readable with their historical decisions but
        # cannot receive new or updated decisions.
        await self._require_subject(decision.subject_id)
        if decision.classified_document_version_id is not None:
            await self._require_classified_version(decision)

        if expected_revision is None:
            model = await self._blind_upsert_decision(decision)
            if model is not None:
                return to_document_subject_decision_record(model)
            return await self._resolve_automatic_noop_or_conflict(decision)
        elif expected_revision == 0:
            model = await self._insert_new_decision(decision)
        else:
            model = await self._update_expected_decision(
                decision,
                expected_revision=expected_revision,
            )
        if model is not None:
            return to_document_subject_decision_record(model)
        raise SubjectDecisionConflictError(
            "Document-subject decision did not match the expected revision."
        )

    async def list_assigned_document_ids(
        self,
        *,
        subject_ids: tuple[uuid.UUID, ...],
        require_all: bool = False,
    ) -> tuple[uuid.UUID, ...]:
        unique_subject_ids = tuple(dict.fromkeys(subject_ids))
        if not unique_subject_ids:
            return ()
        statement = (
            select(DocumentSubjectDecisionModel.document_id)
            .where(
                DocumentSubjectDecisionModel.subject_id.in_(unique_subject_ids),
                DocumentSubjectDecisionModel.state == DecisionState.ASSIGNED.value,
            )
            .group_by(DocumentSubjectDecisionModel.document_id)
        )
        if require_all:
            statement = statement.having(
                func.count(func.distinct(DocumentSubjectDecisionModel.subject_id))
                == len(unique_subject_ids),
            )
        statement = statement.order_by(DocumentSubjectDecisionModel.document_id)
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())

    async def _blind_upsert_decision(
        self,
        decision: DocumentSubjectDecision,
    ) -> DocumentSubjectDecisionModel | None:
        values = _decision_values(decision)
        insert_statement = postgresql_insert(DocumentSubjectDecisionModel).values(
            id=uuid.uuid4(),
            revision=1,
            **values,
        )
        excluded = insert_statement.excluded
        different = _decision_is_distinct_from_excluded(excluded)
        statement = (
            insert_statement.on_conflict_do_update(
                constraint="uq_document_subject_decisions_document_subject",
                set_={
                    **{key: getattr(excluded, key) for key in values if key not in {"document_id", "subject_id"}},
                    "revision": DocumentSubjectDecisionModel.revision + 1,
                    "updated_at": func.now(),
                },
                where=and_(_automatic_write_is_allowed(decision), different),
            )
            .returning(DocumentSubjectDecisionModel)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def _insert_new_decision(
        self,
        decision: DocumentSubjectDecision,
    ) -> DocumentSubjectDecisionModel | None:
        statement = (
            postgresql_insert(DocumentSubjectDecisionModel)
            .values(id=uuid.uuid4(), revision=1, **_decision_values(decision))
            .on_conflict_do_nothing(
                constraint="uq_document_subject_decisions_document_subject",
            )
            .returning(DocumentSubjectDecisionModel)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def _update_expected_decision(
        self,
        decision: DocumentSubjectDecision,
        *,
        expected_revision: int,
    ) -> DocumentSubjectDecisionModel | None:
        values = _decision_values(decision)
        statement = (
            update(DocumentSubjectDecisionModel)
            .where(
                DocumentSubjectDecisionModel.document_id == decision.document_id,
                DocumentSubjectDecisionModel.subject_id == decision.subject_id,
                DocumentSubjectDecisionModel.revision == expected_revision,
                _automatic_write_is_allowed(decision),
            )
            .values(
                **{key: value for key, value in values.items() if key not in {"document_id", "subject_id"}},
                revision=DocumentSubjectDecisionModel.revision + 1,
                updated_at=func.now(),
            )
            .returning(DocumentSubjectDecisionModel)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def _resolve_automatic_noop_or_conflict(
        self,
        decision: DocumentSubjectDecision,
    ) -> DocumentSubjectDecisionRecord:
        current = await self.get_decision(
            document_id=decision.document_id,
            subject_id=decision.subject_id,
        )
        if current is not None and _same_decision(current, decision):
            return current
        raise SubjectDecisionConflictError(
            "Document-subject decision changed concurrently or is controlled by "
            "an authoritative manual assignment/rejection."
        )

    async def _require_classified_version(
        self,
        decision: DocumentSubjectDecision,
    ) -> None:
        result = await self._session.execute(
            select(DocumentVersion.id).where(
                DocumentVersion.id == decision.classified_document_version_id,
                DocumentVersion.document_id == decision.document_id,
            ),
        )
        if result.scalar_one_or_none() is None:
            raise ValueError(
                "classified_document_version_id must belong to the decision document."
            )

    async def _require_subject(
        self,
        subject_id: uuid.UUID,
        *,
        include_aliases: bool = False,
    ) -> Subject:
        statement = select(Subject).where(
            Subject.id == subject_id,
            Subject.archived_at.is_(None),
        ).with_for_update()
        if include_aliases:
            statement = statement.options(selectinload(Subject.aliases))
        result = await self._session.execute(statement)
        subject = result.scalar_one_or_none()
        if subject is None:
            raise LookupError(f"Active subject {subject_id} was not found.")
        return subject


def _decision_values(decision: DocumentSubjectDecision) -> dict[str, Any]:
    return {
        "document_id": decision.document_id,
        "subject_id": decision.subject_id,
        "state": decision.state.value,
        "control_source": decision.control_source.value,
        "confidence": decision.confidence,
        "confidence_band": (
            decision.confidence_band.value
            if decision.confidence_band is not None
            else None
        ),
        "rationale": decision.rationale,
        "classifier_version": decision.classifier_version,
        "policy_version": decision.policy_version,
        "signals": dict(decision.signals),
        "classified_document_version_id": decision.classified_document_version_id,
    }


def _decision_is_distinct_from_excluded(excluded: Any) -> Any:
    compared = (
        "state",
        "control_source",
        "confidence",
        "confidence_band",
        "rationale",
        "classifier_version",
        "policy_version",
        "signals",
        "classified_document_version_id",
    )
    return or_(
        *(
            getattr(DocumentSubjectDecisionModel, name).is_distinct_from(
                getattr(excluded, name),
            )
            for name in compared
        ),
    )


def _automatic_write_is_allowed(decision: DocumentSubjectDecision) -> Any:
    if decision.control_source is DecisionControlSource.MANUAL:
        return True
    return not_(
        and_(
            DocumentSubjectDecisionModel.control_source
            == DecisionControlSource.MANUAL.value,
            DocumentSubjectDecisionModel.state.in_(
                (DecisionState.ASSIGNED.value, DecisionState.REJECTED.value),
            ),
        ),
    )


def _same_decision(
    current: DocumentSubjectDecisionRecord,
    proposed: DocumentSubjectDecision,
) -> bool:
    return (
        current.state is proposed.state
        and current.control_source is proposed.control_source
        and current.confidence == proposed.confidence
        and current.confidence_band is proposed.confidence_band
        and current.rationale == proposed.rationale
        and current.classifier_version == proposed.classifier_version
        and current.policy_version == proposed.policy_version
        and current.signals == dict(proposed.signals)
        and current.classified_document_version_id
        == proposed.classified_document_version_id
    )


def _to_alias_record(alias: SubjectAlias) -> SubjectAliasRecord:
    return SubjectAliasRecord(
        id=alias.id,
        subject_id=alias.subject_id,
        name=alias.name,
        normalized_name=alias.normalized_name,
        created_at=alias.created_at,
        archived_at=alias.archived_at,
    )


def _optional_description(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    if len(cleaned) > 2_000:
        raise ValueError("subject description must not exceed 2000 characters.")
    return cleaned


def _kind_order() -> Any:
    return case(
        (Subject.kind == SubjectKind.PROJECT.value, 0),
        (Subject.kind == SubjectKind.TOPIC.value, 1),
        (Subject.kind == SubjectKind.ORGANIZATION.value, 2),
        else_=3,
    )


def _kind_rank(kind: str) -> int:
    return {
        SubjectKind.PROJECT.value: 0,
        SubjectKind.TOPIC.value: 1,
        SubjectKind.ORGANIZATION.value: 2,
        SubjectKind.CUSTOM.value: 3,
    }[kind]


def _raise_canonical_name_conflict(
    error: IntegrityError,
    *,
    kind: SubjectKind,
) -> None:
    if _integrity_constraint_name(error) == "uq_subjects_kind_normalized_name":
        raise SubjectCanonicalNameConflictError(
            f"A {kind.value} subject with that canonical name already exists."
        ) from error
    raise error


def _integrity_constraint_name(error: IntegrityError) -> str | None:
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
        candidate = getattr(candidate, "__cause__", None) or getattr(
            candidate,
            "__context__",
            None,
        )
    return None
