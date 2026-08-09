from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from packages.indexer_application.dto import (
    DocumentRecord,
    DocumentVersionRecord,
    DocumentVersionStatus,
)
from packages.indexer_application.ports import DocumentOrganizationConflictError, UnitOfWork
from packages.indexer_application.services.background_jobs.execution import ProgressReporter
from packages.rag_core.document_organization import (
    ClassificationConfidenceBand,
    ClassificationSource,
    ContentGroupAssignmentState,
    ContentGroupModelProvider,
    ContentGroupName,
    DocumentContentGroupAssignment,
    DocumentOrganizationPolicy,
    DocumentTypeCandidate,
    DocumentTypeDecision,
    DocumentTypeDecisionState,
    DocumentTypeModelProvider,
    GroupCatalogueCandidate,
    GroupContentCandidate,
    GroupCreateResolution,
    GroupReuseResolution,
    GroupUnresolvedResolution,
    ORGANIZATION_CLASSIFIER_VERSION,
    select_document_types,
)


@dataclass(frozen=True, slots=True)
class DocumentOrganizationJobConfig:
    policy: DocumentOrganizationPolicy
    classifier_version: str = ORGANIZATION_CLASSIFIER_VERSION
    max_summary_chars: int = 2_000
    max_representative_documents_per_group: int = 3
    max_representative_assignments_scanned_per_group: int = 12
    max_representative_chars_per_group: int = 3_000


@dataclass(frozen=True, slots=True)
class _GroupPlan:
    content_group_id: uuid.UUID | None
    create_name: str | None
    state: ContentGroupAssignmentState
    confidence: float | None
    confidence_band: ClassificationConfidenceBand | None
    unresolved_reason: str | None
    signals: dict[str, Any]


class ClassifyDocumentOrganizationJobHandler:
    def __init__(
        self,
        *,
        uow: UnitOfWork,
        config: DocumentOrganizationJobConfig,
        type_provider: DocumentTypeModelProvider | None,
        group_provider: ContentGroupModelProvider | None,
    ) -> None:
        self._uow = uow
        self._config = config
        self._type_provider = type_provider
        self._group_provider = group_provider

    async def __call__(
        self,
        payload: dict[str, object],
        report: ProgressReporter,
        *,
        job_id: uuid.UUID,
    ) -> dict[str, object]:
        document_id = uuid.UUID(_required_string(payload, "document_id"))
        version_id = uuid.UUID(_required_string(payload, "document_version_id"))
        policy_version = _required_string(payload, "policy_version")
        if policy_version != self._config.policy.policy_version:
            raise ValueError("Organization job policy version does not match this worker.")

        await report(0.08, "loading_document_organization_catalogues")
        document = await self._uow.documents.get(document_id)
        if document is None:
            raise LookupError(f"Document {document_id} was not found.")
        latest = _latest_ready(document)
        if latest is None or latest.id != version_id:
            return _terminal(document_id, version_id, "superseded")
        summary = _root_summary(latest, self._config.max_summary_chars)
        summary_hash = _summary_hash(summary)

        type_records = await self._uow.document_types.list(limit=500)
        type_candidates = tuple(
            DocumentTypeCandidate(item.id, item.key, item.label, item.description)
            for item in type_records
        )
        type_scores: dict[str, float] | None = None
        if summary:
            if self._type_provider is None:
                raise RuntimeError("Document type model provider is unavailable.")
            await report(0.24, "classifying_document_types")
            type_scores = {
                item.key: item.confidence
                for item in await self._type_provider.classify_types(
                    title=document.title,
                    filename=document.original_filename,
                    summary=summary,
                    candidates=type_candidates,
                )
            }

        group_plan = _GroupPlan(
            content_group_id=None,
            create_name=None,
            state=ContentGroupAssignmentState.UNRESOLVED,
            confidence=None,
            confidence_band=None,
            unresolved_reason="no_summary" if not summary else "provider_unavailable",
            signals={},
        )
        if summary:
            if self._group_provider is None:
                raise RuntimeError("Content group model provider is unavailable.")
            await report(0.38, "resolving_content_group")
            group_plan = await self._resolve_group_with_catalogue_retry(
                document_id=document_id,
                summary=summary,
            )

        await report(0.62, "locking_document_organization_target")
        locked = await self._uow.documents.get_for_update(document_id)
        if locked is None:
            raise LookupError(f"Document {document_id} was not found.")
        locked_latest = _latest_ready(locked)
        if locked_latest is None or locked_latest.id != version_id:
            return _terminal(document_id, version_id, "superseded")
        locked_summary = _root_summary(locked_latest, self._config.max_summary_chars)
        if _summary_hash(locked_summary) != summary_hash:
            raise RuntimeError("Organization summary changed before locked classification.")
        owns_status = await self._uow.documents.set_organization_classification_status(
            document_id=document_id,
            status={
                "status": "running",
                "job_id": str(job_id),
                "document_version_id": str(version_id),
                "policy_version": policy_version,
            },
            expected_job_id=job_id,
            expected_document_version_id=version_id,
            expected_policy_version=policy_version,
            expected_statuses=("queued", "running", "retry_scheduled"),
        )
        if not owns_status:
            return _terminal(document_id, version_id, "stale")

        await report(0.70, "persisting_document_type_decisions")
        selected_type_ids: set[uuid.UUID] = set()
        type_write_count = 0
        if type_scores is not None:
            outcomes = select_document_types(
                type_scores,
                type_candidates,
                policy=self._config.policy,
            )
            if not outcomes:
                fallback = next((item for item in type_candidates if item.key == "other"), None)
                if fallback is None:
                    raise RuntimeError("Active document type catalogue has no 'other' fallback.")
                outcomes = (
                    _fallback_type_outcome(fallback.id),
                )
            selected_type_ids = {item.document_type_id for item in outcomes}
            current = await self._uow.document_types.list_document_decisions(document_id=document_id)
            current_by_type = {item.document_type_id: item for item in current}
            for outcome in outcomes:
                existing = current_by_type.get(outcome.document_type_id)
                if existing is not None and existing.source is ClassificationSource.MANUAL:
                    continue
                try:
                    await self._uow.document_types.write_decision(
                        DocumentTypeDecision(
                            document_id=document_id,
                            document_type_id=outcome.document_type_id,
                            state=DocumentTypeDecisionState.ASSIGNED,
                            source=ClassificationSource.AUTOMATIC,
                            confidence=outcome.confidence,
                            confidence_band=outcome.confidence_band,
                            rationale="Structured title, filename, and root-summary classification.",
                            classifier_version=self._config.classifier_version,
                            policy_version=policy_version,
                            signals={"root_summary_hash": summary_hash, "catalogue_key_supported": True},
                            classified_document_version_id=version_id,
                        )
                    )
                    type_write_count += 1
                except DocumentOrganizationConflictError:
                    pass
            for existing in current:
                if (
                    existing.source is ClassificationSource.AUTOMATIC
                    and existing.document_type_id not in selected_type_ids
                ):
                    await self._uow.document_types.write_decision(
                        DocumentTypeDecision(
                            document_id=document_id,
                            document_type_id=existing.document_type_id,
                            state=DocumentTypeDecisionState.REJECTED,
                            source=ClassificationSource.AUTOMATIC,
                            confidence=0.0,
                            confidence_band=ClassificationConfidenceBand.LOW,
                            rationale="The current classifier no longer selected this document type.",
                            classifier_version=self._config.classifier_version,
                            policy_version=policy_version,
                            signals={"root_summary_hash": summary_hash},
                            classified_document_version_id=version_id,
                        )
                    )
                    type_write_count += 1

        await report(0.82, "persisting_content_group_assignment")
        assignment_write_count = 0
        existing_assignment = await self._uow.content_groups.get_assignment(document_id)
        if existing_assignment is None or existing_assignment.source is not ClassificationSource.MANUAL:
            content_group_id = group_plan.content_group_id
            if group_plan.create_name is not None:
                group, _ = await self._uow.content_groups.create_or_get_canonical(
                    name=group_plan.create_name,
                    metadata={
                        "created_by": "automatic_document_organization",
                        "classifier_version": self._config.classifier_version,
                        "policy_version": policy_version,
                    },
                )
                content_group_id = group.id
            assignment = DocumentContentGroupAssignment(
                document_id=document_id,
                content_group_id=content_group_id,
                state=group_plan.state,
                source=ClassificationSource.AUTOMATIC,
                unresolved_reason=group_plan.unresolved_reason,
                confidence=group_plan.confidence,
                confidence_band=group_plan.confidence_band,
                rationale=(
                    "Root-summary content was classified into one content group."
                    if group_plan.state in {ContentGroupAssignmentState.ASSIGNED, ContentGroupAssignmentState.SUGGESTED}
                    else None
                ),
                classifier_version=self._config.classifier_version,
                policy_version=policy_version,
                signals={**group_plan.signals, **({"root_summary_hash": summary_hash} if summary_hash else {})},
                summary_hash=summary_hash,
                classified_document_version_id=version_id,
            )
            try:
                await self._uow.content_groups.write_assignment(assignment)
                assignment_write_count = 1
            except DocumentOrganizationConflictError:
                pass

        status = {
            "status": "succeeded",
            "job_id": str(job_id),
            "document_version_id": str(version_id),
            "policy_version": policy_version,
            "classifier_version": self._config.classifier_version,
            "assigned_type_count": len(selected_type_ids),
            "content_group_state": group_plan.state.value,
        }
        updated = await self._uow.documents.set_organization_classification_status(
            document_id=document_id,
            status=status,
            expected_job_id=job_id,
            expected_document_version_id=version_id,
            expected_policy_version=policy_version,
            expected_statuses=("running",),
        )
        if not updated:
            raise RuntimeError("Organization classification status ownership changed while locked.")
        await report(0.96, "document_organization_complete")
        return {
            "document_id": str(document_id),
            **status,
            "type_decision_write_count": type_write_count,
            "assignment_write_count": assignment_write_count,
        }

    async def _resolve_group_with_catalogue_retry(
        self,
        *,
        document_id: uuid.UUID,
        summary: str,
    ) -> _GroupPlan:
        assert self._group_provider is not None
        for attempt in range(2):
            groups = await self._uow.content_groups.list(limit=500)
            revision = _catalogue_revision(groups)
            representatives = await _representatives(
                self._uow,
                groups=groups,
                current_document_id=document_id,
                config=self._config,
            )
            confirmed = await _confirmed_content_match(
                self._group_provider,
                summary=summary,
                candidates=_shortlist(summary, representatives, limit=3),
                policy=self._config.policy,
            )
            if confirmed is not None:
                plan = _GroupPlan(
                    content_group_id=confirmed[0].content_group_id,
                    create_name=None,
                    state=ContentGroupAssignmentState.ASSIGNED,
                    confidence=confirmed[1],
                    confidence_band=_band(confirmed[1], self._config.policy),
                    unresolved_reason=None,
                    signals={
                        "resolution": "confirmed_representative_content",
                        "selection_confidence": confirmed[2],
                        "confirmation_confidence": confirmed[3],
                        "representative_content_hash": confirmed[0].representative_content_hash,
                        "representative_document_count": confirmed[0].document_count,
                    },
                )
            else:
                resolution = await self._group_provider.resolve_group(
                    summary,
                    tuple(
                        GroupCatalogueCandidate(group.id, group.name, tuple(alias.name for alias in group.aliases))
                        for group in groups
                    ),
                )
                if isinstance(resolution, GroupCreateResolution):
                    if resolution.confidence < self._config.policy.medium_threshold or _role_only_label(resolution.name):
                        plan = _unresolved("invalid_or_low_confidence_group_label")
                    else:
                        ContentGroupName.from_automatic_proposal(resolution.name)
                        plan = _GroupPlan(
                            content_group_id=None,
                            create_name=resolution.name,
                            state=ContentGroupAssignmentState.ASSIGNED,
                            confidence=resolution.confidence,
                            confidence_band=_band(resolution.confidence, self._config.policy),
                            unresolved_reason=None,
                            signals={"resolution": "created_from_root_summary"},
                        )
                elif isinstance(resolution, GroupReuseResolution):
                    plan = _unresolved("reuse_requires_content_confirmation")
                else:
                    assert isinstance(resolution, GroupUnresolvedResolution)
                    plan = _unresolved(_safe_reason(resolution.reason))
            current_groups = await self._uow.content_groups.list(limit=500)
            if revision == _catalogue_revision(current_groups):
                return plan
            if attempt == 1:
                return _unresolved("catalogue_changed_during_classification")
        raise AssertionError("unreachable")


async def update_document_organization_job_status(
    *,
    uow: UnitOfWork,
    document_id: uuid.UUID,
    job_id: uuid.UUID,
    document_version_id: uuid.UUID,
    policy_version: str,
    status: str,
    error_message: str | None = None,
) -> bool:
    payload: dict[str, Any] = {
        "status": status,
        "job_id": str(job_id),
        "document_version_id": str(document_version_id),
        "policy_version": policy_version,
    }
    if error_message:
        payload["error_message"] = error_message[:2_000]
    return await uow.documents.set_organization_classification_status(
        document_id=document_id,
        status=payload,
        expected_job_id=job_id,
        expected_document_version_id=document_version_id,
        expected_policy_version=policy_version,
        expected_statuses=("queued", "running", "retry_scheduled"),
    )


async def _representatives(
    uow: UnitOfWork,
    *,
    groups,
    current_document_id: uuid.UUID,
    config: DocumentOrganizationJobConfig,
) -> tuple[GroupContentCandidate, ...]:
    values: list[GroupContentCandidate] = []
    for group in groups:
        assignments = await uow.content_groups.list_assignments(
            content_group_id=group.id,
            states=(ContentGroupAssignmentState.ASSIGNED,),
            limit=config.max_representative_assignments_scanned_per_group,
        )
        summaries: list[str] = []
        for assignment in assignments:
            if assignment.document_id == current_document_id:
                continue
            document = await uow.documents.get(assignment.document_id)
            latest = _latest_ready(document) if document else None
            if latest is None:
                continue
            summary = _root_summary(latest, config.max_summary_chars)
            if summary:
                summaries.append(summary)
            if len(summaries) >= config.max_representative_documents_per_group:
                break
        if summaries:
            content = "\n".join(summaries)[:config.max_representative_chars_per_group]
            values.append(GroupContentCandidate(
                group.id,
                content,
                hashlib.sha256(content.encode()).hexdigest()[:24],
                len(summaries),
            ))
    return tuple(values)


async def _confirmed_content_match(provider, *, summary, candidates, policy):
    confirmed: list[tuple[GroupContentCandidate, float, float, float]] = []
    for candidate in candidates:
        selection = await provider.match_group_content(summary, (candidate,))
        if selection is None or selection.content_group_id != candidate.content_group_id or selection.confidence < policy.medium_threshold:
            continue
        confirmation = await provider.match_group_content(summary, (candidate,), pairwise_confirmation=True)
        if confirmation is None or confirmation.content_group_id != candidate.content_group_id or confirmation.confidence < policy.medium_threshold:
            continue
        confirmed.append((candidate, min(selection.confidence, confirmation.confidence), selection.confidence, confirmation.confidence))
    confirmed.sort(key=lambda item: (-item[1], str(item[0].content_group_id)))
    if not confirmed:
        return None
    runner = confirmed[1][1] if len(confirmed) > 1 else 0.0
    return confirmed[0] if confirmed[0][1] - runner >= policy.confirmation_margin else None


_GENERIC_CONTENT = {
    "a", "an", "and", "for", "in", "of", "on", "the", "to", "with",
    "plan", "report", "realization", "specification", "presentation", "notes",
    "reference", "document", "implementation", "technology", "system", "project",
}
_BANNED_GROUP_ROLE_TERMS = {
    "plan", "report", "realization", "specification", "presentation", "notes", "reference",
}


def _shortlist(summary: str, candidates: tuple[GroupContentCandidate, ...], *, limit: int):
    root = _tokens(summary)
    ranked = []
    for candidate in candidates:
        candidate_tokens = _tokens(candidate.representative_content)
        shared = root & candidate_tokens
        if len(shared) >= 2:
            union = root | candidate_tokens
            ranked.append((len(shared), len(shared) / len(union), str(candidate.content_group_id), candidate))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(item[3] for item in ranked[:limit])


def _tokens(value: str) -> set[str]:
    normalized = _normalize(value)
    return {token for token in normalized.split() if token not in _GENERIC_CONTENT and len(token) > 1}


def _normalize(value: str) -> str:
    import re
    import unicodedata
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", folded).split())


def _role_only_label(value: str) -> bool:
    tokens = set(_normalize(value).split())
    return bool(tokens & _BANNED_GROUP_ROLE_TERMS)


def _catalogue_revision(groups) -> str:
    material = "\n".join(
        f"{group.id}:{group.normalized_name}:" + ",".join(alias.normalized_name for alias in group.aliases)
        for group in groups
    )
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def _root_summary(version: DocumentVersionRecord, maximum: int) -> str | None:
    for key in ("hierarchical_retrieval", "contextualization"):
        container = version.metadata.get(key)
        if isinstance(container, dict):
            value = container.get("document_summary")
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:maximum]
    return None


def _latest_ready(document: DocumentRecord | None) -> DocumentVersionRecord | None:
    if document is None:
        return None
    return max(
        (version for version in document.versions if version.status is DocumentVersionStatus.READY),
        key=lambda version: version.version_number,
        default=None,
    )


def _summary_hash(summary: str | None) -> str | None:
    return hashlib.sha256(summary.encode()).hexdigest() if summary else None


def _band(confidence: float, policy: DocumentOrganizationPolicy) -> ClassificationConfidenceBand:
    return ClassificationConfidenceBand.HIGH if confidence >= policy.high_threshold else ClassificationConfidenceBand.MEDIUM


def _fallback_type_outcome(document_type_id: uuid.UUID):
    from packages.rag_core.document_organization import DocumentTypeOutcome
    return DocumentTypeOutcome(document_type_id, DocumentTypeDecisionState.ASSIGNED, 1.0, ClassificationConfidenceBand.HIGH)


def _unresolved(reason: str) -> _GroupPlan:
    return _GroupPlan(None, None, ContentGroupAssignmentState.UNRESOLVED, None, None, reason, {})


def _safe_reason(value: str) -> str:
    cleaned = " ".join(value.split())[:1000]
    return cleaned or "model_unresolved"


def _terminal(document_id: uuid.UUID, version_id: uuid.UUID, status: str) -> dict[str, object]:
    return {
        "document_id": str(document_id),
        "document_version_id": str(version_id),
        "status": status,
        "assigned_type_count": 0,
        "type_decision_write_count": 0,
        "assignment_write_count": 0,
    }


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
