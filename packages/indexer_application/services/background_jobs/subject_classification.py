from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from packages.indexer_application.dto import (
    DocumentRecord,
    DocumentSubjectDecisionRecord,
    DocumentVersionStatus,
    SubjectRecord,
)
from packages.indexer_application.ports import SubjectDecisionConflictError, UnitOfWork
from packages.indexer_application.services.background_jobs.execution import ProgressReporter
from packages.rag_core.subjects import (
    CLASSIFIER_VERSION,
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SignalFamily,
    DocumentSubjectDecision,
    SubjectClassificationCandidate,
    SubjectClassificationInput,
    SubjectClassificationOutcome,
    SubjectClassificationPolicy,
    SubjectDiscoveryProposal,
    SubjectDiscoveryProvider,
    SubjectName,
    SubjectNameMatchType,
    SubjectModelEvidenceProvider,
    corroborating_subject_name_families,
    classify_document_subjects,
)


@dataclass(frozen=True, slots=True)
class SubjectClassificationJobConfig:
    policy: SubjectClassificationPolicy
    classifier_version: str = CLASSIFIER_VERSION
    max_summary_chars: int = 2_000
    discovery_enabled: bool = True


class ClassifyDocumentSubjectsJobHandler:
    def __init__(
        self,
        *,
        uow: UnitOfWork,
        config: SubjectClassificationJobConfig,
        model_evidence: SubjectModelEvidenceProvider | None = None,
        subject_discovery: SubjectDiscoveryProvider | None = None,
    ) -> None:
        self._uow = uow
        self._config = config
        self._model_evidence = model_evidence
        self._subject_discovery = subject_discovery

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
            raise ValueError(
                "Classification job policy version does not match this worker."
            )

        await report(0.08, "loading_document_and_subjects")
        document = await self._uow.documents.get(document_id)
        if document is None:
            raise LookupError(f"Document {document_id} was not found.")
        latest_ready = max(
            (
                version
                for version in document.versions
                if version.status is DocumentVersionStatus.READY
            ),
            key=lambda version: version.version_number,
            default=None,
        )
        if latest_ready is None:
            raise ValueError("The document has no ready version to classify.")
        if latest_ready.id != version_id:
            return {
                "document_id": str(document_id),
                "document_version_id": str(version_id),
                "status": "superseded",
                "assigned_count": 0,
                "suggested_count": 0,
                "review_required_count": 0,
                "decision_write_count": 0,
                "manual_decision_skip_count": 0,
            }

        subjects = await _active_subjects(self._uow)
        candidates = tuple(_candidate(subject) for subject in subjects)
        summary = _document_summary(latest_ready.metadata, self._config.max_summary_chars)
        model_scores: dict[uuid.UUID, float] = {}
        if self._model_evidence is not None and summary and candidates:
            await report(0.30, "scoring_document_summary")
            model_scores = await self._model_evidence.score(summary, candidates)

        initial_input = _classification_input(document, summary, model_scores)
        initial_outcomes = classify_document_subjects(
            initial_input,
            candidates,
            policy=self._config.policy,
        )
        discovery_proposal: SubjectDiscoveryProposal | None = None
        discovery = _discovery_observation(
            status="skipped",
            reason=(
                "disabled"
                if not self._config.discovery_enabled
                else "provider_unavailable"
                if self._subject_discovery is None
                else "summary_unavailable"
                if not summary
                else "existing_subject_evidence"
                if initial_outcomes
                else "no_proposal"
            ),
        )
        if (
            self._config.discovery_enabled
            and self._subject_discovery is not None
            and summary
            and not initial_outcomes
        ):
            await report(0.40, "discovering_document_subject")
            discovery_proposal = await self._subject_discovery.discover(summary)
            if discovery_proposal is not None:
                discovery = _evaluate_discovery_proposal(
                    discovery_proposal,
                    request=initial_input,
                    policy=self._config.policy,
                )

        # Model scoring is intentionally outside the document lock. Before any
        # decision or status write, serialize with activation on the aggregate
        # row and re-read the latest ready version.
        await report(0.48, "locking_classification_target")
        locked_document = await self._uow.documents.get_for_update(document_id)
        if locked_document is None:
            raise LookupError(f"Document {document_id} was not found.")
        locked_latest_ready = max(
            (
                version
                for version in locked_document.versions
                if version.status is DocumentVersionStatus.READY
            ),
            key=lambda version: version.version_number,
            default=None,
        )
        if locked_latest_ready is None or locked_latest_ready.id != version_id:
            return {
                "document_id": str(document_id),
                "document_version_id": str(version_id),
                "status": "superseded",
                "assigned_count": 0,
                "suggested_count": 0,
                "review_required_count": 0,
                "decision_write_count": 0,
                "manual_decision_skip_count": 0,
            }
        owns_status = await self._uow.documents.set_subject_classification_status(
            document_id=document_id,
            status={
                "status": "running",
                "job_id": str(job_id),
                "document_version_id": str(version_id),
                "policy_version": self._config.policy.policy_version,
            },
            expected_job_id=job_id,
            expected_document_version_id=version_id,
            expected_policy_version=self._config.policy.policy_version,
            expected_statuses=("queued", "running", "retry_scheduled"),
        )
        if not owns_status:
            return {
                "document_id": str(document_id),
                "document_version_id": str(version_id),
                "status": "stale",
                "assigned_count": 0,
                "suggested_count": 0,
                "review_required_count": 0,
                "decision_write_count": 0,
                "manual_decision_skip_count": 0,
            }

        await report(0.52, "applying_subject_decision_policy")
        locked_summary = _document_summary(
            locked_latest_ready.metadata,
            self._config.max_summary_chars,
        )
        locked_input = _classification_input(
            locked_document,
            locked_summary,
            model_scores,
        )
        locked_existing_outcomes = classify_document_subjects(
            locked_input,
            candidates,
            policy=self._config.policy,
        )
        discovered_subject: SubjectRecord | None = None
        if discovery_proposal is not None and discovery.get("status") == "eligible":
            if locked_existing_outcomes:
                discovery = _discovery_observation(
                    status="skipped",
                    reason="existing_subject_evidence_after_lock",
                    proposal=discovery_proposal,
                    corroborating_families=corroborating_subject_name_families(
                        locked_input,
                        discovery_proposal.name,
                    ),
                )
            else:
                locked_evaluation = _evaluate_discovery_proposal(
                    discovery_proposal,
                    request=locked_input,
                    policy=self._config.policy,
                )
                if locked_evaluation.get("status") != "eligible":
                    discovery = locked_evaluation
                else:
                    discovered_subject, discovery = await _resolve_discovered_subject(
                        self._uow,
                        proposal=discovery_proposal,
                        observation=locked_evaluation,
                        config=self._config,
        )
        if discovered_subject is not None:
            assert discovery_proposal is not None
            if all(subject.id != discovered_subject.id for subject in subjects):
                subjects.append(discovered_subject)
                candidates = (*candidates, _candidate(discovered_subject))
            model_scores = {
                **model_scores,
                discovered_subject.id: discovery_proposal.confidence,
            }
            locked_input = _classification_input(
                locked_document,
                locked_summary,
                model_scores,
            )
        outcomes = classify_document_subjects(
            locked_input,
            candidates,
            policy=self._config.policy,
        )
        outcomes_by_subject = {outcome.subject_id: outcome for outcome in outcomes}
        current = {
            decision.subject_id: decision
            for decision in await self._uow.subjects.list_document_decisions(
                document_id=document_id,
            )
        }
        active_subject_ids = {subject.id for subject in subjects}
        effective: dict[uuid.UUID, DocumentSubjectDecisionRecord | DocumentSubjectDecision] = {
            subject_id: decision
            for subject_id, decision in current.items()
            if subject_id in active_subject_ids
        }

        written = 0
        manual_skipped = 0
        await report(0.68, "persisting_subject_decisions")
        for subject in subjects:
            existing = current.get(subject.id)
            if existing is not None and existing.control_source is DecisionControlSource.MANUAL:
                manual_skipped += 1
                continue
            outcome = outcomes_by_subject.get(subject.id)
            if outcome is None and not _is_automatic_assignment(existing):
                continue
            proposed = _decision_for(
                document_id=document_id,
                version_id=version_id,
                subject_id=subject.id,
                outcome=outcome,
                existing=existing,
                config=self._config,
                discovery=(
                    discovery
                    if discovered_subject is not None
                    and subject.id == discovered_subject.id
                    else None
                ),
            )
            if existing is not None and _same_decision(existing, proposed):
                continue
            try:
                await self._uow.subjects.write_decision(
                    proposed,
                    expected_revision=existing.revision if existing is not None else 0,
                )
            except SubjectDecisionConflictError:
                raced = await self._uow.subjects.get_decision(
                    document_id=document_id,
                    subject_id=subject.id,
                )
                if raced is not None and raced.control_source is DecisionControlSource.MANUAL:
                    manual_skipped += 1
                    effective[subject.id] = raced
                    continue
                raise
            effective[subject.id] = proposed
            written += 1

        assigned_count = sum(
            decision.state is DecisionState.ASSIGNED
            for decision in effective.values()
        )
        suggested_count = sum(
            decision.state is DecisionState.SUGGESTED
            for decision in effective.values()
        )
        review_required = sum(
            decision.signals.get("review_recommended") is True
            for decision in effective.values()
        )
        status = {
            "status": "succeeded",
            "job_id": str(job_id),
            "document_version_id": str(version_id),
            "policy_version": self._config.policy.policy_version,
            "classifier_version": self._config.classifier_version,
            "assigned_count": assigned_count,
            "suggested_count": suggested_count,
            "review_required_count": review_required,
            "discovery": discovery,
        }
        status_updated = await self._uow.documents.set_subject_classification_status(
            document_id=document_id,
            status=status,
            expected_job_id=job_id,
            expected_document_version_id=version_id,
            expected_policy_version=self._config.policy.policy_version,
            expected_statuses=("running",),
        )
        if not status_updated:
            raise RuntimeError("Classification status ownership changed while locked.")
        await report(0.96, "subject_classification_complete")
        return {
            "document_id": str(document_id),
            **status,
            "decision_write_count": written,
            "manual_decision_skip_count": manual_skipped,
        }


async def update_subject_classification_job_status(
    *,
    uow: UnitOfWork,
    document_id: uuid.UUID,
    job_id: uuid.UUID,
    document_version_id: uuid.UUID,
    policy_version: str,
    status: str,
    error_message: str | None = None,
) -> bool:
    """Conditionally publish a worker retry/failure state for the current job."""

    payload: dict[str, Any] = {
        "status": status,
        "job_id": str(job_id),
        "document_version_id": str(document_version_id),
        "policy_version": policy_version,
    }
    if error_message is not None:
        payload["error_message"] = error_message[:2_000]
    return await uow.documents.set_subject_classification_status(
        document_id=document_id,
        status=payload,
        expected_job_id=job_id,
        expected_document_version_id=document_version_id,
        expected_policy_version=policy_version,
        expected_statuses=("queued", "running", "retry_scheduled"),
    )


def _decision_for(
    *,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    subject_id: uuid.UUID,
    outcome: SubjectClassificationOutcome | None,
    existing: DocumentSubjectDecisionRecord | None,
    config: SubjectClassificationJobConfig,
    discovery: dict[str, object] | None = None,
) -> DocumentSubjectDecision:
    preserve_assignment = _is_automatic_assignment(existing) and (
        outcome is None or outcome.state is not DecisionState.ASSIGNED
    )
    confidence = outcome.confidence if outcome is not None else 0.0
    band = outcome.confidence_band if outcome is not None else ConfidenceBand.LOW
    signal_payload: dict[str, Any] = {
        "signal_families": (
            sorted({signal.family.value for signal in outcome.signals})
            if outcome is not None
            else []
        ),
        "signal_labels": (
            [signal.label for signal in outcome.signals] if outcome is not None else []
        ),
        "signal_hashes": (
            [signal.evidence_hash for signal in outcome.signals]
            if outcome is not None
            else []
        ),
        "independent_family_count": (
            outcome.independent_family_count if outcome is not None else 0
        ),
        "margin": outcome.margin if outcome is not None else 0.0,
        "thresholds": {
            "high": config.policy.high_threshold,
            "medium": config.policy.medium_threshold,
            "high_margin": config.policy.high_margin,
            "medium_margin": config.policy.medium_margin,
        },
    }
    if preserve_assignment:
        signal_payload["review_recommended"] = True
    if discovery is not None:
        signal_payload["discovery"] = {
            "source": "structured_document_summary",
            "kind": discovery["kind"],
            "confidence": discovery["confidence"],
            "proposal_name_hash": discovery["proposal_name_hash"],
            "corroborating_signal_families": discovery[
                "corroborating_signal_families"
            ],
        }
    state = (
        DecisionState.ASSIGNED
        if preserve_assignment
        else outcome.state  # type: ignore[union-attr]
    )
    rationale = (
        "Latest automatic evidence weakened; the existing automatic assignment "
        "was retained for review."
        if preserve_assignment
        else "Automatic subject evidence satisfied the assignment policy."
        if state is DecisionState.ASSIGNED
        else "Automatic subject evidence requires human review."
    )
    return DocumentSubjectDecision(
        document_id=document_id,
        subject_id=subject_id,
        state=state,
        control_source=DecisionControlSource.AUTOMATIC,
        confidence=confidence,
        confidence_band=band,
        rationale=rationale,
        classifier_version=config.classifier_version,
        policy_version=config.policy.policy_version,
        signals=signal_payload,
        classified_document_version_id=version_id,
    )


def _is_automatic_assignment(
    decision: DocumentSubjectDecisionRecord | None,
) -> bool:
    return (
        decision is not None
        and decision.control_source is DecisionControlSource.AUTOMATIC
        and decision.state is DecisionState.ASSIGNED
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


async def _active_subjects(uow: UnitOfWork) -> list[SubjectRecord]:
    subjects: list[SubjectRecord] = []
    offset = 0
    while True:
        batch = await uow.subjects.list(limit=100, offset=offset)
        subjects.extend(batch)
        if len(batch) < 100:
            return subjects
        offset += len(batch)


def _classification_input(
    document: DocumentRecord,
    summary: str | None,
    model_scores: dict[uuid.UUID, float],
) -> SubjectClassificationInput:
    return SubjectClassificationInput(
        title=document.title,
        filename=document.original_filename,
        explicit_metadata_values=_explicit_metadata_values(document.metadata),
        document_summary=summary,
        model_scores=model_scores,
    )


def _evaluate_discovery_proposal(
    proposal: SubjectDiscoveryProposal,
    *,
    request: SubjectClassificationInput,
    policy: SubjectClassificationPolicy,
) -> dict[str, object]:
    corroborating_families = corroborating_subject_name_families(
        request,
        proposal.name,
    )
    eligible = proposal.confidence >= policy.high_threshold or (
        proposal.confidence >= policy.medium_threshold
        and bool(corroborating_families)
    )
    reason = None
    if not eligible:
        reason = (
            "below_medium_confidence"
            if proposal.confidence < policy.medium_threshold
            else "medium_confidence_without_name_corroboration"
        )
    return _discovery_observation(
        status="eligible" if eligible else "skipped",
        reason=reason,
        proposal=proposal,
        corroborating_families=corroborating_families,
    )


async def _resolve_discovered_subject(
    uow: UnitOfWork,
    *,
    proposal: SubjectDiscoveryProposal,
    observation: dict[str, object],
    config: SubjectClassificationJobConfig,
) -> tuple[SubjectRecord | None, dict[str, object]]:
    matches = await uow.subjects.resolve_name(
        proposal.name,
        kind=proposal.kind,
        include_archived=True,
    )
    if any(match.subject.archived_at is not None for match in matches):
        return None, {
            **observation,
            "status": "skipped",
            "reason": "archived_name_collision",
        }
    active_by_id = {match.subject.id: match for match in matches}
    if len(active_by_id) > 1:
        return None, {
            **observation,
            "status": "skipped",
            "reason": "ambiguous_name_resolution",
        }
    if active_by_id:
        match = next(iter(active_by_id.values()))
        return match.subject, {
            **observation,
            "status": "reused",
            "reason": None,
            "subject_id": str(match.subject.id),
            "resolution": match.match_type.value,
        }

    subject, created = await uow.subjects.create_or_get_canonical(
        kind=proposal.kind,
        name=proposal.name,
        metadata={
            "created_by": "automatic_subject_discovery",
            "classifier_version": config.classifier_version,
            "policy_version": config.policy.policy_version,
            "proposal_confidence": proposal.confidence,
            "proposal_name_hash": observation["proposal_name_hash"],
        },
    )
    if subject.archived_at is not None:
        return None, {
            **observation,
            "status": "skipped",
            "reason": "archived_name_collision",
        }
    return subject, {
        **observation,
        "status": "created" if created else "reused",
        "reason": None,
        "subject_id": str(subject.id),
        "resolution": "canonical",
    }


def _discovery_observation(
    *,
    status: str,
    reason: str | None,
    proposal: SubjectDiscoveryProposal | None = None,
    corroborating_families: tuple[SignalFamily, ...] = (),
) -> dict[str, object]:
    observation: dict[str, object] = {
        "status": status,
        "reason": reason,
    }
    if proposal is None:
        return observation
    normalized_name = SubjectName.from_value(proposal.name).normalized
    observation.update(
        {
            "kind": proposal.kind.value,
            "confidence": proposal.confidence,
            "proposal_name_hash": hashlib.sha256(
                normalized_name.encode("utf-8"),
            ).hexdigest()[:24],
            "corroborating_signal_families": [
                family.value for family in corroborating_families
            ],
        },
    )
    return observation


def _candidate(subject: SubjectRecord) -> SubjectClassificationCandidate:
    return SubjectClassificationCandidate(
        subject_id=subject.id,
        kind=subject.kind,
        canonical_name=subject.name,
        aliases=tuple(alias.name for alias in subject.aliases),
    )


def _document_summary(metadata: dict[str, Any], max_chars: int) -> str | None:
    for container_name in ("hierarchical_retrieval", "contextualization"):
        container = metadata.get(container_name)
        if isinstance(container, dict):
            value = container.get("document_summary")
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())[:max_chars]
    return None


def _explicit_metadata_values(metadata: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("subject", "subjects", "project", "topic", "organization", "tags"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value[:500])
        elif isinstance(value, list):
            values.extend(
                item[:500]
                for item in value[:20]
                if isinstance(item, str) and item.strip()
            )
    return tuple(values[:32])


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
