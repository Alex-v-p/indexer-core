from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, replace
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
    SubjectContentCandidate,
    SubjectContentMatchProvider,
    SubjectCreateResolution,
    SubjectModelEvidenceError,
    SubjectModelResolutionProvider,
    SubjectReuseResolution,
    corroborating_subject_name_families,
    classify_document_subjects,
    grounded_subject_reuse_families,
    normalize_subject_name,
)


_GENERIC_CREATE_NAME_TOKENS = {
    "project", "projects", "plan", "system", "dashboard", "topic",
    "organization", "program",
}
_TRIVIAL_NAME_CONNECTOR_TOKENS = {
    "a", "an", "and", "for", "in", "of", "on", "the", "to", "with",
}
_GENERIC_CONTENT_TOKENS = _GENERIC_CREATE_NAME_TOKENS | _TRIVIAL_NAME_CONNECTOR_TOKENS | {
    "care", "delivery", "implementation", "integration", "iot", "llm",
    "medical", "method", "methods", "monitoring", "platform", "process",
    "technology", "technologies",
}


@dataclass(frozen=True, slots=True)
class SubjectClassificationJobConfig:
    policy: SubjectClassificationPolicy
    classifier_version: str = CLASSIFIER_VERSION
    max_summary_chars: int = 2_000
    discovery_enabled: bool = True
    max_representative_documents_per_subject: int = 3
    max_representative_chars_per_subject: int = 3_000
    max_representative_assigned_ids_scanned_per_subject: int = 12


class ClassifyDocumentSubjectsJobHandler:
    def __init__(
        self,
        *,
        uow: UnitOfWork,
        config: SubjectClassificationJobConfig,
        model_resolution: SubjectModelResolutionProvider | None = None,
        content_match: SubjectContentMatchProvider | None = None,
    ) -> None:
        self._uow = uow
        self._config = config
        self._model_resolution = model_resolution
        self._content_match = content_match

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
        initial_input = _classification_input(document, summary, {})
        initial_outcomes = classify_document_subjects(
            initial_input,
            candidates,
            policy=self._config.policy,
        )
        resolution: SubjectReuseResolution | SubjectCreateResolution | None = None
        grounded_reuse_families: tuple[SignalFamily, ...] = ()
        content_confirmed_subject_id: uuid.UUID | None = None
        content_confirmed_summary: str | None = None
        content_reuse_disabled = False
        discovery = _discovery_observation(
            status="skipped",
            reason=(
                "disabled"
                if not self._config.discovery_enabled
                else "provider_unavailable"
                if self._model_resolution is None
                else "summary_unavailable"
                if not summary
                else "no_resolution"
            ),
        )
        if self._model_resolution is not None and summary:
            await report(0.30, "resolving_document_subject")
            resolution = await self._model_resolution.resolve(summary, candidates)
            if isinstance(resolution, SubjectReuseResolution):
                selected = next(
                    (candidate for candidate in candidates if candidate.subject_id == resolution.subject_id),
                    None,
                )
                if selected is None:
                    raise SubjectModelEvidenceError("Model selected an unknown subject.")
                grounded_reuse_families = grounded_subject_reuse_families(
                    initial_input, selected
                )

        if (
            summary
            and self._content_match is not None
            and (not isinstance(resolution, SubjectReuseResolution) or not grounded_reuse_families)
        ):
            try:
                representatives = await _subject_content_candidates(
                    self._uow,
                    subjects=subjects,
                    current_document_id=document_id,
                    config=self._config,
                )
                shortlist = _content_shortlist(summary, representatives, limit=3)
                confirmed: list[tuple[SubjectContentCandidate, float, float, float]] = []
                for representative in shortlist:
                    selection = await self._content_match.match_content(
                        summary, (representative,)
                    )
                    if (
                        selection is None
                        or selection.subject_id != representative.subject_id
                        or selection.confidence < self._config.policy.medium_threshold
                    ):
                        continue
                    confirmation = await self._content_match.match_content(
                        summary, (representative,), pairwise_confirmation=True
                    )
                    if (
                        confirmation is None
                        or confirmation.subject_id != representative.subject_id
                        or confirmation.confidence < self._config.policy.medium_threshold
                    ):
                        continue
                    confirmed.append(
                        (
                            representative,
                            min(selection.confidence, confirmation.confidence),
                            selection.confidence,
                            confirmation.confidence,
                        )
                    )
                confirmed.sort(key=lambda item: (-item[1], str(item[0].subject_id)))
                if confirmed:
                    winner = confirmed[0]
                    runner = confirmed[1][1] if len(confirmed) > 1 else 0.0
                    if winner[1] - runner >= self._config.policy.medium_margin:
                        resolution = SubjectReuseResolution(winner[0].subject_id, winner[1])
                        content_confirmed_subject_id = winner[0].subject_id
                        content_confirmed_summary = normalize_subject_name(summary)
                        grounded_reuse_families = ()
                        discovery = _content_reuse_observation(
                            winner[0],
                            confidence=winner[1],
                            selection_confidence=winner[2],
                            confirmation_confidence=winner[3],
                        )
            except SubjectModelEvidenceError:
                pass

        if (
            isinstance(resolution, SubjectReuseResolution)
            and not grounded_reuse_families
            and content_confirmed_subject_id is None
            and self._model_resolution is not None
            and summary
        ):
            challenged_subject_id = resolution.subject_id
            fallback = await self._model_resolution.resolve(summary, ())
            if isinstance(fallback, SubjectReuseResolution):
                raise SubjectModelEvidenceError("Empty-catalogue fallback cannot reuse a subject.")
            resolution = fallback
            discovery = _resolution_observation(
                fallback,
                request=initial_input,
                policy=self._config.policy,
                challenged_subject_id=challenged_subject_id,
            )
        elif isinstance(resolution, SubjectCreateResolution):
            discovery = _resolution_observation(
                resolution, request=initial_input, policy=self._config.policy
            )

        model_scores: dict[uuid.UUID, float] = {}
        if isinstance(resolution, SubjectReuseResolution):
            model_scores[resolution.subject_id] = resolution.confidence

        deterministic_assigned = {
            outcome.subject_id: next(
                candidate.kind for candidate in candidates if candidate.subject_id == outcome.subject_id
            )
            for outcome in initial_outcomes
            if outcome.state is DecisionState.ASSIGNED
        }
        if isinstance(resolution, SubjectReuseResolution):
            selected = next(candidate for candidate in candidates if candidate.subject_id == resolution.subject_id)
            if any(
                subject_id != resolution.subject_id and kind is selected.kind
                for subject_id, kind in deterministic_assigned.items()
            ):
                model_scores = {}
                if content_confirmed_subject_id is not None:
                    content_confirmed_subject_id = None
                    content_confirmed_summary = None
                    content_reuse_disabled = True

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
        locked_summary = _document_summary(
            locked_latest_ready.metadata, self._config.max_summary_chars
        )
        locked_input_without_model = _classification_input(
            locked_document, locked_summary, {}
        )
        if (
            content_confirmed_summary is not None
            and content_confirmed_summary != normalize_subject_name(locked_summary or "")
        ):
            raise RuntimeError(
                "Subject content confirmation summary changed before locked classification."
            )
        if (
            isinstance(resolution, SubjectReuseResolution)
            and grounded_reuse_families
            and content_confirmed_subject_id is None
        ):
            selected = next(
                candidate for candidate in candidates if candidate.subject_id == resolution.subject_id
            )
            if not grounded_subject_reuse_families(locked_input_without_model, selected):
                raise RuntimeError(
                    "Subject reuse grounding changed before locked classification."
                )
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
        locked_deterministic_outcomes = classify_document_subjects(
            locked_input_without_model,
            candidates,
            policy=self._config.policy,
        )
        if isinstance(resolution, SubjectReuseResolution):
            selected = next(
                candidate for candidate in candidates if candidate.subject_id == resolution.subject_id
            )
            locked_conflict = any(
                outcome.state is DecisionState.ASSIGNED
                and outcome.subject_id != resolution.subject_id
                and next(
                    candidate.kind
                    for candidate in candidates
                    if candidate.subject_id == outcome.subject_id
                )
                is selected.kind
                for outcome in locked_deterministic_outcomes
            )
            if locked_conflict:
                model_scores = {}
                content_confirmed_subject_id = None
                content_reuse_disabled = True
                locked_input = locked_input_without_model
                locked_existing_outcomes = locked_deterministic_outcomes
        discovered_subject: SubjectRecord | None = None
        create_resolution = (
            resolution if isinstance(resolution, SubjectCreateResolution) else None
        )
        if create_resolution is not None and discovery.get("status") == "eligible":
            locked_assigned = [
                outcome
                for outcome in locked_existing_outcomes
                if outcome.state is DecisionState.ASSIGNED
            ]
            if locked_assigned:
                discovery = _discovery_observation(
                    status="skipped",
                    reason="existing_subject_evidence_after_lock",
                    proposal=create_resolution,
                    corroborating_families=corroborating_subject_name_families(locked_input, create_resolution.name),
                )
            else:
                locked_evaluation = _evaluate_discovery_proposal(
                    create_resolution,
                    request=locked_input,
                    policy=self._config.policy,
                )
                if locked_evaluation.get("status") != "eligible":
                    discovery = locked_evaluation
                else:
                    discovered_subject, discovery = await _resolve_discovered_subject(
                        self._uow,
                        proposal=create_resolution,
                        observation=locked_evaluation,
                        config=self._config,
        )
        if discovered_subject is not None:
            assert create_resolution is not None
            if all(subject.id != discovered_subject.id for subject in subjects):
                subjects.append(discovered_subject)
                candidates = (*candidates, _candidate(discovered_subject))
            model_scores = {
                **model_scores,
                discovered_subject.id: create_resolution.confidence,
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
        if discovered_subject is not None and create_resolution is not None:
            outcomes = tuple(
                replace(
                    outcome,
                    state=DecisionState.ASSIGNED,
                )
                if outcome.subject_id == discovered_subject.id
                and create_resolution.confidence >= self._config.policy.medium_threshold
                else outcome
                for outcome in outcomes
            )
        content_reuse_applied = (
            content_confirmed_subject_id is not None and not content_reuse_disabled
        )
        if content_reuse_applied:
            outcomes = tuple(
                replace(outcome, state=DecisionState.ASSIGNED)
                if outcome.subject_id == content_confirmed_subject_id
                and outcome.confidence >= self._config.policy.medium_threshold
                else outcome
                for outcome in outcomes
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
            if outcome is None and not _is_automatic_decision(existing):
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
                    if (
                        discovered_subject is not None
                        and subject.id == discovered_subject.id
                    )
                    or content_reuse_applied
                    and subject.id == content_confirmed_subject_id
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
    demoted_assignment = (
        _is_automatic_assignment(existing)
        and outcome is not None
        and outcome.state is DecisionState.SUGGESTED
    )
    if demoted_assignment:
        signal_payload["review_recommended"] = True
    if discovery is not None:
        allowed = {
            "action", "kind", "confidence", "proposal_name_hash",
            "corroborating_signal_families", "subject_id", "resolution",
            "resolved_kind", "challenged_subject_id", "fallback_action",
            "summary_name_grounded", "summary_significant_token_count",
            "summary_matched_token_count", "selection_confidence",
            "confirmation_confidence", "representative_content_hash",
            "representative_document_count",
        }
        signal_payload["discovery"] = {
            "source": "structured_document_summary",
            **{key: value for key, value in discovery.items() if key in allowed},
        }
    state = outcome.state if outcome is not None else DecisionState.REJECTED
    rationale = (
        "Latest automatic evidence weakened; the assignment now requires review."
        if demoted_assignment
        else "Automatic subject evidence satisfied the assignment policy."
        if state is DecisionState.ASSIGNED
        else "Automatic subject evidence requires human review."
        if state is DecisionState.SUGGESTED
        else "Latest automatic evidence no longer supports this subject."
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


def _is_automatic_decision(
    decision: DocumentSubjectDecisionRecord | None,
) -> bool:
    return (
        decision is not None
        and decision.control_source is DecisionControlSource.AUTOMATIC
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
    proposal: SubjectCreateResolution,
    *,
    request: SubjectClassificationInput,
    policy: SubjectClassificationPolicy,
) -> dict[str, object]:
    corroborating_families = corroborating_subject_name_families(
        request,
        proposal.name,
    )
    summary_grounding = _summary_name_grounding(
        request.document_summary or "", proposal.name
    )
    eligible = proposal.confidence >= policy.high_threshold or (
        proposal.confidence >= policy.medium_threshold
        and (bool(corroborating_families) or summary_grounding[0])
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
        summary_grounding=summary_grounding,
    )


async def _resolve_discovered_subject(
    uow: UnitOfWork,
    *,
    proposal: SubjectCreateResolution,
    observation: dict[str, object],
    config: SubjectClassificationJobConfig,
) -> tuple[SubjectRecord | None, dict[str, object]]:
    matches = await uow.subjects.resolve_name(
        proposal.name,
        kind=None,
        include_archived=True,
    )
    active_by_id = {
        match.subject.id: match
        for match in matches
        if match.subject.archived_at is None
    }
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
            **(
                {"resolved_kind": match.subject.kind.value}
                if match.subject.kind is not proposal.kind
                else {}
            ),
        }

    if matches:
        return None, {
            **observation,
            "status": "skipped",
            "reason": "archived_name_collision",
        }

    all_subjects = await _subjects_including_archived(uow)
    equivalents = [
        subject
        for subject in all_subjects
        if any(
            _names_are_equivalent(proposal.name, value)
            for value in (subject.name, *(alias.name for alias in subject.aliases if alias.archived_at is None))
        )
    ]
    active_equivalents = {subject.id: subject for subject in equivalents if subject.archived_at is None}
    if len(active_equivalents) > 1:
        return None, {**observation, "status": "skipped", "reason": "ambiguous_name_resolution"}
    if active_equivalents:
        subject = next(iter(active_equivalents.values()))
        return subject, {
            **observation,
            "status": "reused",
            "reason": None,
            "subject_id": str(subject.id),
            "resolution": "normalized_token_equivalence",
            **({"resolved_kind": subject.kind.value} if subject.kind is not proposal.kind else {}),
        }
    if equivalents:
        return None, {**observation, "status": "skipped", "reason": "archived_name_collision"}

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
    proposal: SubjectCreateResolution | None = None,
    corroborating_families: tuple[SignalFamily, ...] = (),
    summary_grounding: tuple[bool, int, int] = (False, 0, 0),
) -> dict[str, object]:
    observation: dict[str, object] = {
        "status": status,
        "reason": reason,
    }
    if proposal is None:
        return observation
    normalized_name = normalize_subject_name(proposal.name)
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
            "action": "create",
            "summary_name_grounded": summary_grounding[0],
            "summary_significant_token_count": summary_grounding[1],
            "summary_matched_token_count": summary_grounding[2],
        },
    )
    return observation


def _resolution_observation(
    resolution: SubjectCreateResolution | None,
    *,
    request: SubjectClassificationInput,
    policy: SubjectClassificationPolicy,
    challenged_subject_id: uuid.UUID | None = None,
) -> dict[str, object]:
    if resolution is None:
        observation = _discovery_observation(status="skipped", reason="no_proposal")
        if challenged_subject_id is not None:
            observation.update(
                challenged_subject_id=str(challenged_subject_id), fallback_action="none"
            )
        return observation
    observation = _evaluate_discovery_proposal(
        resolution, request=request, policy=policy
    )
    if challenged_subject_id is not None:
        observation.update(
            challenged_subject_id=str(challenged_subject_id), fallback_action="create"
        )
    return observation


def _content_reuse_observation(
    candidate: SubjectContentCandidate,
    *,
    confidence: float,
    selection_confidence: float,
    confirmation_confidence: float,
) -> dict[str, object]:
    return {
        "status": "reused",
        "reason": None,
        "action": "reuse",
        "kind": candidate.kind.value,
        "confidence": confidence,
        "subject_id": str(candidate.subject_id),
        "resolution": "confirmed_representative_content",
        "selection_confidence": selection_confidence,
        "confirmation_confidence": confirmation_confidence,
        "representative_content_hash": candidate.representative_content_hash,
        "representative_document_count": candidate.document_count,
    }


async def _subject_content_candidates(
    uow: UnitOfWork,
    *,
    subjects: list[SubjectRecord],
    current_document_id: uuid.UUID,
    config: SubjectClassificationJobConfig,
) -> tuple[SubjectContentCandidate, ...]:
    candidates: list[SubjectContentCandidate] = []
    for subject in subjects:
        assigned_ids = await uow.subjects.list_assigned_document_ids(
            subject_ids=(subject.id,)
        )
        summaries: list[str] = []
        for assigned_id in assigned_ids[
            : config.max_representative_assigned_ids_scanned_per_subject
        ]:
            if assigned_id == current_document_id:
                continue
            document = await uow.documents.get(assigned_id)
            if document is None:
                continue
            latest = max(
                (version for version in document.versions if version.status is DocumentVersionStatus.READY),
                key=lambda version: version.version_number,
                default=None,
            )
            if latest is None:
                continue
            summary = _document_summary(latest.metadata, config.max_summary_chars)
            if summary:
                summaries.append(summary)
            if len(summaries) >= config.max_representative_documents_per_subject:
                break
        if not summaries:
            continue
        content = "\n".join(summaries)[: config.max_representative_chars_per_subject]
        candidates.append(
            SubjectContentCandidate(
                subject_id=subject.id,
                kind=subject.kind,
                representative_content=content,
                representative_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest()[:24],
                document_count=len(summaries),
            )
        )
    return tuple(candidates)


def _content_shortlist(
    summary: str,
    candidates: tuple[SubjectContentCandidate, ...],
    *,
    limit: int,
) -> tuple[SubjectContentCandidate, ...]:
    summary_tokens = _significant_tokens(summary, ignored=_GENERIC_CONTENT_TOKENS)
    ranked: list[tuple[int, float, str, SubjectContentCandidate]] = []
    for candidate in candidates:
        candidate_tokens = _significant_tokens(
            candidate.representative_content, ignored=_GENERIC_CONTENT_TOKENS
        )
        shared = summary_tokens & candidate_tokens
        if len(shared) < 2:
            continue
        union = summary_tokens | candidate_tokens
        ranked.append(
            (len(shared), len(shared) / len(union), str(candidate.subject_id), candidate)
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(item[3] for item in ranked[:limit])


def _summary_name_grounding(summary: str, name: str) -> tuple[bool, int, int]:
    proposal = _significant_tokens(
        name, ignored=_GENERIC_CREATE_NAME_TOKENS | _TRIVIAL_NAME_CONNECTOR_TOKENS
    )
    summary_tokens = set(normalize_subject_name(summary).split())
    matched = proposal & summary_tokens
    grounded = len(proposal) >= 2 and len(matched) / len(proposal) >= 0.75
    return grounded, len(proposal), len(matched)


def _names_are_equivalent(left: str, right: str) -> bool:
    ignored = _GENERIC_CREATE_NAME_TOKENS | _TRIVIAL_NAME_CONNECTOR_TOKENS
    left_tokens = _significant_tokens(left, ignored=ignored)
    right_tokens = _significant_tokens(right, ignored=ignored)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    shared = left_tokens & right_tokens
    return (
        len(shared) >= 2
        and len(shared) / max(len(left_tokens), len(right_tokens)) >= 0.8
        and len(shared) / len(left_tokens | right_tokens) >= 0.6
    )


def _significant_tokens(value: str, *, ignored: set[str]) -> set[str]:
    return {
        token for token in normalize_subject_name(value).split()
        if token not in ignored and len(token) > 1
    }


async def _subjects_including_archived(uow: UnitOfWork) -> list[SubjectRecord]:
    subjects: list[SubjectRecord] = []
    offset = 0
    while True:
        batch = await uow.subjects.list(
            limit=100, offset=offset, include_archived=True
        )
        subjects.extend(batch)
        if len(batch) < 100:
            return subjects
        offset += len(batch)


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
    for key in ("subject", "subjects", "project", "topic", "organization"):
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
