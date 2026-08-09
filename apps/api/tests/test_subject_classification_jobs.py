from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from packages.indexer_application.commands import (
    EnqueueDocumentSubjectClassificationCommand,
    EnqueueDocumentSubjectClassificationHandler,
)
from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobStatus,
    BackgroundJobType,
    DocumentRecord,
    DocumentStatus,
    DocumentSubjectDecisionRecord,
    DocumentVersionRecord,
    DocumentVersionStatus,
    SubjectAliasRecord,
    SubjectNameMatchRecord,
    SubjectRecord,
)
from packages.indexer_application.services.background_jobs import (
    ClassifyDocumentSubjectsJobHandler,
    SubjectClassificationJobConfig,
    update_subject_classification_job_status,
)
from packages.rag_core.subjects import (
    ConfidenceBand,
    DecisionControlSource,
    DecisionState,
    SubjectClassificationPolicy,
    SubjectCreateResolution,
    SubjectKind,
    SubjectNameMatchType,
    SubjectReuseResolution,
    normalize_subject_name,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)


class FakeDocuments:
    def __init__(self, documents: list[DocumentRecord]) -> None:
        self.documents = {document.id: document for document in documents}
        self.statuses: list[tuple[uuid.UUID, dict[str, object]]] = []
        self.classification_statuses: dict[uuid.UUID, dict[str, object]] = {}
        self.lock_calls: list[uuid.UUID] = []

    async def get(self, document_id: uuid.UUID):
        return self.documents.get(document_id)

    async def get_for_update(self, document_id: uuid.UUID):
        self.lock_calls.append(document_id)
        return self.documents.get(document_id)

    async def list(self, *, limit: int, offset: int):
        return list(self.documents.values())[offset : offset + limit]

    async def set_subject_classification_status(
        self,
        *,
        document_id,
        status,
        expected_job_id=None,
        expected_document_version_id=None,
        expected_policy_version=None,
        expected_statuses=None,
    ):
        current = self.classification_statuses.get(document_id)
        if any(
            value is not None
            for value in (
                expected_job_id,
                expected_document_version_id,
                expected_policy_version,
                expected_statuses,
            )
        ):
            if current is None:
                return False
            if expected_job_id is not None and current.get("job_id") != str(expected_job_id):
                return False
            if (
                expected_document_version_id is not None
                and current.get("document_version_id") != str(expected_document_version_id)
            ):
                return False
            if (
                expected_policy_version is not None
                and current.get("policy_version") != expected_policy_version
            ):
                return False
            if expected_statuses is not None and current.get("status") not in expected_statuses:
                return False
        self.classification_statuses[document_id] = dict(status)
        self.statuses.append((document_id, dict(status)))
        return True

    def queue_classification(
        self,
        *,
        document_id: uuid.UUID,
        job_id: uuid.UUID,
        version_id: uuid.UUID,
        policy_version: str,
    ) -> None:
        self.classification_statuses[document_id] = {
            "status": "queued",
            "job_id": str(job_id),
            "document_version_id": str(version_id),
            "policy_version": policy_version,
        }


class FakeSubjects:
    def __init__(
        self,
        subjects: list[SubjectRecord],
        decisions: list[DocumentSubjectDecisionRecord] | None = None,
    ) -> None:
        self.subjects = subjects
        self.decisions = {
            (decision.document_id, decision.subject_id): decision
            for decision in (decisions or [])
        }
        self.writes: list[tuple[object, int | None]] = []
        self.create_or_get_calls: list[tuple[SubjectKind, str]] = []

    async def list(self, *, limit: int, offset: int, **kwargs):
        include_archived = kwargs.get("include_archived", False)
        active = [
            subject
            for subject in self.subjects
            if include_archived or subject.archived_at is None
        ]
        return active[offset : offset + limit]

    async def resolve_name(self, name, *, kind=None, include_archived=False):
        normalized = normalize_subject_name(name)
        matches = []
        for subject in self.subjects:
            if kind is not None and subject.kind is not kind:
                continue
            if not include_archived and subject.archived_at is not None:
                continue
            if subject.normalized_name == normalized:
                matches.append(
                    SubjectNameMatchRecord(subject, SubjectNameMatchType.CANONICAL),
                )
            elif any(
                alias.normalized_name == normalized and alias.archived_at is None
                for alias in subject.aliases
            ):
                matches.append(
                    SubjectNameMatchRecord(subject, SubjectNameMatchType.ALIAS),
                )
        return matches

    async def create_or_get_canonical(self, *, kind, name, **kwargs):
        self.create_or_get_calls.append((kind, name))
        normalized = normalize_subject_name(name)
        existing = next(
            (
                subject
                for subject in self.subjects
                if subject.kind is kind and subject.normalized_name == normalized
            ),
            None,
        )
        if existing is not None:
            return existing, False
        created = SubjectRecord(
            id=uuid.uuid4(),
            kind=kind,
            name=name,
            normalized_name=normalized,
            description=None,
            metadata=dict(kwargs.get("metadata") or {}),
            created_at=NOW,
            updated_at=NOW,
        )
        self.subjects.append(created)
        return created, True

    async def list_document_decisions(self, *, document_id, states=None):
        return [
            decision
            for (candidate_document_id, _), decision in self.decisions.items()
            if candidate_document_id == document_id
        ]

    async def get_decision(self, *, document_id, subject_id):
        return self.decisions.get((document_id, subject_id))

    async def write_decision(self, decision, *, expected_revision=None):
        current = self.decisions.get((decision.document_id, decision.subject_id))
        record = DocumentSubjectDecisionRecord(
            id=current.id if current is not None else uuid.uuid4(),
            document_id=decision.document_id,
            subject_id=decision.subject_id,
            state=decision.state,
            control_source=decision.control_source,
            confidence=decision.confidence,
            confidence_band=decision.confidence_band,
            rationale=decision.rationale,
            classifier_version=decision.classifier_version,
            policy_version=decision.policy_version,
            signals=dict(decision.signals),
            classified_document_version_id=decision.classified_document_version_id,
            revision=(current.revision + 1 if current is not None else 1),
            created_at=current.created_at if current is not None else NOW,
            updated_at=NOW,
        )
        self.decisions[(decision.document_id, decision.subject_id)] = record
        self.writes.append((decision, expected_revision))
        return record


class FakeJobs:
    def __init__(self) -> None:
        self.jobs: dict[str, BackgroundJobRecord] = {}
        self.submissions = []

    async def enqueue(self, submission):
        if submission.dedupe_key in self.jobs:
            return self.jobs[submission.dedupe_key]
        self.submissions.append(submission)
        job = _job(submission)
        self.jobs[submission.dedupe_key] = job
        return job


class FakeUnitOfWork:
    def __init__(self, documents, subjects) -> None:
        self.documents = documents
        self.subjects = subjects
        self.background_jobs = FakeJobs()
        self.commit_calls = 0

    async def commit(self):
        self.commit_calls += 1


async def _report(progress: float, stage: str) -> None:
    assert 0 <= progress <= 1
    assert stage


async def test_enqueue_is_deduplicated_by_document_version_and_policy() -> None:
    document = _document("Apollo")
    uow = FakeUnitOfWork(FakeDocuments([document]), FakeSubjects([]))
    handler = EnqueueDocumentSubjectClassificationHandler(
        uow=uow, enabled=True, policy_version="policy/1", max_attempts=4,
    )

    first = await handler(
        EnqueueDocumentSubjectClassificationCommand(document_id=document.id),
    )
    second = await handler(
        EnqueueDocumentSubjectClassificationCommand(document_id=document.id),
    )

    assert first.jobs[0].id == second.jobs[0].id
    assert len(uow.background_jobs.submissions) == 1
    submission = uow.background_jobs.submissions[0]
    assert submission.job_type is BackgroundJobType.CLASSIFY_DOCUMENT_SUBJECTS
    assert str(document.id) in submission.dedupe_key
    assert str(document.versions[0].id) in submission.dedupe_key
    assert submission.dedupe_key.endswith(":policy/1")


async def test_backfill_skips_documents_without_a_ready_version() -> None:
    ready = _document("Ready")
    pending = _document("Pending", ready=False)
    uow = FakeUnitOfWork(FakeDocuments([ready, pending]), FakeSubjects([]))
    result = await EnqueueDocumentSubjectClassificationHandler(
        uow=uow, enabled=True, policy_version="policy/1",
    )(EnqueueDocumentSubjectClassificationCommand())

    assert [job.payload["document_id"] for job in result.jobs] == [str(ready.id)]
    assert result.skipped_document_ids == (pending.id,)


async def test_classification_is_idempotent_and_never_overwrites_manual_authority() -> None:
    document = _document("Apollo")
    automatic = _subject("Apollo", SubjectKind.PROJECT)
    manual = _subject("Apollo", SubjectKind.TOPIC)
    manual_decision = _decision(
        document.id,
        manual.id,
        state=DecisionState.REJECTED,
        source=DecisionControlSource.MANUAL,
    )
    subjects = FakeSubjects([automatic, manual], [manual_decision])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    uow = FakeUnitOfWork(documents, subjects)
    handler = ClassifyDocumentSubjectsJobHandler(
        uow=uow,
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
    )
    payload = _payload(document, "policy/1")

    first = await handler(payload, _report, job_id=job_id)
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    second = await handler(payload, _report, job_id=job_id)

    assert len(subjects.writes) == 1
    proposed, expected_revision = subjects.writes[0]
    assert proposed.subject_id == automatic.id
    assert proposed.state is DecisionState.ASSIGNED
    assert expected_revision == 0
    assert subjects.decisions[(document.id, manual.id)] == manual_decision
    assert first["assigned_count"] == second["assigned_count"] == 1
    assert first["suggested_count"] == second["suggested_count"] == 0
    assert all(
        "Apollo" not in str(value)
        for value in proposed.signals.values()
    )


async def test_automatic_demotion_rejects_stale_assignment() -> None:
    document = _document("Unrelated")
    subject = _subject("Apollo", SubjectKind.PROJECT)
    existing = _decision(
        document.id,
        subject.id,
        state=DecisionState.ASSIGNED,
        source=DecisionControlSource.AUTOMATIC,
    )
    subjects = FakeSubjects([subject], [existing])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    uow = FakeUnitOfWork(documents, subjects)

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=uow,
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
    )(_payload(document, "policy/1"), _report, job_id=job_id)

    proposed, expected_revision = subjects.writes[0]
    assert proposed.state is DecisionState.REJECTED
    assert "review_recommended" not in proposed.signals
    assert expected_revision == existing.revision
    assert result["assigned_count"] == 0
    assert result["suggested_count"] == 0
    assert result["review_required_count"] == 0


async def test_model_failure_is_retryable_and_does_not_mutate_ready_document() -> None:
    class FailingModel:
        async def resolve(self, summary, candidates):
            raise RuntimeError("model unavailable")

    document = _document("Unrelated", summary="Apollo deployment summary")
    subjects = FakeSubjects([_subject("Apollo", SubjectKind.PROJECT)])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    uow = FakeUnitOfWork(documents, subjects)

    with pytest.raises(RuntimeError, match="model unavailable"):
        await ClassifyDocumentSubjectsJobHandler(
            uow=uow,
            config=SubjectClassificationJobConfig(
                policy=SubjectClassificationPolicy(policy_version="policy/1"),
            ),
            model_resolution=FailingModel(),
        )(_payload(document, "policy/1"), _report, job_id=job_id)

    assert document.status is DocumentStatus.READY
    assert subjects.writes == []
    assert documents.statuses == []


async def test_manual_rejection_is_excluded_from_effective_counts() -> None:
    document = _document("Apollo")
    subject = _subject("Apollo", SubjectKind.PROJECT)
    rejected = _decision(
        document.id,
        subject.id,
        state=DecisionState.REJECTED,
        source=DecisionControlSource.MANUAL,
    )
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    subjects = FakeSubjects([subject], [rejected])

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
    )(_payload(document, "policy/1"), _report, job_id=job_id)

    assert result["assigned_count"] == 0
    assert result["suggested_count"] == 0
    assert result["review_required_count"] == 0
    assert subjects.writes == []


async def test_deferred_v1_model_cannot_overwrite_activated_v2() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class DeferredModel:
        async def resolve(self, summary, candidates):
            started.set()
            await release.wait()
            return SubjectReuseResolution(candidates[0].subject_id, 0.99)

    document_v1 = _document("Apollo", summary="Apollo v1 summary")
    documents = FakeDocuments([document_v1])
    old_job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document_v1.id,
        job_id=old_job_id,
        version_id=document_v1.versions[0].id,
        policy_version="policy/1",
    )
    subjects = FakeSubjects([_subject("Apollo", SubjectKind.PROJECT)])
    handler = ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=DeferredModel(),
    )

    running = asyncio.create_task(
        handler(_payload(document_v1, "policy/1"), _report, job_id=old_job_id),
    )
    await started.wait()
    version_v2 = replace(
        document_v1.versions[0],
        id=uuid.uuid4(),
        version_number=2,
        metadata={"hierarchical_retrieval": {"document_summary": "Apollo v2 summary"}},
    )
    documents.documents[document_v1.id] = replace(
        document_v1,
        versions=(*document_v1.versions, version_v2),
    )
    new_job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document_v1.id,
        job_id=new_job_id,
        version_id=version_v2.id,
        policy_version="policy/1",
    )
    release.set()

    result = await running

    assert result["status"] == "superseded"
    assert result["decision_write_count"] == 0
    assert subjects.writes == []
    assert documents.classification_statuses[document_v1.id]["job_id"] == str(new_job_id)
    assert documents.lock_calls == [document_v1.id]


async def test_stale_old_job_success_cannot_replace_newer_status_or_decisions() -> None:
    document = _document("Apollo")
    documents = FakeDocuments([document])
    old_job_id = uuid.uuid4()
    new_job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=new_job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    subjects = FakeSubjects([_subject("Apollo", SubjectKind.PROJECT)])

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
    )(_payload(document, "policy/1"), _report, job_id=old_job_id)

    assert result["status"] == "stale"
    assert result["decision_write_count"] == 0
    assert subjects.writes == []
    assert documents.classification_statuses[document.id]["job_id"] == str(new_job_id)


async def test_stale_old_job_failure_cannot_replace_newer_status() -> None:
    document = _document("Apollo")
    documents = FakeDocuments([document])
    old_job_id = uuid.uuid4()
    new_job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=new_job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )

    updated = await update_subject_classification_job_status(
        uow=FakeUnitOfWork(documents, FakeSubjects([])),
        document_id=document.id,
        job_id=old_job_id,
        document_version_id=document.versions[0].id,
        policy_version="policy/1",
        status="failed",
        error_message="old attempt failed",
    )

    assert updated is False
    assert documents.statuses == []
    assert documents.classification_statuses[document.id]["job_id"] == str(new_job_id)


class FakeDiscovery:
    def __init__(self, proposal: SubjectCreateResolution | None) -> None:
        self.proposal = proposal
        self.calls: list[str] = []

    async def resolve(self, summary: str, candidates):
        del candidates
        self.calls.append(summary)
        return self.proposal


async def test_high_confidence_discovery_creates_and_assigns_from_empty_catalog() -> None:
    document = _document("Release Notes", summary="Project Orion delivery status")
    subjects = FakeSubjects([])
    discovery = FakeDiscovery(
        SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.92),
    )
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=discovery,
    )(_payload(document, "policy/1"), _report, job_id=job_id)

    assert result["discovery"]["status"] == "created"
    assert result["assigned_count"] == 1
    assert subjects.create_or_get_calls == [(SubjectKind.PROJECT, "Orion")]
    proposed, expected_revision = subjects.writes[0]
    assert expected_revision == 0
    assert proposed.signals["discovery"]["source"] == "structured_document_summary"
    assert "Orion" not in str(proposed.signals)


async def test_medium_discovery_requires_independent_name_corroboration() -> None:
    corroborated = _document(
        "Orion delivery notes",
        summary="The delivery program has reached phase two.",
    )
    uncorroborated = _document(
        "Release Notes",
        summary="Project Orion has reached phase two.",
    )

    for document, expected_status in (
        (corroborated, "created"),
        (uncorroborated, "skipped"),
    ):
        subjects = FakeSubjects([])
        documents = FakeDocuments([document])
        job_id = uuid.uuid4()
        documents.queue_classification(
            document_id=document.id,
            job_id=job_id,
            version_id=document.versions[0].id,
            policy_version="policy/1",
        )
        result = await ClassifyDocumentSubjectsJobHandler(
            uow=FakeUnitOfWork(documents, subjects),
            config=SubjectClassificationJobConfig(
                policy=SubjectClassificationPolicy(policy_version="policy/1"),
            ),
            model_resolution=FakeDiscovery(
                SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.70),
            ),
        )(_payload(document, "policy/1"), _report, job_id=job_id)

        assert result["discovery"]["status"] == expected_status
        if expected_status == "created":
            assert result["assigned_count"] == 1
        else:
            assert result["discovery"]["reason"] == (
                "medium_confidence_without_name_corroboration"
            )
            assert subjects.create_or_get_calls == []
            assert subjects.writes == []


async def test_low_confidence_discovery_does_not_mutate() -> None:
    document = _document("Orion", summary="Orion might be mentioned")
    subjects = FakeSubjects([])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=FakeDiscovery(
            SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.40),
        ),
    )(_payload(document, "policy/1"), _report, job_id=job_id)

    assert result["discovery"]["reason"] == "below_medium_confidence"
    assert subjects.create_or_get_calls == []
    assert subjects.writes == []


async def test_existing_subject_assignment_blocks_creation_after_resolution() -> None:
    document = _document("Apollo", summary="Project Orion delivery status")
    discovery = FakeDiscovery(
        SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.99),
    )
    subjects = FakeSubjects([_subject("Apollo", SubjectKind.PROJECT)])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=discovery,
    )(_payload(document, "policy/1"), _report, job_id=job_id)

    assert discovery.calls == ["Project Orion delivery status"]
    assert result["discovery"]["status"] == "skipped"
    assert result["discovery"]["reason"] == "existing_subject_evidence_after_lock"
    assert subjects.create_or_get_calls == []
    assert result["assigned_count"] == 1


async def test_discovery_is_idempotent_and_manual_rejection_remains_authoritative() -> None:
    document = _document("Release Notes", summary="Project Orion delivery status")
    subjects = FakeSubjects([])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    discovery = FakeDiscovery(
        SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.92),
    )
    handler = ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=discovery,
    )
    for _ in range(2):
        documents.queue_classification(
            document_id=document.id,
            job_id=job_id,
            version_id=document.versions[0].id,
            policy_version="policy/1",
        )
        await handler(_payload(document, "policy/1"), _report, job_id=job_id)

    assert len(subjects.subjects) == 1
    assert len(subjects.writes) == 1

    subject = subjects.subjects[0]
    manual = _decision(
        document.id,
        subject.id,
        state=DecisionState.REJECTED,
        source=DecisionControlSource.MANUAL,
    )
    subjects.decisions[(document.id, subject.id)] = manual
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    result = await handler(_payload(document, "policy/1"), _report, job_id=job_id)

    assert result["assigned_count"] == 0
    assert subjects.decisions[(document.id, subject.id)] is manual
    assert len(subjects.writes) == 1


async def test_discovery_skips_archived_and_ambiguous_name_collisions() -> None:
    document = _document("Release Notes", summary="Project Orion delivery status")
    archived = replace(_subject("Orion", SubjectKind.PROJECT), archived_at=NOW)
    alpha = replace(
        _subject("Alpha", SubjectKind.PROJECT),
        aliases=(_alias("Orion"),),
    )
    beta = replace(
        _subject("Beta", SubjectKind.PROJECT),
        aliases=(_alias("Orion"),),
    )
    for catalogue, expected_reason in (
        ([archived], "archived_name_collision"),
        ([alpha, beta], "ambiguous_name_resolution"),
    ):
        subjects = FakeSubjects(catalogue)
        documents = FakeDocuments([document])
        job_id = uuid.uuid4()
        documents.queue_classification(
            document_id=document.id,
            job_id=job_id,
            version_id=document.versions[0].id,
            policy_version="policy/1",
        )
        result = await ClassifyDocumentSubjectsJobHandler(
            uow=FakeUnitOfWork(documents, subjects),
            config=SubjectClassificationJobConfig(
                policy=SubjectClassificationPolicy(policy_version="policy/1"),
            ),
            model_resolution=FakeDiscovery(
                SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.92),
            ),
        )(_payload(document, "policy/1"), _report, job_id=job_id)

        assert result["discovery"]["reason"] == expected_reason
        assert subjects.create_or_get_calls == []
        assert subjects.writes == []


async def test_discovery_safely_reuses_one_active_alias() -> None:
    document = _document("Release Notes", summary="Project Orion delivery status")
    existing = replace(
        _subject("Alpha Program", SubjectKind.PROJECT),
        aliases=(_alias("Orion"),),
    )
    subjects = FakeSubjects([existing])
    documents = FakeDocuments([document])
    job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=FakeDiscovery(
            SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.92),
        ),
    )(_payload(document, "policy/1"), _report, job_id=job_id)

    assert result["discovery"]["status"] == "reused"
    assert result["discovery"]["resolution"] == "alias"
    assert subjects.create_or_get_calls == []
    assert subjects.writes[0][0].subject_id == existing.id


async def test_stale_discovery_job_never_creates_a_subject() -> None:
    document = _document("Release Notes", summary="Project Orion delivery status")
    documents = FakeDocuments([document])
    current_job_id = uuid.uuid4()
    documents.queue_classification(
        document_id=document.id,
        job_id=current_job_id,
        version_id=document.versions[0].id,
        policy_version="policy/1",
    )
    subjects = FakeSubjects([])

    result = await ClassifyDocumentSubjectsJobHandler(
        uow=FakeUnitOfWork(documents, subjects),
        config=SubjectClassificationJobConfig(
            policy=SubjectClassificationPolicy(policy_version="policy/1"),
        ),
        model_resolution=FakeDiscovery(
            SubjectCreateResolution(SubjectKind.PROJECT, "Orion", 0.92),
        ),
    )(_payload(document, "policy/1"), _report, job_id=uuid.uuid4())

    assert result["status"] == "stale"
    assert subjects.create_or_get_calls == []
    assert subjects.writes == []


def _document(title: str, *, ready: bool = True, summary: str | None = None):
    document_id = uuid.uuid4()
    version = DocumentVersionRecord(
        id=uuid.uuid4(), version_number=1, storage_uri="file://document.md",
        content_type="text/markdown", checksum_sha256="abc", parser_name="markdown",
        parser_version="1", status=(DocumentVersionStatus.READY if ready else DocumentVersionStatus.PROCESSING),
        metadata=(
            {"hierarchical_retrieval": {"document_summary": summary}}
            if summary is not None else {}
        ),
        created_at=NOW, updated_at=NOW,
    )
    return DocumentRecord(
        id=document_id, title=title, original_filename=f"{title}.md",
        content_type="text/markdown", storage_uri=version.storage_uri, size_bytes=1,
        checksum_sha256="abc", status=(DocumentStatus.READY if ready else DocumentStatus.PROCESSING),
        metadata={}, created_at=NOW, updated_at=NOW, versions=(version,),
    )


def _subject(name: str, kind: SubjectKind) -> SubjectRecord:
    return SubjectRecord(
        id=uuid.uuid4(), kind=kind, name=name, normalized_name=name.casefold(),
        description=None, metadata={}, created_at=NOW, updated_at=NOW,
    )


def _alias(name: str) -> SubjectAliasRecord:
    return SubjectAliasRecord(
        id=uuid.uuid4(),
        subject_id=uuid.uuid4(),
        name=name,
        normalized_name=normalize_subject_name(name),
        created_at=NOW,
    )


def _decision(document_id, subject_id, *, state, source):
    automatic = source is DecisionControlSource.AUTOMATIC
    return DocumentSubjectDecisionRecord(
        id=uuid.uuid4(), document_id=document_id, subject_id=subject_id,
        state=state, control_source=source, confidence=0.9 if automatic else None,
        confidence_band=ConfidenceBand.HIGH if automatic else None,
        rationale="Existing decision.", classifier_version="old/1" if automatic else None,
        policy_version="old/1" if automatic else None, signals={} if not automatic else {"old": True},
        classified_document_version_id=uuid.uuid4() if automatic else None,
        revision=3, created_at=NOW, updated_at=NOW,
    )


def _payload(document: DocumentRecord, policy_version: str):
    return {
        "document_id": str(document.id),
        "document_version_id": str(document.versions[0].id),
        "document_version_number": 1,
        "policy_version": policy_version,
    }


def _job(submission) -> BackgroundJobRecord:
    return BackgroundJobRecord(
        id=uuid.uuid4(), job_type=submission.job_type,
        status=BackgroundJobStatus.QUEUED, priority=submission.priority,
        payload=dict(submission.payload), result={}, progress=0.0,
        current_stage="queued", attempts=0, max_attempts=submission.max_attempts,
        dedupe_key=submission.dedupe_key, scheduled_at=NOW, locked_at=None,
        locked_by=None, heartbeat_at=None, started_at=None, completed_at=None,
        error_message=None, created_at=NOW, updated_at=NOW,
    )
