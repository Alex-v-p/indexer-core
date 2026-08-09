from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from packages.indexer_application.dto import DocumentVersionStatus
from packages.indexer_application.services.background_jobs import (
    ClassifyDocumentOrganizationJobHandler,
    DocumentOrganizationJobConfig,
)
from packages.indexer_application.services.background_jobs.document_organization import (
    _grouping_context,
    _representatives,
)
from packages.rag_core.document_organization import (
    ClassificationSource,
    ContentGroupAssignmentState,
    DocumentOrganizationPolicy,
    DocumentTypeScore,
    GroupCreateResolution,
    ORGANIZATION_POLICY_VERSION,
)


def _document(*, summary=None, sections=(), title="Document", filename="document.md"):
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    metadata = {}
    if summary is not None:
        metadata = {"hierarchical_retrieval": {
            "document_summary": summary,
            "sections": [{"summary": value} for value in sections],
        }}
    version = SimpleNamespace(
        id=version_id,
        version_number=1,
        status=DocumentVersionStatus.READY,
        metadata=metadata,
    )
    return SimpleNamespace(
        id=document_id,
        title=title,
        original_filename=filename,
        versions=(version,),
    )


class Documents:
    def __init__(self, documents):
        self.items = {item.id: item for item in documents}
        self.statuses = []
        self.get_calls = []
    async def get(self, document_id):
        self.get_calls.append(document_id)
        return self.items.get(document_id)
    async def get_for_update(self, document_id):
        return self.items.get(document_id)
    async def set_organization_classification_status(self, **kwargs):
        self.statuses.append(kwargs)
        return True


class Types:
    async def list(self, **kwargs):
        del kwargs
        return []
    async def list_document_decisions(self, **kwargs):
        del kwargs
        return []


class Groups:
    def __init__(self, current=None):
        self.current = current
        self.writes = []
        self.assignments = []
    async def get_assignment(self, document_id):
        del document_id
        return self.current
    async def write_assignment(self, assignment):
        self.writes.append(assignment)
    async def list_assignments(self, **kwargs):
        del kwargs
        return self.assignments


class Uow:
    def __init__(self, document, current=None):
        self.documents = Documents([document])
        self.document_types = Types()
        self.content_groups = Groups(current)


class UnusedEmbeddingProvider:
    vector_size = 4

    async def embed_texts(self, texts):
        raise AssertionError(f"Embeddings were not expected for {texts!r}")


async def _report(progress, stage):
    del progress, stage


@pytest.mark.asyncio
async def test_missing_root_summary_becomes_unresolved_without_fabricated_group() -> None:
    document = _document()
    uow = Uow(document)
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=None,
        group_provider=None,
        embedding_provider=UnusedEmbeddingProvider(),
    )
    result = await handler(
        {
            "document_id": str(document.id),
            "document_version_id": str(document.versions[0].id),
            "policy_version": ORGANIZATION_POLICY_VERSION,
        },
        _report,
        job_id=uuid.uuid4(),
    )

    assignment = uow.content_groups.writes[0]
    assert assignment.state is ContentGroupAssignmentState.UNRESOLVED
    assert assignment.content_group_id is None
    assert assignment.unresolved_reason == "no_summary"
    assert assignment.summary_hash is None
    assert "summary" not in str(assignment.signals).lower()
    assert result["content_group_state"] == "unresolved"


@pytest.mark.asyncio
async def test_manual_group_assignment_is_never_overwritten() -> None:
    document = _document()
    current = SimpleNamespace(source=ClassificationSource.MANUAL)
    uow = Uow(document, current)
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=None,
        group_provider=None,
        embedding_provider=UnusedEmbeddingProvider(),
    )
    await handler(
        {"document_id": str(document.id), "document_version_id": str(document.versions[0].id), "policy_version": ORGANIZATION_POLICY_VERSION},
        _report,
        job_id=uuid.uuid4(),
    )
    assert uow.content_groups.writes == []


@pytest.mark.asyncio
async def test_representatives_exclude_current_document() -> None:
    current = _document(summary="Mobilab plan")
    other = _document(summary="Mobilab care platform delivery")
    uow = Uow(current)
    uow.documents.items[other.id] = other
    uow.content_groups.assignments = [
        SimpleNamespace(document_id=current.id),
        SimpleNamespace(document_id=other.id),
    ]
    group = SimpleNamespace(id=uuid.uuid4())
    candidates = await _representatives(
        uow,
        groups=[group],
        current_document_id=current.id,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
    )

    assert current.id not in uow.documents.get_calls
    assert candidates[0].document_count == 1
    assert "Mobilab care platform" in candidates[0].representative_content


def test_grouping_context_requires_root_and_includes_only_first_three_sections_within_bound() -> None:
    document = _document(
        summary=" Mobilab   root summary ",
        sections=("section one", "section two", "section three", "section four"),
    )
    config = DocumentOrganizationJobConfig(
        policy=DocumentOrganizationPolicy(),
        max_summary_chars=65,
    )

    context = _grouping_context(document.versions[0], config)

    assert context is not None
    assert context.startswith("Mobilab root summary\n\nsection one")
    assert "section three" in context
    assert "section four" not in context
    assert len(context) <= 65
    missing_root = _document()
    missing_root.versions[0].metadata = {
        "hierarchical_retrieval": {"sections": [{"summary": "orphan section"}]}
    }
    assert _grouping_context(missing_root.versions[0], config) is None


@pytest.mark.parametrize(
    "override",
    (
        {"max_representative_documents_per_group": 0},
        {"max_representative_assignments_scanned_per_group": 0},
        {"max_representative_chars_per_group": 0},
    ),
)
def test_representative_bounds_must_be_positive(override) -> None:
    with pytest.raises(ValueError, match="representative bounds"):
        DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy(), **override)


class LiveTypes:
    def __init__(self) -> None:
        self.other = SimpleNamespace(
            id=uuid.uuid4(), key="other", label="Other", description="Fallback"
        )
        self.writes = []

    async def list(self, **kwargs):
        del kwargs
        return [self.other]

    async def list_document_decisions(self, *, document_id):
        return [item for item in self.writes if item.document_id == document_id]

    async def write_decision(self, decision):
        self.writes.append(decision)


class LiveGroups:
    def __init__(self) -> None:
        self.groups = []
        self.writes = []
        self.seeded_assignments = []
        self.publish_lock_count = 0
        self.create_count = 0
        self.on_publish_lock = None

    async def list(self, **kwargs):
        del kwargs
        return list(self.groups)

    async def list_assignments(self, *, content_group_id, **kwargs):
        del kwargs
        return [
            item
            for item in (*self.seeded_assignments, *self.writes)
            if item.content_group_id == content_group_id
            and item.state is ContentGroupAssignmentState.ASSIGNED
        ]

    async def get_assignment(self, document_id):
        return next(
            (item for item in reversed(self.writes) if item.document_id == document_id),
            None,
        )

    async def write_assignment(self, assignment):
        self.writes.append(assignment)

    async def acquire_publish_lock(self):
        self.publish_lock_count += 1
        if self.on_publish_lock is not None:
            self.on_publish_lock()

    async def create_or_get_canonical(self, *, name, metadata):
        self.create_count += 1
        normalized = " ".join(name.casefold().split())
        existing = next(
            (item for item in self.groups if item.normalized_name == normalized),
            None,
        )
        if existing is not None:
            return existing, False
        group = SimpleNamespace(
            id=uuid.uuid4(),
            name=name,
            normalized_name=normalized,
            aliases=[],
            metadata=metadata,
        )
        self.groups.append(group)
        return group, True


class AdditiveCatalogueChurnGroups(LiveGroups):
    def __init__(self) -> None:
        super().__init__()
        self.list_calls = 0

    async def list(self, **kwargs):
        del kwargs
        self.list_calls += 1
        if self.list_calls in {2, 4}:
            index = len(self.groups) + 1
            self.groups.append(SimpleNamespace(
                id=uuid.uuid4(),
                name=f"Unrelated Initiative {index}",
                normalized_name=f"unrelated initiative {index}",
                aliases=[],
            ))
        return list(self.groups)


class LiveUow:
    def __init__(self, documents):
        self.documents = Documents(documents)
        self.document_types = LiveTypes()
        self.content_groups = LiveGroups()


class SanitizedEmbeddingProvider:
    vector_size = 4

    async def embed_texts(self, texts):
        return [self._vector(value.casefold()) for value in texts]

    @staticmethod
    def _vector(value):
        if "mobilab" in value and "realization" in value:
            return [0.8, 0.0, 0.0, 0.6]
        if "mobilab" in value:
            return [1.0, 0.0, 0.0, 0.0]
        if "torque" in value:
            return [0.0, 1.0, 0.0, 0.0]
        if "sensor" in value:
            return [0.0, 0.6, 0.8, 0.0]
        return [0.0, 0.6, 0.0, 0.8]


class SanitizedModelProvider:
    def __init__(self) -> None:
        self.resolve_calls = 0
        self.match_calls = 0

    async def classify_types(self, **kwargs):
        del kwargs
        return (DocumentTypeScore("other", 0.90),)

    async def match_group_content(self, summary, candidates, *, pairwise_confirmation=False):
        del summary, candidates, pairwise_confirmation
        self.match_calls += 1
        return None

    async def resolve_group(self, summary, candidates):
        del candidates
        self.resolve_calls += 1
        lowered = summary.casefold()
        if "mobilab" in lowered:
            name = "Mobilab Care Initiative"
        elif "torque" in lowered:
            name = "Torque Down Monitoring Dashboard"
        elif "sensor" in lowered:
            name = "Industrial Sensor Monitoring"
        else:
            name = "Independent Quality Research"
        return GroupCreateResolution(name, 0.90)


class PlanningLabelModelProvider(SanitizedModelProvider):
    async def resolve_group(self, summary, candidates):
        del summary, candidates
        self.resolve_calls += 1
        return GroupCreateResolution(
            "Mobilab Care Project Planning-Report.pdf",
            0.90,
        )


class FixedLabelModelProvider(SanitizedModelProvider):
    def __init__(self, label) -> None:
        super().__init__()
        self.label = label

    async def resolve_group(self, summary, candidates):
        del summary, candidates
        self.resolve_calls += 1
        return GroupCreateResolution(self.label, 0.90)


async def _classify(handler, document):
    return await handler(
        {
            "document_id": str(document.id),
            "document_version_id": str(document.versions[0].id),
            "policy_version": ORGANIZATION_POLICY_VERSION,
        },
        _report,
        job_id=uuid.uuid4(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", (False, True))
async def test_mobilab_plan_and_realization_reuse_one_group_in_both_orders(reverse) -> None:
    plan = _document(
        summary="Mobilab patient care platform delivery plan",
        sections=("Care workflows and mobile delivery",),
        title="MC Project Plan",
        filename="MC_ProjectPlan.docx",
    )
    realization = _document(
        summary="Mobilab patient care platform realization outcomes",
        sections=("Care workflows and mobile delivery results",),
        title="Mobilab Realization",
        filename="Mobilab_realization.pdf",
    )
    documents = [realization, plan] if reverse else [plan, realization]
    uow = LiveUow(documents)
    provider = SanitizedModelProvider()
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=provider,
        group_provider=provider,
        embedding_provider=SanitizedEmbeddingProvider(),
    )

    await _classify(handler, documents[0])
    await _classify(handler, documents[1])

    assert len(uow.content_groups.groups) == 1
    assert uow.content_groups.create_count == 1
    assert provider.resolve_calls == 1
    second = uow.content_groups.writes[-1]
    assert second.content_group_id == uow.content_groups.groups[0].id
    assert second.signals["resolution"] == "semantic_representative_content"
    assert second.signals["semantic_score"] == pytest.approx(0.8)
    assert all(not isinstance(value, list) for value in second.signals.values())
    assert "mobilab" not in str(second.signals).casefold()


@pytest.mark.asyncio
async def test_daf_iot_and_unrelated_documents_remain_separate_below_semantic_gate() -> None:
    documents = [
        _document(summary="Industrial monitoring torque down dashboard", sections=("industrial monitoring controls",)),
        _document(summary="Industrial monitoring sensor gateway", sections=("industrial monitoring telemetry",)),
        _document(summary="Industrial monitoring independent quality study", sections=("industrial monitoring research",)),
    ]
    uow = LiveUow(documents)
    provider = SanitizedModelProvider()
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=provider,
        group_provider=provider,
        embedding_provider=SanitizedEmbeddingProvider(),
    )

    for document in documents:
        await _classify(handler, document)

    assert len(uow.content_groups.groups) == 3
    assert len({item.content_group_id for item in uow.content_groups.writes}) == 3
    assert provider.resolve_calls == 3
    assert provider.match_calls > 0


@pytest.mark.asyncio
async def test_create_publish_lock_rechecks_new_representative_before_creating_duplicate() -> None:
    current = _document(
        summary="Mobilab patient care platform delivery plan",
        sections=("Care workflows and mobile delivery",),
    )
    representative = _document(
        summary="Mobilab patient care platform realization outcomes",
        sections=("Care workflows and mobile delivery results",),
    )
    uow = LiveUow([current, representative])
    provider = SanitizedModelProvider()
    concurrent_group = SimpleNamespace(
        id=uuid.uuid4(),
        name="Mobilab Care Initiative",
        normalized_name="mobilab care initiative",
        aliases=[],
    )

    def publish_concurrent_group():
        uow.content_groups.groups.append(concurrent_group)
        uow.content_groups.seeded_assignments.append(SimpleNamespace(
            document_id=representative.id,
            content_group_id=concurrent_group.id,
            state=ContentGroupAssignmentState.ASSIGNED,
        ))

    uow.content_groups.on_publish_lock = publish_concurrent_group
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=provider,
        group_provider=provider,
        embedding_provider=SanitizedEmbeddingProvider(),
    )

    await _classify(handler, current)

    assignment = uow.content_groups.writes[-1]
    assert uow.content_groups.publish_lock_count == 1
    assert uow.content_groups.create_count == 0
    assert assignment.content_group_id == concurrent_group.id
    assert assignment.signals["resolution"] == "semantic_publish_recheck"


@pytest.mark.asyncio
async def test_created_group_label_removes_document_role_inflections() -> None:
    document = _document(summary="Mobilab care internship project delivery")
    uow = LiveUow([document])
    provider = PlanningLabelModelProvider()
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=provider,
        group_provider=provider,
        embedding_provider=SanitizedEmbeddingProvider(),
    )

    await _classify(handler, document)

    assert uow.content_groups.groups[0].name == "Mobilab Care Project"
    assert uow.content_groups.writes[0].content_group_id == uow.content_groups.groups[0].id


@pytest.mark.asyncio
async def test_additive_catalogue_churn_does_not_make_coherent_create_unresolved() -> None:
    document = _document(
        summary="Industrial monitoring torque down dashboard",
        sections=("Torque monitoring controls",),
    )
    uow = LiveUow([document])
    uow.content_groups = AdditiveCatalogueChurnGroups()
    provider = SanitizedModelProvider()
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=provider,
        group_provider=provider,
        embedding_provider=SanitizedEmbeddingProvider(),
    )

    result = await _classify(handler, document)

    assignment = uow.content_groups.writes[-1]
    assert provider.resolve_calls == 2
    assert uow.content_groups.publish_lock_count == 1
    assert uow.content_groups.create_count == 1
    assert assignment.state is ContentGroupAssignmentState.ASSIGNED
    assert assignment.unresolved_reason is None
    assert result["content_group_state"] == "assigned"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("proposed", "expected"),
    (
        ("MD Anderson Cancer Care Report.pdf", "MD Anderson Cancer Care"),
        ("DOC Health Initiative Report.pdf", "DOC Health Initiative"),
    ),
)
async def test_created_group_label_preserves_subject_acronyms(proposed, expected) -> None:
    document = _document(summary="Healthcare initiative delivery context")
    uow = LiveUow([document])
    provider = FixedLabelModelProvider(proposed)
    handler = ClassifyDocumentOrganizationJobHandler(
        uow=uow,
        config=DocumentOrganizationJobConfig(policy=DocumentOrganizationPolicy()),
        type_provider=provider,
        group_provider=provider,
        embedding_provider=SanitizedEmbeddingProvider(),
    )

    await _classify(handler, document)

    assert uow.content_groups.groups[0].name == expected
