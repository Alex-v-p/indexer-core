from pathlib import Path
from contextlib import asynccontextmanager
from types import SimpleNamespace
import uuid

import pytest

from packages.indexer_application.services.evaluation_subject_scope import (
    ApplicationSubjectScopeEvaluationProvider,
    load_subject_scope_fixture_manifest,
)
from packages.indexer_application.dto import DocumentStatus
from packages.rag_core.evaluation import SubjectScopeFixtureRequirement
from packages.rag_core.subjects import SubjectKind
from scripts.seed_subject_scoping_fixture import list_all_api, plan_fixture_changes


def test_fixture_manifest_parses_and_idempotency_plan_converges() -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = load_subject_scope_fixture_manifest(
        root / "datasets/eval_sets/subject_scoping_fixture_v1.json",
    )
    assert manifest["revision"] == "subject-scoping-fixture/1.0"

    empty = plan_fixture_changes(manifest, [], [])
    assert len(empty["subjects"]) == len(manifest["subjects"])
    assert len(empty["documents"]) == len(manifest["documents"])

    subjects = [{"name": item["name"], "kind": item["kind"]} for item in manifest["subjects"]]
    documents = [
        {"original_filename": item["filename"], "status": "ready", "chunk_count": 1}
        for item in manifest["documents"]
    ]
    assert plan_fixture_changes(manifest, subjects, documents) == {"subjects": [], "documents": []}

    documents[0]["status"] = "failed"
    documents[1]["chunk_count"] = 0
    pending = plan_fixture_changes(manifest, subjects, documents)
    assert manifest["documents"][0]["key"] in pending["documents"]
    assert manifest["documents"][1]["key"] in pending["documents"]


def test_api_discovery_paginates_past_first_hundred_for_idempotency() -> None:
    subjects = [
        {"name": f"Noise {index}", "kind": "custom"} for index in range(150)
    ] + [{"name": "Target", "kind": "project"}]
    documents = [
        {"original_filename": f"noise-{index}.md", "status": "ready", "chunk_count": 1}
        for index in range(150)
    ] + [{"original_filename": "target.md", "status": "ready", "chunk_count": 1}]

    class Client:
        def get_json(self, path):
            source = subjects if path.startswith("/subjects") else documents
            offset = int(path.split("offset=")[1])
            return source[offset : offset + 100]

    discovered_subjects = list_all_api(Client(), "/subjects")
    discovered_documents = list_all_api(Client(), "/documents")
    manifest = {
        "subjects": [{"key": "target", "name": "Target", "kind": "project"}],
        "documents": [{"key": "target", "filename": "target.md"}],
    }

    assert len(discovered_subjects) == 151
    assert len(discovered_documents) == 151
    assert plan_fixture_changes(manifest, discovered_subjects, discovered_documents) == {
        "subjects": [],
        "documents": [],
    }


def test_fixture_manifest_rejects_unknown_membership_key(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(
        '{"schema_version":"1.0","revision":"r1",'
        '"subjects":[{"key":"known","kind":"project","name":"Known","aliases":[]}],'
        '"documents":[{"key":"doc","filename":"doc.md","path":"doc.md","subject_keys":["missing"]}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown subject keys"):
        load_subject_scope_fixture_manifest(path)


async def test_provider_preflight_rejects_revision_and_missing_membership(tmp_path: Path) -> None:
    manifest = tmp_path / "fixture.json"
    manifest.write_text(
        '{"schema_version":"1.0","revision":"r1",'
        '"subjects":[{"key":"project","kind":"project","name":"Project","aliases":[]}],'
        '"documents":[{"key":"doc","filename":"doc.md","path":"doc.md","subject_keys":["project"]}]}',
        encoding="utf-8",
    )
    subject = SimpleNamespace(
        id=uuid.uuid4(), name="Project", kind=SubjectKind.PROJECT, aliases=[],
    )
    document = SimpleNamespace(
        id=uuid.uuid4(),
        original_filename="doc.md",
        status=DocumentStatus.READY,
        chunk_indexes=(object(),),
    )

    class Subjects:
        async def list(self, **kwargs):
            return [subject] if kwargs["offset"] == 0 else []

        async def list_assigned_document_ids(self, **kwargs):
            return ()

    class Documents:
        async def list(self, **kwargs):
            return [document] if kwargs["offset"] == 0 else []

    @asynccontextmanager
    async def uow_factory():
        yield SimpleNamespace(subjects=Subjects(), documents=Documents())

    provider = ApplicationSubjectScopeEvaluationProvider(
        uow_factory=uow_factory,
        repository_root=tmp_path,
    )
    with pytest.raises(ValueError, match="revision mismatch"):
        await provider.preflight(SubjectScopeFixtureRequirement("fixture.json", "r2"))
    document.chunk_indexes = ()
    with pytest.raises(ValueError, match="not ready and indexed"):
        await provider.preflight(SubjectScopeFixtureRequirement("fixture.json", "r1"))
    document.chunk_indexes = (object(),)
    with pytest.raises(ValueError, match="not assigned"):
        await provider.preflight(SubjectScopeFixtureRequirement("fixture.json", "r1"))
