from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from packages.indexer_application.dto import DocumentRecord, DocumentStatus, SubjectRecord
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services.query_subject_scope import (
    QuerySubjectScopeConfig,
    resolve_query_subject_scope_snapshot,
)
from packages.rag_core.evaluation.runner import (
    SubjectScopeEvaluationRequest,
    SubjectScopeFixtureRequirement,
)
from packages.rag_core.document_scope import ResolvedSubjectScopeSnapshot
from packages.rag_core.subjects import normalize_subject_name


class ApplicationSubjectScopeEvaluationProvider:
    """Resolve evaluation scope through the production application boundary."""

    def __init__(
        self,
        *,
        uow_factory: Callable[[], AbstractAsyncContextManager[UnitOfWork]],
        repository_root: Path,
        config: QuerySubjectScopeConfig = QuerySubjectScopeConfig(),
    ) -> None:
        self._uow_factory = uow_factory
        self._repository_root = repository_root.resolve()
        self._config = config

    async def preflight(self, requirement: SubjectScopeFixtureRequirement) -> None:
        manifest = load_subject_scope_fixture_manifest(
            self._resolve_manifest_path(requirement.manifest_path),
        )
        if manifest["revision"] != requirement.revision:
            raise ValueError(
                "Subject-scoping fixture revision mismatch: "
                f"dataset requires {requirement.revision!r}, manifest declares {manifest['revision']!r}.",
            )
        async with self._uow_factory() as uow:
            await _validate_seeded_fixture(uow, manifest)

    async def resolve(
        self,
        request: SubjectScopeEvaluationRequest,
    ) -> ResolvedSubjectScopeSnapshot:
        async with self._uow_factory() as uow:
            requested_ids = list(request.requested_subject_ids)
            for name in request.requested_subject_names:
                matches = await uow.subjects.resolve_name(name)
                if len(matches) != 1:
                    raise ValueError(
                        f"Evaluation subject name {name!r} resolved to {len(matches)} active catalog entries; expected one.",
                    )
                requested_ids.append(matches[0].subject.id)
            return await resolve_query_subject_scope_snapshot(
                uow=uow,
                question=request.question,
                requested_subject_ids=tuple(dict.fromkeys(requested_ids)),
                coverage_mode=request.coverage_mode,
                config=self._config,
            )

    def _resolve_manifest_path(self, value: str) -> Path:
        path = (self._repository_root / value).resolve()
        if not path.is_relative_to(self._repository_root):
            raise ValueError("Subject-scoping fixture manifest must stay inside the repository.")
        return path


def load_subject_scope_fixture_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            f"Subject-scoping fixture manifest is missing: {path}. Run the documented seed command first.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Subject-scoping fixture manifest is invalid JSON: {path}.") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ValueError("Subject-scoping fixture manifest must use schema_version '1.0'.")
    if not isinstance(raw.get("revision"), str) or not raw["revision"].strip():
        raise ValueError("Subject-scoping fixture manifest revision must be non-empty.")
    subjects = _fixture_list(raw, "subjects")
    documents = _fixture_list(raw, "documents")
    subject_keys = {_fixture_string(item, "key") for item in subjects}
    if len(subject_keys) != len(subjects):
        raise ValueError("Subject-scoping fixture subject keys must be unique.")
    document_keys = {_fixture_string(item, "key") for item in documents}
    if len(document_keys) != len(documents):
        raise ValueError("Subject-scoping fixture document keys must be unique.")
    for item in subjects:
        _fixture_string(item, "name")
        if item.get("kind") not in {"project", "topic", "organization", "custom"}:
            raise ValueError(f"Fixture subject {_fixture_string(item, 'key')!r} has an invalid kind.")
        _fixture_strings(item.get("aliases", []), "aliases")
    for item in documents:
        _fixture_string(item, "filename")
        _fixture_string(item, "path")
        referenced = set(_fixture_strings(item.get("subject_keys", []), "subject_keys"))
        missing = referenced - subject_keys
        if missing:
            raise ValueError(f"Fixture document references unknown subject keys: {sorted(missing)}.")
    return raw


async def _validate_seeded_fixture(uow: UnitOfWork, manifest: dict[str, Any]) -> None:
    subjects = await _all_subjects(uow)
    by_name_and_kind = {
        (normalize_subject_name(item.name), item.kind.value): item
        for item in subjects
    }
    subject_by_key: dict[str, SubjectRecord] = {}
    problems: list[str] = []
    for expected in manifest["subjects"]:
        name = _fixture_string(expected, "name")
        actual = by_name_and_kind.get((normalize_subject_name(name), expected["kind"]))
        if actual is None:
            problems.append(f"missing subject {name!r}")
            continue
        actual_aliases = {
            normalize_subject_name(item.name)
            for item in actual.aliases
            if item.archived_at is None
        }
        expected_aliases = {normalize_subject_name(item) for item in expected.get("aliases", [])}
        if not expected_aliases.issubset(actual_aliases):
            problems.append(f"subject {name!r} is missing aliases {sorted(expected_aliases - actual_aliases)}")
        subject_by_key[_fixture_string(expected, "key")] = actual

    documents = await _all_documents(uow)
    by_filename: dict[str, list[DocumentRecord]] = {}
    for item in documents:
        if item.original_filename:
            by_filename.setdefault(item.original_filename, []).append(item)
    document_by_key: dict[str, DocumentRecord] = {}
    for expected in manifest["documents"]:
        filename = _fixture_string(expected, "filename")
        candidates = by_filename.get(filename, [])
        actual = next((item for item in candidates if _is_ready_and_indexed(item)), None)
        if not candidates:
            problems.append(f"missing indexed document {filename!r}")
            continue
        if actual is None:
            states = [
                f"status={item.status.value}, chunks={len(item.chunk_indexes)}"
                for item in candidates
            ]
            problems.append(
                f"document {filename!r} is not ready and indexed ({', '.join(states)})",
            )
            continue
        document_by_key[_fixture_string(expected, "key")] = actual

    for expected in manifest["subjects"]:
        subject_key = _fixture_string(expected, "key")
        subject = subject_by_key.get(subject_key)
        if subject is None:
            continue
        assigned = set(await uow.subjects.list_assigned_document_ids(subject_ids=(subject.id,)))
        for document_spec in manifest["documents"]:
            if subject_key not in document_spec.get("subject_keys", []):
                continue
            document = document_by_key.get(_fixture_string(document_spec, "key"))
            if document is not None and document.id not in assigned:
                problems.append(
                    f"document {_fixture_string(document_spec, 'filename')!r} is not assigned to subject {subject.name!r}",
                )
    if problems:
        raise ValueError(
            "Subject-scoping fixture preflight failed: " + "; ".join(problems)
            + ". Run `python -m scripts.seed_subject_scoping_fixture`.",
        )


async def _all_subjects(uow: UnitOfWork) -> list[SubjectRecord]:
    result: list[SubjectRecord] = []
    offset = 0
    while True:
        page = await uow.subjects.list(limit=100, offset=offset)
        result.extend(page)
        if len(page) < 100:
            return result
        offset += len(page)


async def _all_documents(uow: UnitOfWork) -> list[DocumentRecord]:
    result: list[DocumentRecord] = []
    offset = 0
    while True:
        page = await uow.documents.list(limit=100, offset=offset)
        result.extend(page)
        if len(page) < 100:
            return result
        offset += len(page)


def _fixture_list(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Fixture field {key!r} must be a non-empty array of objects.")
    return value


def _fixture_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Fixture field {key!r} must be a non-empty string.")
    return value.strip()


def _fixture_strings(value: object, key: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Fixture field {key!r} must be an array of non-empty strings.")
    return [item.strip() for item in value]


def _is_ready_and_indexed(document: DocumentRecord) -> bool:
    return document.status is DocumentStatus.READY and bool(document.chunk_indexes)
