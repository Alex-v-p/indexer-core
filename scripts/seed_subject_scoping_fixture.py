from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from packages.indexer_application.services.evaluation_subject_scope import (  # noqa: E402
    load_subject_scope_fixture_manifest,
)
from packages.rag_core.subjects import normalize_subject_name  # noqa: E402

DEFAULT_MANIFEST = REPOSITORY_ROOT / "datasets" / "eval_sets" / "subject_scoping_fixture_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Idempotently seed the local subject-scoping evaluation fixture.")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/api/v1",
        help="Local Indexer API base URL including its configured prefix.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-only", action="store_true", help="Validate without creating anything.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def plan_fixture_changes(
    manifest: dict[str, Any],
    subjects: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Return stable missing-object keys; repeated calls converge to empty lists."""

    subject_identities = {
        (normalize_subject_name(str(item.get("name", ""))), str(item.get("kind", "")))
        for item in subjects
    }
    document_names = {
        str(item.get("original_filename", ""))
        for item in documents
        if _api_document_is_ready_and_indexed(item)
    }
    return {
        "subjects": [
            item["key"]
            for item in manifest["subjects"]
            if (normalize_subject_name(item["name"]), item["kind"]) not in subject_identities
        ],
        "documents": [item["key"] for item in manifest["documents"] if item["filename"] not in document_names],
    }


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else REPOSITORY_ROOT / args.manifest
    try:
        manifest = load_subject_scope_fixture_manifest(manifest_path.resolve())
        client = LocalApiClient(args.api_url, timeout_seconds=args.timeout_seconds)
        subjects = list_all_api(client, "/subjects")
        documents = list_all_api(client, "/documents")
        plan = plan_fixture_changes(manifest, subjects, documents)
        if args.preflight_only:
            problems = _remote_preflight(client, manifest, subjects, documents)
            if problems:
                raise ValueError("; ".join(problems))
            print(f"Fixture {manifest['revision']} is ready.")
            return 0

        subject_by_key = _ensure_subjects(client, manifest, subjects)
        _ensure_documents(client, manifest, documents, subject_by_key, args.timeout_seconds)
        subjects = list_all_api(client, "/subjects")
        documents = list_all_api(client, "/documents")
        _ensure_memberships(client, manifest, subjects, documents)
        problems = _remote_preflight(client, manifest, subjects, documents)
        if problems:
            raise ValueError("Post-seed preflight failed: " + "; ".join(problems))
        print(
            f"Fixture {manifest['revision']} is ready "
            f"(created {len(plan['subjects'])} subject(s), {len(plan['documents'])} document(s)).",
        )
        return 0
    except (ValueError, OSError, urllib.error.URLError) as exc:
        print(f"Subject-scoping fixture error: {exc}", file=sys.stderr)
        return 2


class LocalApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_json(self, path: str) -> Any:
        return self._json_request("GET", path)

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        return self._json_request("POST", path, body)

    def put_json(self, path: str, body: dict[str, Any]) -> Any:
        return self._json_request("PUT", path, body)

    def upload(self, path: str, fields: list[tuple[str, str]], file_path: Path) -> tuple[Any, str | None]:
        boundary = f"codex-fixture-{uuid.uuid4().hex}"
        body = bytearray()
        for name, value in fields:
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        content_type = mimetypes.guess_type(file_path.name)[0] or "text/markdown"
        body.extend(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode(),
        )
        body.extend(file_path.read_bytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            self.base_url + path,
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read()), response.headers.get("Location")

    def _json_request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self._url(path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise ValueError(f"{method} {path} returned HTTP {exc.code}: {detail}") from exc

    def _url(self, path: str) -> str:
        parsed = urllib.parse.urlsplit(path)
        if parsed.scheme and parsed.netloc:
            return path
        base = urllib.parse.urlsplit(self.base_url)
        if path.startswith(base.path.rstrip("/") + "/"):
            return urllib.parse.urlunsplit((base.scheme, base.netloc, path, "", ""))
        return self.base_url + path


def list_all_api(
    client: LocalApiClient,
    resource: str,
    *,
    page_size: int = 100,
    max_items: int = 10_000,
) -> list[dict[str, Any]]:
    if page_size <= 0 or max_items <= 0:
        raise ValueError("API pagination bounds must be positive.")
    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = client.get_json(f"{resource}?limit={page_size}&offset={offset}")
        if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
            raise ValueError(f"{resource} did not return an array of objects.")
        items.extend(page)
        if len(items) > max_items:
            raise ValueError(
                f"{resource} exceeded the fixture discovery safety cap of {max_items} items.",
            )
        if len(page) < page_size:
            return items
        offset += len(page)


def _ensure_subjects(client: LocalApiClient, manifest: dict[str, Any], current: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_name_and_kind = {
        (normalize_subject_name(item["name"]), item["kind"]): item
        for item in current
    }
    result: dict[str, dict[str, Any]] = {}
    for spec in manifest["subjects"]:
        identity = (normalize_subject_name(spec["name"]), spec["kind"])
        subject = by_name_and_kind.get(identity)
        if subject is None:
            subject = client.post_json("/subjects", {"kind": spec["kind"], "name": spec["name"], "metadata": {"fixture_revision": manifest["revision"]}})
            by_name_and_kind[identity] = subject
        aliases = {normalize_subject_name(item["name"]) for item in subject.get("aliases", []) if item.get("archived_at") is None}
        for alias in spec.get("aliases", []):
            if normalize_subject_name(alias) not in aliases:
                client.post_json(f"/subjects/{subject['id']}/aliases", {"name": alias})
        result[spec["key"]] = subject
    return result


def _ensure_documents(
    client: LocalApiClient,
    manifest: dict[str, Any],
    current: list[dict[str, Any]],
    subject_by_key: dict[str, dict[str, Any]],
    timeout_seconds: int,
) -> None:
    filenames = {
        item.get("original_filename")
        for item in current
        if _api_document_is_ready_and_indexed(item)
    }
    for spec in manifest["documents"]:
        if spec["filename"] in filenames:
            continue
        source = (REPOSITORY_ROOT / spec["path"]).resolve()
        fields = [("title", spec["title"]), ("detect_existing_versions", "true")]
        fields.extend(("subject_ids", subject_by_key[key]["id"]) for key in spec.get("subject_keys", []))
        _, location = client.upload("/documents", fields, source)
        if location:
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                job = client.get_json(location)
                status = job.get("status")
                if status == "succeeded":
                    break
                if status in {"failed", "cancelled"}:
                    raise ValueError(f"Ingestion job {location} ended with status {status!r}.")
                time.sleep(1)
            else:
                raise ValueError(f"Timed out waiting for ingestion job {location}.")


def _ensure_memberships(client: LocalApiClient, manifest: dict[str, Any], subjects: list[dict[str, Any]], documents: list[dict[str, Any]]) -> None:
    subject_by_key = {
        spec["key"]: next(
            item
            for item in subjects
            if normalize_subject_name(item["name"]) == normalize_subject_name(spec["name"])
            and item["kind"] == spec["kind"]
        )
        for spec in manifest["subjects"]
    }
    document_by_filename = _ready_documents_by_filename(documents)
    for spec in manifest["documents"]:
        document = document_by_filename.get(spec["filename"])
        if document is None:
            continue
        decisions = client.get_json(f"/documents/{document['id']}/subjects")
        by_subject = {item["subject_id"]: item for item in decisions}
        for key in spec.get("subject_keys", []):
            subject_id = subject_by_key[key]["id"]
            current = by_subject.get(subject_id)
            if current is None or current.get("state") != "assigned":
                client.put_json(
                    f"/documents/{document['id']}/subjects/{subject_id}",
                    {"state": "assigned", "expected_revision": current["revision"] if current else 0, "rationale": f"Fixture {manifest['revision']}"},
                )


def _remote_preflight(client: LocalApiClient, manifest: dict[str, Any], subjects: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[str]:
    changes = plan_fixture_changes(manifest, subjects, documents)
    problems = [f"missing subject key {key!r}" for key in changes["subjects"]]
    subject_by_key = {
        spec["key"]: next(
            (
                item
                for item in subjects
                if normalize_subject_name(item["name"]) == normalize_subject_name(spec["name"])
                and item["kind"] == spec["kind"]
            ),
            None,
        )
        for spec in manifest["subjects"]
    }
    for spec in manifest["subjects"]:
        subject = subject_by_key[spec["key"]]
        if subject is None:
            continue
        aliases = {
            normalize_subject_name(item["name"])
            for item in subject.get("aliases", [])
            if item.get("archived_at") is None
        }
        missing_aliases = {
            normalize_subject_name(item) for item in spec.get("aliases", [])
        } - aliases
        if missing_aliases:
            problems.append(f"subject {spec['name']!r} is missing aliases {sorted(missing_aliases)}")
    all_documents_by_filename: dict[str, list[dict[str, Any]]] = {}
    for item in documents:
        filename = item.get("original_filename")
        if isinstance(filename, str):
            all_documents_by_filename.setdefault(filename, []).append(item)
    document_by_filename = _ready_documents_by_filename(documents)
    for spec in manifest["documents"]:
        filename = spec["filename"]
        if filename in document_by_filename:
            continue
        candidates = all_documents_by_filename.get(filename, [])
        if not candidates:
            problems.append(f"missing document key {spec['key']!r}")
        else:
            states = [
                f"status={item.get('status')}, chunks={item.get('chunk_count', 0)}"
                for item in candidates
            ]
            problems.append(
                f"document {filename!r} is not ready and indexed ({', '.join(states)})",
            )
    for spec in manifest["documents"]:
        document = document_by_filename.get(spec["filename"])
        if document is None:
            continue
        assigned = {item["subject_id"] for item in client.get_json(f"/documents/{document['id']}/subjects") if item.get("state") == "assigned"}
        for key in spec.get("subject_keys", []):
            subject = subject_by_key.get(key)
            if subject is not None and subject["id"] not in assigned:
                problems.append(f"{spec['filename']!r} is not assigned to {subject['name']!r}")
    return problems


def _api_document_is_ready_and_indexed(document: dict[str, Any]) -> bool:
    chunk_count = document.get("chunk_count", 0)
    return (
        document.get("status") == "ready"
        and isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count > 0
    )


def _ready_documents_by_filename(
    documents: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(item["original_filename"]): item
        for item in documents
        if isinstance(item.get("original_filename"), str)
        and _api_document_is_ready_and_indexed(item)
    }


if __name__ == "__main__":
    raise SystemExit(main())
