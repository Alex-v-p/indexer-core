from __future__ import annotations

from typing import Any

import httpx

from packages.rag_core.ports.keyword_indexes import KeywordDocument
from packages.rag_core.ports.keyword_indexes import KeywordStoreError


class QdrantKeywordCorpusSource:
    """Load searchable text and original evidence text from Qdrant payloads."""

    def __init__(
        self,
        *,
        base_url: str,
        collection_name: str,
        timeout_seconds: float = 30.0,
        scroll_batch_size: int = 256,
        search_text_field: str = "text",
        evidence_text_field: str = "text",
        fallback_to_evidence_text: bool = True,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if scroll_batch_size <= 0:
            raise ValueError("scroll_batch_size must be positive.")
        if not search_text_field.strip() or not evidence_text_field.strip():
            raise ValueError("Payload text fields must not be empty.")

        self._base_url = base_url.rstrip("/")
        self._collection_name = collection_name
        self._timeout_seconds = timeout_seconds
        self._scroll_batch_size = scroll_batch_size
        self._search_text_field = search_text_field
        self._evidence_text_field = evidence_text_field
        self._fallback_to_evidence_text = fallback_to_evidence_text

    async def list_documents(self) -> list[KeywordDocument]:
        documents: list[KeywordDocument] = []
        offset: object | None = None
        seen_offsets: set[str] = set()
        fallback_field = self._evidence_text_field if self._fallback_to_evidence_text else None

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            while True:
                request_body: dict[str, Any] = {
                    "limit": self._scroll_batch_size,
                    "with_payload": True,
                    "with_vector": False,
                }
                if offset is not None:
                    request_body["offset"] = offset

                response = await client.post(
                    f"{self._base_url}/collections/{self._collection_name}/points/scroll",
                    json=request_body,
                )
                if response.status_code == 404:
                    return []
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise KeywordStoreError(f"Qdrant keyword corpus scroll failed: {exc.response.text}") from exc

                points, next_offset = _parse_scroll_page(response.json())
                documents.extend(
                    _to_keyword_document(
                        point,
                        search_text_field=self._search_text_field,
                        evidence_text_field=self._evidence_text_field,
                        fallback_to_evidence_text=self._fallback_to_evidence_text,
                    )
                    for point in points
                    if _payload_text(point, self._search_text_field, fallback_field=fallback_field)
                )
                if next_offset is None or not points:
                    break
                offset_marker = repr(next_offset)
                if offset_marker in seen_offsets:
                    raise KeywordStoreError("Qdrant scroll returned a repeated page offset.")
                seen_offsets.add(offset_marker)
                offset = next_offset

        return documents


def _parse_scroll_page(body: dict[str, Any]) -> tuple[list[dict[str, Any]], object | None]:
    result = body.get("result")
    if not isinstance(result, dict):
        raise KeywordStoreError("Qdrant scroll response did not include a result object.")

    raw_points = result.get("points", [])
    if not isinstance(raw_points, list):
        raise KeywordStoreError("Qdrant scroll response did not include a points list.")

    points = [point for point in raw_points if isinstance(point, dict)]
    return points, result.get("next_page_offset")


def _to_keyword_document(
    point: dict[str, Any],
    *,
    search_text_field: str = "text",
    evidence_text_field: str = "text",
    fallback_to_evidence_text: bool = True,
) -> KeywordDocument:
    payload = point.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    fallback_field = evidence_text_field if fallback_to_evidence_text else None
    searchable_text = _payload_text(point, search_text_field, fallback_field=fallback_field)
    evidence_text = _payload_text(point, evidence_text_field)
    result_payload = {key: value for key, value in payload.items() if key != search_text_field}
    if evidence_text and evidence_text_field != search_text_field:
        result_payload["text"] = evidence_text

    return KeywordDocument(
        id=str(point.get("id")),
        text=searchable_text,
        payload=result_payload,
    )


def _payload_text(point: dict[str, Any], field: str, *, fallback_field: str | None = None) -> str:
    payload = point.get("payload") or {}
    if not isinstance(payload, dict):
        return ""
    text = payload.get(field)
    if isinstance(text, str) and text.strip():
        return text.strip()
    if fallback_field and fallback_field != field:
        fallback = payload.get(fallback_field)
        if isinstance(fallback, str):
            return fallback.strip()
    return ""
