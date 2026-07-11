from __future__ import annotations

from typing import Any

import httpx

from packages.rag_core.ports.keyword_indexes import KeywordDocument
from packages.rag_core.ports.keyword_indexes import KeywordStoreError


class QdrantKeywordCorpusSource:
    """Load chunk text and metadata from Qdrant point payloads.

    Qdrant remains the single chunk-payload store for the current repository.
    The source uses the scroll endpoint so hybrid retrieval also works for
    documents indexed before the keyword pipeline was introduced.
    """

    def __init__(
        self,
        *,
        base_url: str,
        collection_name: str,
        timeout_seconds: float = 30.0,
        scroll_batch_size: int = 256,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if scroll_batch_size <= 0:
            raise ValueError("scroll_batch_size must be positive.")

        self._base_url = base_url.rstrip("/")
        self._collection_name = collection_name
        self._timeout_seconds = timeout_seconds
        self._scroll_batch_size = scroll_batch_size

    async def list_documents(self) -> list[KeywordDocument]:
        documents: list[KeywordDocument] = []
        offset: object | None = None
        seen_offsets: set[str] = set()

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
                documents.extend(_to_keyword_document(point) for point in points if _payload_text(point))
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


def _to_keyword_document(point: dict[str, Any]) -> KeywordDocument:
    payload = point.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    return KeywordDocument(
        id=str(point.get("id")),
        text=_payload_text(point),
        payload={key: value for key, value in payload.items() if key != "text"},
    )


def _payload_text(point: dict[str, Any]) -> str:
    payload = point.get("payload") or {}
    if not isinstance(payload, dict):
        return ""
    text = payload.get("text")
    return text.strip() if isinstance(text, str) else ""
