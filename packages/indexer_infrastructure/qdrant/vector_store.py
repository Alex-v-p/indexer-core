from __future__ import annotations

from typing import Any

import httpx

from packages.rag_core.ports.vector_indexes import VectorPoint, VectorSearchResult
from packages.rag_core.ports.vector_indexes import VectorStoreError


class QdrantVectorStore:
    """Small Qdrant REST adapter used by ingestion and baseline retrieval.

    Keeping this adapter lightweight avoids coupling the project to one client
    library while the provider interface is still small in Phase 1.
    """

    def __init__(self, *, base_url: str, collection_name: str, vector_size: int, timeout_seconds: float = 30.0) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout_seconds = timeout_seconds

    async def ensure_collection(self) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.base_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                return
            if response.status_code != 404:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise VectorStoreError(f"Qdrant collection lookup failed: {exc.response.text}") from exc

            create_response = await client.put(
                f"{self.base_url}/collections/{self.collection_name}",
                json={
                    "vectors": {
                        "size": self.vector_size,
                        "distance": "Cosine",
                    },
                },
            )
            try:
                create_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise VectorStoreError(f"Qdrant collection creation failed: {exc.response.text}") from exc

    async def upsert_points(self, points: list[VectorPoint], *, batch_size: int = 64) -> None:
        if not points:
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                response = await client.put(
                    f"{self.base_url}/collections/{self.collection_name}/points",
                    params={"wait": "true"},
                    json={
                        "points": [
                            {
                                "id": point.id,
                                "vector": point.vector,
                                "payload": point.payload,
                            }
                            for point in batch
                        ],
                    },
                )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise VectorStoreError(f"Qdrant point upsert failed: {exc.response.text}") from exc

    async def search_by_vector(self, vector: list[float], *, top_k: int) -> list[VectorSearchResult]:
        """Search Qdrant using a dense query vector.

        Qdrant has supported both the older `/points/search` endpoint and the
        newer query-points endpoint across recent versions. The adapter tries
        the older endpoint first because it matches the simple Phase 1 use case,
        then falls back to query-points for newer deployments.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if len(vector) != self.vector_size:
            raise VectorStoreError(
                f"Query vector has size {len(vector)}, but collection expects {self.vector_size}.",
            )

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/search",
                json={
                    "vector": vector,
                    "limit": top_k,
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            if response.status_code in {404, 405}:
                response = await client.post(
                    f"{self.base_url}/collections/{self.collection_name}/points/query",
                    json={
                        "query": vector,
                        "limit": top_k,
                        "with_payload": True,
                        "with_vector": False,
                    },
                )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(f"Qdrant vector search failed: {exc.response.text}") from exc

        return _parse_search_results(response.json())


def _parse_search_results(body: dict[str, Any]) -> list[VectorSearchResult]:
    result = body.get("result")
    if isinstance(result, dict):
        raw_points = result.get("points", [])
    else:
        raw_points = result

    if not isinstance(raw_points, list):
        raise VectorStoreError("Qdrant search response did not include a result list.")

    parsed: list[VectorSearchResult] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        point_id = point.get("id")
        payload = point.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        parsed.append(
            VectorSearchResult(
                id=str(point_id),
                score=_optional_float(point.get("score")),
                payload=payload,
            ),
        )
    return parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
