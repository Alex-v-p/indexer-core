from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx

from packages.rag_core.documents import DocumentVersionConstraint, VersionSelectionMode
from packages.rag_core.ports.vector_indexes import VectorPoint, VectorSearchResult, VectorStoreError


class QdrantVectorStore:
    """Qdrant REST adapter for logical points with named dense vectors."""

    def __init__(
        self,
        *,
        base_url: str,
        collection_name: str,
        vector_size: int,
        vector_names: Iterable[str],
        timeout_seconds: float = 30.0,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        normalized_names = tuple(dict.fromkeys(name.strip() for name in vector_names if name.strip()))
        if not normalized_names:
            raise ValueError("At least one vector name must be configured.")

        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.vector_names = normalized_names
        self.timeout_seconds = timeout_seconds

    async def ensure_collection(self) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(f"{self.base_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                _validate_collection_vectors(
                    response.json(),
                    expected_names=self.vector_names,
                    expected_size=self.vector_size,
                )
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
                        name: {
                            "size": self.vector_size,
                            "distance": "Cosine",
                        }
                        for name in self.vector_names
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

        for point in points:
            self._validate_point(point)

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
                                "vector": point.vectors,
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

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        version_constraint: DocumentVersionConstraint | None = None,
    ) -> list[VectorSearchResult]:
        """Search one named vector space through Qdrant's Query API."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if vector_name not in self.vector_names:
            raise VectorStoreError(
                f"Vector name {vector_name!r} is not configured for collection {self.collection_name!r}.",
            )
        if len(vector) != self.vector_size:
            raise VectorStoreError(
                f"Query vector has size {len(vector)}, but collection expects {self.vector_size}.",
            )

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/query",
                json={
                    "query": vector,
                    "using": vector_name,
                    "limit": top_k,
                    "with_payload": True,
                    "with_vector": False,
                    **_version_filter_body(version_constraint),
                },
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(f"Qdrant vector search failed: {exc.response.text}") from exc

        return _parse_search_results(response.json())

    async def mark_document_version_current(self, *, document_id: str, version_id: str) -> None:
        """Mark every other version of a document as non-latest in Qdrant."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/payload",
                json={
                    "payload": {"is_latest_version": False},
                    "filter": {
                        "must": [{"key": "document_id", "match": {"value": document_id}}],
                        "must_not": [{"key": "document_version_id", "match": {"value": version_id}}],
                    },
                },
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(f"Qdrant version promotion failed: {exc.response.text}") from exc

    def _validate_point(self, point: VectorPoint) -> None:
        if not point.vectors:
            raise VectorStoreError(f"Point {point.id!r} must contain at least one named vector.")

        unknown_names = set(point.vectors) - set(self.vector_names)
        if unknown_names:
            raise VectorStoreError(
                f"Point {point.id!r} contains unconfigured vectors: {sorted(unknown_names)}.",
            )

        for name, vector in point.vectors.items():
            if len(vector) != self.vector_size:
                raise VectorStoreError(
                    f"Point {point.id!r} vector {name!r} has size {len(vector)}, "
                    f"but collection expects {self.vector_size}.",
                )

def _version_filter_body(constraint: DocumentVersionConstraint | None) -> dict[str, Any]:
    if constraint is None or not constraint.active:
        return {}
    if constraint.mode is VersionSelectionMode.LATEST:
        return {"filter": {"must": [{"key": "is_latest_version", "match": {"value": True}}]}}
    if constraint.mode is VersionSelectionMode.PREVIOUS:
        return {"filter": {"must": [{"key": "is_latest_version", "match": {"value": False}}]}}
    if constraint.mode is VersionSelectionMode.SPECIFIC:
        return {
            "filter": {
                "must": [
                    {
                        "key": "document_version_number",
                        "match": {"any": list(constraint.version_numbers)},
                    },
                ],
            },
        }
    return {}


def _validate_collection_vectors(
    body: dict[str, Any],
    *,
    expected_names: tuple[str, ...],
    expected_size: int,
) -> None:
    result = body.get("result")
    if not isinstance(result, dict):
        raise VectorStoreError("Qdrant collection response did not include a result object.")

    config = result.get("config")
    params = config.get("params") if isinstance(config, dict) else None
    vectors = params.get("vectors") if isinstance(params, dict) else None
    if not isinstance(vectors, dict):
        raise VectorStoreError("Qdrant collection response did not include vector configuration.")

    if "size" in vectors:
        raise VectorStoreError(
            "The Qdrant collection uses an unnamed vector. Recreate the collection with the configured named vectors.",
        )

    missing = [name for name in expected_names if name not in vectors]
    if missing:
        raise VectorStoreError(
            f"Qdrant collection is missing configured named vectors: {', '.join(missing)}.",
        )

    for name in expected_names:
        vector_config = vectors.get(name)
        if not isinstance(vector_config, dict):
            raise VectorStoreError(f"Qdrant vector configuration for {name!r} is invalid.")
        size = vector_config.get("size")
        if size != expected_size:
            raise VectorStoreError(
                f"Qdrant vector {name!r} has size {size}, but the application expects {expected_size}.",
            )


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
