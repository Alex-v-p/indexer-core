from __future__ import annotations

from collections.abc import Iterable
import uuid
from typing import Any

import httpx

from packages.rag_core.documents import DocumentNameConstraint, DocumentVersionConstraint, VersionSelectionMode
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint, DocumentDateField
from packages.rag_core.ports.vector_indexes import (
    VectorPayloadCondition,
    VectorPoint,
    VectorSearchResult,
    VectorStoreError,
)


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
                existing_names = _validate_existing_collection_vectors(
                    response.json(),
                    expected_names=self.vector_names,
                    expected_size=self.vector_size,
                )
                for vector_name in self.vector_names:
                    if vector_name in existing_names:
                        continue
                    create_vector_response = await client.put(
                        f"{self.base_url}/collections/{self.collection_name}/vectors/{vector_name}",
                        params={"wait": "true"},
                        json={
                            "dense": {
                                "size": self.vector_size,
                                "distance": "Cosine",
                            },
                        },
                    )
                    try:
                        create_vector_response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise VectorStoreError(
                            f"Qdrant named-vector creation failed for {vector_name!r}: "
                            f"{exc.response.text}",
                        ) from exc
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

    async def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/delete",
                params={"wait": "true"},
                json={"points": point_ids},
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(f"Qdrant point deletion failed: {exc.response.text}") from exc

    async def delete_document_version(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> None:
        """Delete every chunk and hierarchy point owned by one source version."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/delete",
                params={"wait": "true"},
                json={
                    "filter": {
                        "must": [
                            {
                                "key": "document_id",
                                "match": {"value": str(document_id)},
                            },
                            {
                                "key": "document_version_id",
                                "match": {"value": str(version_id)},
                            },
                        ]
                    }
                },
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(
                f"Qdrant document-version deletion failed: {exc.response.text}"
            ) from exc

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        document_constraint: DocumentNameConstraint | None = None,
        version_constraint: DocumentVersionConstraint | None = None,
        date_constraints: tuple[DocumentDateConstraint, ...] = (),
        payload_conditions: tuple[VectorPayloadCondition, ...] = (),
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
                    **_constraint_filter_body(
                        document_constraint,
                        version_constraint,
                        date_constraints,
                        payload_conditions,
                    ),
                },
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VectorStoreError(f"Qdrant vector search failed: {exc.response.text}") from exc

        return _parse_search_results(response.json())

    async def activate_document_version(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> None:
        """Application-facing alias for promoting an indexed document version."""

        await self.mark_document_version_current(
            document_id=str(document_id),
            version_id=str(version_id),
        )

    async def mark_document_version_current(self, *, document_id: str, version_id: str) -> None:
        """Mark one version current and every sibling version non-current in Qdrant."""

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            promote_response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/payload",
                params={"wait": "true"},
                json={
                    "payload": {"is_latest_version": True},
                    "filter": {
                        "must": [
                            {"key": "document_id", "match": {"value": document_id}},
                            {"key": "document_version_id", "match": {"value": version_id}},
                        ]
                    },
                },
            )
            demote_response = await client.post(
                f"{self.base_url}/collections/{self.collection_name}/points/payload",
                params={"wait": "true"},
                json={
                    "payload": {"is_latest_version": False},
                    "filter": {
                        "must": [{"key": "document_id", "match": {"value": document_id}}],
                        "must_not": [{"key": "document_version_id", "match": {"value": version_id}}],
                    },
                },
            )
        for response in (promote_response, demote_response):
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


def _constraint_filter_body(
    document_constraint: DocumentNameConstraint | None,
    version_constraint: DocumentVersionConstraint | None,
    date_constraints: tuple[DocumentDateConstraint, ...],
    payload_conditions: tuple[VectorPayloadCondition, ...] = (),
) -> dict[str, Any]:
    must: list[dict[str, Any]] = []

    if document_constraint is not None and document_constraint.active:
        raw_names = list(document_constraint.names)
        normalized_names = list(document_constraint.normalized_names)
        must.append(
            {
                "should": [
                    {"key": "document_title", "match": {"any": raw_names}},
                    {"key": "original_filename", "match": {"any": raw_names}},
                    {"key": "document_title_normalized", "match": {"any": normalized_names}},
                    {"key": "original_filename_normalized", "match": {"any": normalized_names}},
                ],
            },
        )

    relative_version_scope = bool(date_constraints) and version_constraint is not None and (
        version_constraint.mode
        in {
            VersionSelectionMode.LATEST,
            VersionSelectionMode.OLDEST,
            VersionSelectionMode.PREVIOUS,
            VersionSelectionMode.ALL_EXCEPT_LATEST,
            VersionSelectionMode.LATEST_AND_PREVIOUS,
            VersionSelectionMode.OLDEST_AND_LATEST,
        }
    )
    if version_constraint is not None and version_constraint.active and not relative_version_scope:
        if version_constraint.mode is VersionSelectionMode.LATEST:
            must.append({"key": "is_latest_version", "match": {"value": True}})
        elif version_constraint.mode is VersionSelectionMode.OLDEST:
            must.append({"key": "document_version_number", "match": {"value": 1}})
        elif version_constraint.mode is VersionSelectionMode.PREVIOUS:
            must.append({"key": "is_latest_version", "match": {"value": False}})
        elif version_constraint.mode is VersionSelectionMode.ALL_EXCEPT_LATEST:
            must.append({"key": "is_latest_version", "match": {"value": False}})
        elif version_constraint.mode is VersionSelectionMode.OLDEST_AND_LATEST:
            must.append(
                {
                    "should": [
                        {"key": "document_version_number", "match": {"value": 1}},
                        {"key": "is_latest_version", "match": {"value": True}},
                    ],
                },
            )
        elif version_constraint.mode is VersionSelectionMode.SPECIFIC:
            must.append(
                {
                    "key": "document_version_number",
                    "match": {"any": list(version_constraint.version_numbers)},
                },
            )

    for payload_condition in payload_conditions:
        values = list(payload_condition.values)
        match = {"value": values[0]} if len(values) == 1 else {"any": values}
        must.append({"key": payload_condition.field, "match": match})

    for constraint in date_constraints:
        range_body: dict[str, float] = {}
        if constraint.date_range.start is not None:
            range_body["gte"] = constraint.date_range.start.timestamp()
        if constraint.date_range.end is not None:
            range_body["lt"] = constraint.date_range.end.timestamp()
        if constraint.field is DocumentDateField.UPLOADED_AT:
            must.append({"key": "uploaded_at_epoch", "range": range_body})
        elif constraint.field is DocumentDateField.PUBLISHED_AT:
            must.append({"key": "published_at_epoch", "range": range_body})
        else:
            must.append(
                {
                    "should": [
                        {"key": "published_at_epoch", "range": range_body},
                        {"key": "uploaded_at_epoch", "range": range_body},
                    ],
                },
            )

    return {"filter": {"must": must}} if must else {}


def _validate_existing_collection_vectors(
    body: dict[str, Any],
    *,
    expected_names: tuple[str, ...],
    expected_size: int,
) -> set[str]:
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

    existing_names = set(vectors)
    for name in expected_names:
        vector_config = vectors.get(name)
        if vector_config is None:
            continue
        if not isinstance(vector_config, dict):
            raise VectorStoreError(f"Qdrant vector configuration for {name!r} is invalid.")
        size = vector_config.get("size")
        if size != expected_size:
            raise VectorStoreError(
                f"Qdrant vector {name!r} has size {size}, but the application expects {expected_size}.",
            )
    return existing_names


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
