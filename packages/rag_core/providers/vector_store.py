from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """Point payload sent to a vector store."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantVectorStore:
    """Small Qdrant REST adapter used by ingestion.

    Keeping this adapter lightweight avoids coupling the project to one client
    library while the provider interface is still small in Phase 1.
    """

    def __init__(self, *, base_url: str, collection_name: str, vector_size: int, timeout_seconds: float = 30.0) -> None:
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
                response.raise_for_status()

            create_response = await client.put(
                f"{self.base_url}/collections/{self.collection_name}",
                json={
                    "vectors": {
                        "size": self.vector_size,
                        "distance": "Cosine",
                    },
                },
            )
            create_response.raise_for_status()

    async def upsert_points(self, points: list[VectorPoint], *, batch_size: int = 64) -> None:
        if not points:
            return

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
                response.raise_for_status()
