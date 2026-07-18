from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol

from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


@dataclass(slots=True)
class RetrievalBatch:
    """Evidence plus strategy-specific metadata for graph traces and evaluation."""

    evidence: list[EvidenceItem]
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Interface for query-time retrieval implementations."""

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        constraints: RetrievalConstraints | None = None,
    ) -> list[EvidenceItem]:
        """Return ranked evidence for a question."""


def callable_accepts_parameter(function: object, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


async def retrieve_compatibly(
    retriever: Retriever,
    question: str,
    *,
    top_k: int,
    constraints: RetrievalConstraints | None,
) -> list[EvidenceItem]:
    method = retriever.retrieve
    if constraints is not None and callable_accepts_parameter(method, "constraints"):
        return await method(question, top_k=top_k, constraints=constraints)
    return await method(question, top_k=top_k)


async def retrieve_batch_compatibly(
    retriever: Retriever,
    question: str,
    *,
    top_k: int,
    constraints: RetrievalConstraints | None,
) -> RetrievalBatch:
    method = getattr(retriever, "retrieve_with_metadata", None)
    if callable(method):
        if constraints is not None and callable_accepts_parameter(method, "constraints"):
            result = await method(question, top_k=top_k, constraints=constraints)
        else:
            result = await method(question, top_k=top_k)
        if not isinstance(result, RetrievalBatch):
            raise TypeError("retrieve_with_metadata must return RetrievalBatch.")
        return result
    return RetrievalBatch(
        evidence=await retrieve_compatibly(
            retriever,
            question,
            top_k=top_k,
            constraints=constraints,
        ),
    )
