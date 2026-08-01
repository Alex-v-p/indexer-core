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


class UnsupportedStrictDocumentScopeError(RuntimeError):
    """A retriever cannot prove enforcement of a requested strict scope."""


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
    if constraints is not None and constraints.document_scope.is_strict_empty:
        return []
    if constraints is not None and callable_accepts_parameter(method, "constraints"):
        evidence = await method(question, top_k=top_k, constraints=constraints)
    else:
        if constraints is not None and constraints.document_scope.strict:
            raise UnsupportedStrictDocumentScopeError(
                f"{type(retriever).__name__} does not accept strict retrieval constraints.",
            )
        evidence = await method(question, top_k=top_k)
    return enforce_document_scope(evidence, constraints)


async def retrieve_batch_compatibly(
    retriever: Retriever,
    question: str,
    *,
    top_k: int,
    constraints: RetrievalConstraints | None,
) -> RetrievalBatch:
    method = getattr(retriever, "retrieve_with_metadata", None)
    if constraints is not None and constraints.document_scope.is_strict_empty:
        return RetrievalBatch(
            evidence=[],
            metadata={"strict_document_scope_short_circuit": True},
        )
    if callable(method):
        if constraints is not None and callable_accepts_parameter(method, "constraints"):
            result = await method(question, top_k=top_k, constraints=constraints)
        elif constraints is not None and constraints.document_scope.strict:
            raise UnsupportedStrictDocumentScopeError(
                f"{type(retriever).__name__}.retrieve_with_metadata does not accept strict constraints.",
            )
        else:
            result = await method(question, top_k=top_k)
        if not isinstance(result, RetrievalBatch):
            raise TypeError("retrieve_with_metadata must return RetrievalBatch.")
        result.evidence, rejected = enforce_document_scope_with_count(
            result.evidence,
            constraints,
        )
        result.metadata["out_of_scope_rejected_count"] = int(
            result.metadata.get("out_of_scope_rejected_count", 0)
        ) + rejected
        return result
    retrieve = retriever.retrieve
    if constraints is not None and callable_accepts_parameter(retrieve, "constraints"):
        evidence = await retrieve(question, top_k=top_k, constraints=constraints)
    else:
        if constraints is not None and constraints.document_scope.strict:
            raise UnsupportedStrictDocumentScopeError(
                f"{type(retriever).__name__} does not accept strict retrieval constraints.",
            )
        evidence = await retrieve(question, top_k=top_k)
    evidence, rejected = enforce_document_scope_with_count(evidence, constraints)
    return RetrievalBatch(
        evidence=evidence,
        metadata={"out_of_scope_rejected_count": rejected},
    )


def enforce_document_scope(
    evidence: list[EvidenceItem],
    constraints: RetrievalConstraints | None,
) -> list[EvidenceItem]:
    evidence, _ = enforce_document_scope_with_count(evidence, constraints)
    return evidence


def enforce_document_scope_with_count(
    evidence: list[EvidenceItem],
    constraints: RetrievalConstraints | None,
) -> tuple[list[EvidenceItem], int]:
    if constraints is None or constraints.document_scope.is_global:
        return evidence, 0
    scope = constraints.document_scope
    scoped = [item for item in evidence if scope.allows(item.document_id)]
    return scoped, len(evidence) - len(scoped)
