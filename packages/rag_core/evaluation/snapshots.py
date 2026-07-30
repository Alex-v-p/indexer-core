from __future__ import annotations

from packages.rag_core.agents.runtime import TraceEvent
from packages.rag_core.generation import CitationItem
from packages.rag_core.retrieval.models import EvidenceItem


def evidence_snapshot(evidence: EvidenceItem) -> dict[str, object]:
    return {
        "rank": evidence.rank,
        "score": evidence.score,
        "text": evidence.text,
        "qdrant_chunk_index_id": string_or_none(evidence.qdrant_chunk_index_id),
        "document_id": string_or_none(evidence.document_id),
        "document_version_id": string_or_none(evidence.document_version_id),
        "metadata": evidence.metadata,
    }


def citation_snapshot(citation: CitationItem) -> dict[str, object]:
    return {
        "citation_index": citation.citation_index,
        "evidence_rank": citation.evidence_rank,
        "label": citation.label,
        "page_number": citation.page_number,
        "quote": citation.quote,
        "qdrant_chunk_index_id": string_or_none(citation.qdrant_chunk_index_id),
        "document_id": string_or_none(citation.document_id),
        "document_version_id": string_or_none(citation.document_version_id),
        "metadata": citation.metadata,
    }


def trace_snapshot(trace: TraceEvent) -> dict[str, object]:
    return {
        "step_order": trace.step_order,
        "name": trace.name,
        "step_type": trace.step_type,
        "status": trace.status,
        "duration_ms": trace.duration_ms,
        "input_summary": trace.input_summary,
        "output_summary": trace.output_summary,
        "error_message": trace.error_message,
        "metadata": trace.metadata,
    }


def string_or_none(value: object) -> str | None:
    return None if value is None else str(value)
