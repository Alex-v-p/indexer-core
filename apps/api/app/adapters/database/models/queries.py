from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.adapters.database.base import Base
from app.adapters.database.models.shared import enum_values

if TYPE_CHECKING:
    from app.adapters.database.models.documents import Chunk, Document, DocumentVersion
    from app.adapters.database.models.trace import TraceStep


class QueryRunStatus(StrEnum):
    """Execution status for a user question run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Query(Base):
    """Canonical user question/request submitted to the system."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_question: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    runs: Mapped[list[QueryRun]] = relationship(
        back_populates="query",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="QueryRun.started_at",
    )


class QueryRun(Base):
    """One execution of a question through a pipeline or graph."""

    __tablename__ = "query_runs"
    __table_args__ = (
        Index("ix_query_runs_query_id_started_at", "query_id", "started_at"),
        Index("ix_query_runs_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=True,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[QueryRunStatus] = mapped_column(
        Enum(QueryRunStatus, name="query_run_status", values_callable=enum_values),
        nullable=False,
        default=QueryRunStatus.PENDING,
        server_default=QueryRunStatus.PENDING.value,
    )
    pipeline_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query: Mapped[Query | None] = relationship(back_populates="runs")
    evidence_items: Mapped[list[Evidence]] = relationship(
        back_populates="query_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Evidence.rank",
    )
    citations: Mapped[list[Citation]] = relationship(
        back_populates="query_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Citation.citation_index",
    )
    trace_steps: Mapped[list[TraceStep]] = relationship(
        back_populates="query_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TraceStep.step_order",
    )


class Evidence(Base):
    """Retrieved or selected evidence considered during answer generation."""

    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint("query_run_id", "rank", name="uq_evidence_query_run_rank"),
        Index("ix_evidence_query_run_id", "query_run_id"),
        Index("ix_evidence_chunk_id", "chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query_run: Mapped[QueryRun] = relationship(back_populates="evidence_items")
    chunk: Mapped[Chunk | None] = relationship(back_populates="evidence_items")
    document: Mapped[Document | None] = relationship(back_populates="evidence_items")
    document_version: Mapped[DocumentVersion | None] = relationship(back_populates="evidence_items")
    citations: Mapped[list[Citation]] = relationship(back_populates="evidence")


class Citation(Base):
    """Answer citation pointing back to evidence and source location metadata."""

    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("query_run_id", "citation_index", name="uq_citations_query_run_citation_index"),
        Index("ix_citations_query_run_id", "query_run_id"),
        Index("ix_citations_evidence_id", "evidence_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query_run: Mapped[QueryRun] = relationship(back_populates="citations")
    evidence: Mapped[Evidence | None] = relationship(back_populates="citations")
    document: Mapped[Document | None] = relationship(back_populates="citations")
    document_version: Mapped[DocumentVersion | None] = relationship(back_populates="citations")
    chunk: Mapped[Chunk | None] = relationship(back_populates="citations")
