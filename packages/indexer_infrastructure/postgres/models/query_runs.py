from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, Index, Integer, String, Text, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.indexer_application.dto import QueryRunStatus

from packages.indexer_infrastructure.postgres.base import Base
from packages.indexer_infrastructure.postgres.models.shared import enum_values

if TYPE_CHECKING:
    from packages.indexer_infrastructure.postgres.models.citations import Citation
    from packages.indexer_infrastructure.postgres.models.evidence import Evidence
    from packages.indexer_infrastructure.postgres.models.trace import TraceStep


class QueryRun(Base):
    """One execution of a question through a pipeline or graph.

    The submitted question is stored directly on the run for now. A separate
    saved-query table can be added later if the product needs reusable queries,
    evaluation datasets, or query templates.
    """

    __tablename__ = "query_runs"
    __table_args__ = (
        Index("ix_query_runs_status", "status"),
        Index("ix_query_runs_started_at", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    evidence_items: Mapped[list[Evidence]] = relationship(
        "Evidence",
        back_populates="query_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Evidence.rank",
    )
    citations: Mapped[list[Citation]] = relationship(
        "Citation",
        back_populates="query_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Citation.citation_index",
    )
    trace_steps: Mapped[list[TraceStep]] = relationship(
        "TraceStep",
        back_populates="query_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TraceStep.step_order",
    )
