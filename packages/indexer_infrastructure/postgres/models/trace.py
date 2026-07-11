from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.indexer_application.dto import TraceStepStatus

from packages.indexer_infrastructure.postgres.base import Base
from packages.indexer_infrastructure.postgres.models.shared import enum_values

if TYPE_CHECKING:
    from packages.indexer_infrastructure.postgres.models.query_runs import QueryRun


class TraceStep(Base):
    """Auditable step emitted by a future graph runner."""

    __tablename__ = "trace_steps"
    __table_args__ = (
        UniqueConstraint("query_run_id", "step_order", name="uq_trace_steps_query_run_step_order"),
        Index("ix_trace_steps_query_run_id", "query_run_id"),
        Index("ix_trace_steps_name", "name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    step_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[TraceStepStatus] = mapped_column(
        Enum(TraceStepStatus, name="trace_step_status", values_callable=enum_values),
        nullable=False,
        default=TraceStepStatus.PENDING,
        server_default=TraceStepStatus.PENDING.value,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query_run: Mapped[QueryRun] = relationship(back_populates="trace_steps")
