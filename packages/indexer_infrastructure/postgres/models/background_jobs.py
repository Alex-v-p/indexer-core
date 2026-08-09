from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Enum, Float, Index, Integer, String, Text, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from packages.indexer_application.dto import BackgroundJobStatus, BackgroundJobType
from packages.indexer_infrastructure.postgres.base import Base
from packages.indexer_infrastructure.postgres.models.shared import enum_values


class BackgroundJob(Base):
    __tablename__ = "background_jobs"
    __table_args__ = (
        CheckConstraint("progress >= 0 AND progress <= 1", name="ck_background_jobs_progress_range"),
        CheckConstraint(
            "job_type::text <> 'classify_document_organization' OR "
            "(payload ? 'document_id' AND payload ? 'document_version_id' AND "
            "payload ? 'policy_version' AND "
            "jsonb_typeof(payload->'document_id') = 'string' AND "
            "jsonb_typeof(payload->'document_version_id') = 'string' AND "
            "jsonb_typeof(payload->'policy_version') = 'string')",
            name="ck_background_jobs_organization_payload",
        ),
        Index("ix_background_jobs_status_scheduled", "status", "scheduled_at", "priority"),
        Index("ix_background_jobs_type_created", "job_type", "created_at"),
        Index("ix_background_jobs_heartbeat", "status", "heartbeat_at"),
        Index(
            "uq_background_jobs_active_dedupe_key",
            "dedupe_key",
            unique=True,
            postgresql_where=sa_text(
                "dedupe_key IS NOT NULL AND status IN ('queued', 'running')"
            ),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type: Mapped[BackgroundJobType] = mapped_column(
        Enum(BackgroundJobType, name="background_job_type", values_callable=enum_values),
        nullable=False,
    )
    status: Mapped[BackgroundJobStatus] = mapped_column(
        Enum(BackgroundJobStatus, name="background_job_status", values_callable=enum_values),
        nullable=False,
        default=BackgroundJobStatus.QUEUED,
        server_default=BackgroundJobStatus.QUEUED.value,
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100, server_default="100")
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    result: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, server_default="0")
    current_stage: Mapped[str | None] = mapped_column(String(255), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3, server_default="3")
    dedupe_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
