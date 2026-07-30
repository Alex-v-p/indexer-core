from __future__ import annotations

from enum import StrEnum


class InformationNeedExecutionStatus(StrEnum):
    """Lifecycle state for one independently resolved information need."""

    PENDING = "pending"
    ACTIVE = "active"
    SUPPORTED = "supported"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


class InformationNeedRoute(StrEnum):
    """Next route chosen by the bounded information-need controller."""

    RETRY = "retry"
    RECLASSIFY = "reclassify"
    COMPLETE_SUPPORTED = "complete_supported"
    COMPLETE_EXHAUSTED = "complete_exhausted"
