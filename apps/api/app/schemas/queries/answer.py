from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnswerPresentationResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    outcome: Literal[
        "complete",
        "partial",
        "blocked_constraint_no_match",
        "blocked_insufficient_evidence",
        "blocked_no_evidence",
    ]
    title: str
    body: str
    supported_information: list[str] = Field(default_factory=list)
    unresolved_information: list[str] = Field(default_factory=list)
    citation_count: int = Field(default=0, ge=0)
