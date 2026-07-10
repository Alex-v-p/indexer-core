from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolConfig:
    """Metadata for one named capability used to compose query pipelines."""

    name: str
    kind: str
    version: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
