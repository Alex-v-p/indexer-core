from __future__ import annotations

import uuid
from typing import Protocol


class DocumentVersionIndexActivator(Protocol):
    """Application capability for making one indexed source revision current."""

    async def activate_document_version(
        self,
        *,
        document_id: uuid.UUID,
        version_id: uuid.UUID,
    ) -> None: ...
