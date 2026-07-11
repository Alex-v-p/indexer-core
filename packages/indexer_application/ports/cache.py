from typing import Protocol


class CacheInvalidator(Protocol):
    """Small application port for invalidating a derived search index cache."""

    def invalidate(self) -> None:
        """Invalidate cached state so the next read rebuilds it."""
