"""Compatibility facade for shared runtime settings.

The settings implementation is shared by deployable applications through
``packages.indexer_bootstrap``. Existing API imports remain stable here.
"""

from packages.indexer_bootstrap.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
