"""Shared runtime configuration and dependency composition for deployable apps.

This outer-layer package may depend on the core, application, and infrastructure
packages, but it must never depend on a deployable app. Both the API and worker
use it to assemble identical providers without importing one another.
"""

from packages.indexer_bootstrap.config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
