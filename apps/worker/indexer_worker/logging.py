from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str) -> None:
    """Configure structured-enough stdout logging for the worker process."""

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
