import logging
import sys
from typing import Any


class UvicornAccessFilter(logging.Filter):
    """Reduce noisy health-check access logs from orchestrators."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message


def configure_logging(log_level: str) -> None:
    """Configure application logging once at startup."""

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger("uvicorn.access").addFilter(UvicornAccessFilter())


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger adapter with optional static context."""

    return logging.LoggerAdapter(logging.getLogger(name), context)
