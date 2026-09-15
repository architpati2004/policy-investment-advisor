"""Logging setup for the whole backend.

Call :func:`configure_logging` once at process start (done in ``main.py``); every
module then uses ``logging.getLogger(__name__)``. No ``print`` calls anywhere.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once, idempotently."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    root.addHandler(handler)

    # Third-party libraries are chatty at DEBUG; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "urllib3", "faiss", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so modules do not import ``logging`` directly."""
    return logging.getLogger(name)
