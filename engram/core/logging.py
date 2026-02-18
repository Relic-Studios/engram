"""
engram.core.logging — Structured logging support (OB-3).

Provides a JSON formatter for stdlib logging that emits structured
log records.  When enabled, all engram loggers output machine-parseable
JSON lines instead of human-readable text.

Usage::

    from engram.core.logging import configure_logging

    configure_logging(structured=True, level="INFO")

When ``structured=False`` (default), logging is left untouched —
the standard stdlib formatting applies, preserving backwards
compatibility.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Fields emitted:
      - ``ts``: ISO-8601 timestamp
      - ``level``: log level name
      - ``logger``: logger name
      - ``msg``: formatted message
      - ``module``: source module
      - ``func``: source function
      - ``line``: source line number

    If the record has an ``exc_info`` tuple, the formatted traceback
    is included as ``exception``.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S")
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


def configure_logging(
    structured: bool = False,
    level: str = "INFO",
    logger_name: str = "engram",
) -> None:
    """Configure engram's logging subsystem.

    Parameters
    ----------
    structured:
        When True, all engram loggers emit JSON lines via
        ``StructuredFormatter``.  When False, stdlib defaults apply.
    level:
        Log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, etc.).
    logger_name:
        Root logger name to configure (default ``"engram"``).
    """
    root = logging.getLogger(logger_name)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if structured:
        # Remove any existing handlers on the engram root logger
        root.handlers.clear()

        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        root.addHandler(handler)

        # Prevent propagation to the root logger to avoid duplicate
        # output when the application has its own handler.
        root.propagate = False
