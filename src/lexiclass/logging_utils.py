from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from .config import get_settings


class JsonLogFormatter(logging.Formatter):
    """Very small JSON log formatter suitable for production logs.

    Keys match common structured logging conventions.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - simple override
        base = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        # Include extras if present
        for key in ("module", "funcName", "lineno", "process", "thread"):
            base[key] = getattr(record, key, None)
        return json.dumps(base, ensure_ascii=False)


def _make_handler(log_file: Optional[str], formatter: logging.Formatter) -> logging.Handler:
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    return handler


def configure_logging(override_level: Optional[str] = None, force: bool = False) -> None:
    """Configure root logging once, using environment-driven settings.

    - override_level: if provided, forces the effective level for the root logger.
    - force: if True, resets handlers even if already configured.
    """
    settings = get_settings()
    effective_level = (override_level or settings.log_level).upper()

    # Avoid duplicate handlers unless forced
    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        root_logger.setLevel(getattr(logging, effective_level, logging.INFO))
        return

    # Clear any existing handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    if settings.log_format == "json":
        formatter: logging.Formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

    handler = _make_handler(settings.log_file, formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, effective_level, logging.INFO))

    # Tune noisy third-party libraries
    logging.getLogger("gensim").setLevel(getattr(logging, settings.gensim_log_level, logging.WARNING))
    logging.getLogger("sklearn").setLevel(getattr(logging, settings.sklearn_log_level, logging.WARNING))


