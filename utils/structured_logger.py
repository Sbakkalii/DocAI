"""
Structured JSON logger with session_id and step correlation.
"""

import json
import logging
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes session_id, step, elapsed, and extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        obj: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("session_id", "step", "elapsed"):
            val = getattr(record, key, None)
            if val is not None:
                obj[key] = val
        if record.exc_info and record.exc_info[1]:
            obj["error"] = str(record.exc_info[1])
        return json.dumps(obj, default=str, ensure_ascii=False)


def setup_structured_logging(level: str = "INFO", json_output: bool = True):
    """Configure root logger for structured JSON output."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        StructuredFormatter() if json_output
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str, session_id: Optional[str] = None) -> logging.LoggerAdapter:
    """Return a logger adapter that injects session_id into every record."""
    logger = logging.getLogger(name)
    extra: Dict[str, Any] = {}
    if session_id:
        extra["session_id"] = session_id
    return logging.LoggerAdapter(logger, extra) if extra else logging.LoggerAdapter(logger, {})
