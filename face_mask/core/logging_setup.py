from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "payload"):
            payload["payload"] = record.payload
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: dict[str, Any]) -> logging.Logger:
    runtime_config = config.get("runtime", {})
    logging_config = config.get("logging", {})
    logger_name = runtime_config.get("logger_name", "face_mask")
    level_name = logging_config.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(console_handler)

    log_file = logging_config.get("file")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(path, maxBytes=512_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(level)
        if logging_config.get("json", False):
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(file_handler)

    return logger


def log_event(logger: logging.Logger, level: int, event: str, **payload: Any) -> None:
    logger.log(level, event, extra={"event": event, "payload": payload})
