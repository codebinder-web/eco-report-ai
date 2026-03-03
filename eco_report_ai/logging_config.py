"""Structured logging configuration for EcoReport AI.

Sets up both file and console handlers with appropriate formatters.
JSON formatter is used for file logs to enable structured log parsing.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for structured log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to JSON.

        Args:
            record: The log record to format.

        Returns:
            A JSON string representation of the record.
        """
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    log_filename: str = "eco_report_ai.log",
) -> logging.Logger:
    """Configure root logger with file (JSON) and console (human) handlers.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory in which to write the log file.
        log_filename: Name of the log file.

    Returns:
        The configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplication on re-configuration
    root_logger.handlers.clear()

    # ── File handler (JSON) ──────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_filename,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(file_handler)

    # ── Console handler (human-readable) ─────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "matplotlib", "PIL", "torch", "statsmodels"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger
