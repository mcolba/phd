"""Logging helpers for long-running calibration entry points."""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Generator

_RUN_KEY = contextvars.ContextVar("calibration_run_key", default="-")


class _RunKeyFilter(logging.Filter):
    """Attach the current calibration key to each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_key = _RUN_KEY.get()
        return True


class _TqdmLoggingHandler(logging.Handler):
    """Write log records without corrupting an active tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))
        self.flush()


def configure_calibration_logging(
    log_file_name: str,
    *,
    file_level: int | str = logging.WARNING,
    stream_level: int | str = logging.WARNING,
) -> Path:
    """Configure file and progress-safe stream logging for a calibration script.

    Args:
        log_file_name: Log filename without directory components.
        file_level: Minimum level written to the log file.
        stream_level: Minimum level written to the terminal.

    Returns:
        Path of the configured log file.
    """
    if not log_file_name or Path(log_file_name).name != log_file_name:
        msg = "log_file_name must be a file name without directory components"
        raise ValueError(msg)
    project_root = Path(__file__).resolve().parents[2]
    log_file_path = project_root / "results" / "logging" / log_file_name
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    run_key_filter = _RunKeyFilter()

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(run_key)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler.addFilter(run_key_filter)

    stream_handler = _TqdmLoggingHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(run_key)s] - %(levelname)s - %(message)s"))
    stream_handler.addFilter(run_key_filter)

    logging.basicConfig(
        level=min(file_handler.level, stream_handler.level),
        handlers=[file_handler, stream_handler],
    )
    return log_file_path


@contextmanager
def calibration_log_context(run_key: str) -> Generator[None]:
    """Set the calibration key included in logs within this context."""
    token = _RUN_KEY.set(run_key)
    try:
        yield
    finally:
        _RUN_KEY.reset(token)
