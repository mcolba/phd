"""Functional SQLite connections and schema definitions for calibration artifacts."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = _PROJECT_ROOT / "data" / "derived" / "market_data.sqlite"

CREATE_CALIBRATION_SPECS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calibration_specs (
    calibration_id TEXT NOT NULL CHECK (length(trim(calibration_id)) > 0),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    algorithm_version TEXT NOT NULL CHECK (length(trim(algorithm_version)) > 0),
    config TEXT NOT NULL CHECK (json_valid(config)),
    PRIMARY KEY (calibration_id),
    UNIQUE (model_id, algorithm_version, config)
) WITHOUT ROWID
"""

CREATE_LINEAR_MARKET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS linear_market (
    ticker TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    date TEXT NOT NULL CHECK (date(date) = date),
    calibration_id TEXT NOT NULL REFERENCES calibration_specs(calibration_id),
    params BLOB NOT NULL,
    stats BLOB,
    update_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (ticker, date, calibration_id)
) WITHOUT ROWID
"""

CREATE_VOL_MARKET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vol_market (
    ticker TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    date TEXT NOT NULL CHECK (date(date) = date),
    calibration_id TEXT NOT NULL REFERENCES calibration_specs(calibration_id),
    params BLOB NOT NULL,
    stats BLOB,
    update_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (ticker, date, calibration_id)
) WITHOUT ROWID
"""

CREATE_VOL_SMOOTHER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vol_smoother (
    ticker TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    calibration_id TEXT NOT NULL CHECK (length(trim(calibration_id)) > 0),
    params BLOB NOT NULL,
    stats BLOB,
    config TEXT NOT NULL CHECK (json_valid(config)),
    update_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (ticker, calibration_id)
) WITHOUT ROWID
"""

CREATE_LINEAR_MARKET_CALIBRATION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_linear_market_calibration_id
ON linear_market (calibration_id)
"""

CREATE_VOL_MARKET_CALIBRATION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_vol_market_ticker_calibration_date
ON vol_market (ticker, calibration_id, date)
"""

CREATE_VOL_SMOOTHER_LATEST_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_vol_smoother_ticker_update_time
ON vol_smoother (ticker, update_time DESC)
"""

SCHEMA_SQL = (
    CREATE_CALIBRATION_SPECS_TABLE_SQL,
    CREATE_LINEAR_MARKET_TABLE_SQL,
    CREATE_VOL_MARKET_TABLE_SQL,
    CREATE_VOL_SMOOTHER_TABLE_SQL,
    CREATE_LINEAR_MARKET_CALIBRATION_INDEX_SQL,
    CREATE_VOL_MARKET_CALIBRATION_INDEX_SQL,
    CREATE_VOL_SMOOTHER_LATEST_INDEX_SQL,
)


@contextmanager
def open_sqlite_connection(
    database_path: Path | str = DATABASE_PATH,
    *,
    timeout_seconds: float = 30.0,
) -> Generator[sqlite3.Connection, None, None]:
    """Yield an open SQLite connection and close it afterwards.

    Manages only the connection lifecycle. Transactions are caller-managed: wrap
    each unit of work in ``with connection:`` (or explicit BEGIN/commit/rollback).
    """
    if timeout_seconds <= 0.0:
        msg = "timeout_seconds must be positive"
        raise ValueError(msg)

    path = Path(database_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=timeout_seconds)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the calibration specification and observation tables."""
    connection.execute("PRAGMA journal_mode = WAL")
    for statement in SCHEMA_SQL:
        connection.execute(statement)


@contextmanager
def open_initialized_connection(
    database_path: Path | str = DATABASE_PATH,
    *,
    timeout_seconds: float = 30.0,
) -> Generator[sqlite3.Connection]:
    """Open a calibration database connection and ensure its schema exists."""
    with open_sqlite_connection(database_path, timeout_seconds=timeout_seconds) as connection:
        initialize_schema(connection)
        yield connection
