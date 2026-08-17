"""SQLite data access and schema definitions for calibration artifacts."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = _PROJECT_ROOT / "data" / "derived" / "market_data.sqlite"

CREATE_LINEAR_MARKET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS linear_market (
    ticker TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    date TEXT NOT NULL CHECK (date(date) = date),
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    params BLOB NOT NULL,
    stats BLOB,
    config TEXT NOT NULL CHECK (json_valid(config)),
    config_hash TEXT NOT NULL CHECK (length(trim(config_hash)) > 0),
    update_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (ticker, date, run_id)
) WITHOUT ROWID
"""

CREATE_VOL_MARKET_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vol_market (
    ticker TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    date TEXT NOT NULL CHECK (date(date) = date),
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    params BLOB NOT NULL,
    stats BLOB,
    config TEXT NOT NULL CHECK (json_valid(config)),
    config_hash TEXT NOT NULL CHECK (length(trim(config_hash)) > 0),
    update_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (ticker, date, run_id)
) WITHOUT ROWID
"""

CREATE_VOL_SMOOTHER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS vol_smoother (
    ticker TEXT NOT NULL CHECK (length(trim(ticker)) > 0),
    run_id TEXT NOT NULL CHECK (length(trim(run_id)) > 0),
    params BLOB NOT NULL,
    stats BLOB,
    config TEXT NOT NULL CHECK (json_valid(config)),
    config_hash TEXT NOT NULL CHECK (length(trim(config_hash)) > 0),
    update_time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (ticker, run_id)
) WITHOUT ROWID
"""

CREATE_LINEAR_MARKET_RUN_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_linear_market_ticker_run_date
ON linear_market (ticker, run_id, date)
"""

CREATE_VOL_MARKET_RUN_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_vol_market_ticker_run_date
ON vol_market (ticker, run_id, date)
"""

CREATE_VOL_SMOOTHER_LATEST_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_vol_smoother_ticker_update_time
ON vol_smoother (ticker, update_time DESC)
"""

SCHEMA_SQL = (
    CREATE_LINEAR_MARKET_TABLE_SQL,
    CREATE_VOL_MARKET_TABLE_SQL,
    CREATE_VOL_SMOOTHER_TABLE_SQL,
    CREATE_LINEAR_MARKET_RUN_INDEX_SQL,
    CREATE_VOL_MARKET_RUN_INDEX_SQL,
    CREATE_VOL_SMOOTHER_LATEST_INDEX_SQL,
)

_STATS_COLUMN_UPGRADES = (
    ("linear_market", "ALTER TABLE linear_market ADD COLUMN stats BLOB"),
    ("vol_market", "ALTER TABLE vol_market ADD COLUMN stats BLOB"),
    ("vol_smoother", "ALTER TABLE vol_smoother ADD COLUMN stats BLOB"),
)

def _add_missing_stats_columns(connection: sqlite3.Connection) -> None:
    """Upgrade databases created before artifact statistics were stored."""
    for table_name, statement in _STATS_COLUMN_UPGRADES:
        row = connection.execute(
            "SELECT 1 FROM pragma_table_info(?) WHERE name = 'stats'",
            (table_name,),
        ).fetchone()
        if row is None:
            connection.execute(statement)


class CalibrationArtifactDAO:
    """Create and connect to the SQLite calibration-artifact store."""

    def __init__(self, database_path: Path = DATABASE_PATH, *, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0.0:
            msg = "timeout_seconds must be positive"
            raise ValueError(msg)
        self.database_path = Path(database_path)
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection]:
        """Yield a transactional connection and always close it afterwards."""
        connection = sqlite3.connect(self.database_path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize_schema(self) -> None:
        """Create or upgrade the database tables and indexes."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            for statement in SCHEMA_SQL:
                connection.execute(statement)
            _add_missing_stats_columns(connection)
