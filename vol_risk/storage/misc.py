"""Shared helpers for SQLite calibration-artifact stores."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime


def normalize_identifier(value: str, name: str) -> str:
    """Strip and validate a persisted identifier."""
    normalized = value.strip()
    if not normalized:
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)
    return normalized


def normalize_date(value: date | datetime | str) -> date:
    """Normalize a supported date value to a calendar date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        msg = f"calibration_date must be an ISO date; received {value!r}"
        raise ValueError(msg) from error


def register_calib_spec(
    connection: sqlite3.Connection,
    *,
    calibration_id: str,
    model_id: str,
    algorithm_version: str,
    config_json: str,
) -> None:
    """Create a calibration specification or verify the existing immutable metadata."""
    row = connection.execute(
        """
        SELECT model_id, algorithm_version, config
        FROM calibration_specs
        WHERE calibration_id = ?
        """,
        (calibration_id,),
    ).fetchone()
    if row is None:
        try:
            connection.execute(
                """
                INSERT INTO calibration_specs (
                    calibration_id, model_id, algorithm_version, config
                ) VALUES (?, ?, ?, ?)
                """,
                (calibration_id, model_id, algorithm_version, config_json),
            )
        except sqlite3.IntegrityError as error:
            msg = "Calibration specification metadata already exists"
            raise ValueError(msg) from error
        return

    stored_model_id, stored_algorithm_version, stored_config = row
    if stored_model_id != model_id or stored_algorithm_version != algorithm_version or stored_config != config_json:
        msg = f"Calibration specification {calibration_id!r} has different metadata"
        raise ValueError(msg)
