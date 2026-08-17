"""SQLite persistence for linear-equity calibration artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import cast

import numpy as np

from vol_risk.dao.dao import DATABASE_PATH, CalibrationArtifactDAO
from vol_risk.models.linear import (
    LinearEquityMarket,
    LinearEquityParams,
    make_raw_disc_curve,
    make_raw_interpolator,
)


@dataclass(frozen=True)
class LinearModelArtifact:
    """A stored linear-equity calibration reconstructed for use."""

    ticker: str
    calibration_date: date
    run_id: str
    model: LinearEquityMarket
    params: LinearEquityParams
    stats: dict[object, object] | None
    config: dict[str, object]
    config_hash: str
    update_time: str


def _normalize_date(value: date | datetime | str) -> date:
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


def _normalize_identifier(value: str, name: str) -> str:
    """Strip and validate a persisted identifier."""
    normalized = value.strip()
    if not normalized:
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)
    return normalized


def _canonical_config(config: Mapping[str, object]) -> tuple[str, str]:
    """Return canonical JSON configuration and its SHA-256 digest."""
    try:
        payload = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        msg = "config must contain only JSON-serializable finite values"
        raise ValueError(msg) from error
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pickle_bytes(value: object) -> bytes:
    """Serialize an internal calibration value for SQLite BLOB storage."""
    return pickle.dumps(value, protocol=5)


def _unpickle_bytes(value: bytes, field_name: str) -> object:
    """Deserialize a value read from a trusted calibration database."""
    try:
        return pickle.loads(value)  # noqa: S301
    except (AttributeError, EOFError, ImportError, IndexError, pickle.UnpicklingError) as error:
        msg = f"Stored linear-model {field_name} are not a valid artifact"
        raise ValueError(msg) from error


def _params_from_mapping(raw: Mapping[str, object]) -> LinearEquityParams:
    """Validate stored curve parameters and rebuild the params dataclass."""
    required = {"spot", "tau", "r", "q"}
    missing = required.difference(raw)
    if missing:
        msg = f"Stored linear-model parameters are missing fields: {sorted(missing)}"
        raise ValueError(msg)

    try:
        spot = float(cast("float", raw["spot"]))
        tau = np.asarray(raw["tau"], dtype=float)
        r = np.asarray(raw["r"], dtype=float)
        q = np.asarray(raw["q"], dtype=float)
    except (TypeError, ValueError) as error:
        msg = "Stored linear-model parameters must be numeric"
        raise ValueError(msg) from error

    if not np.isfinite(spot) or spot <= 0.0:
        msg = f"Stored spot must be finite and positive; received {spot}"
        raise ValueError(msg)
    if tau.ndim != 1 or tau.size == 0 or r.shape != tau.shape or q.shape != tau.shape:
        msg = "Stored tau, rate, and dividend-yield arrays must be equally sized 1-D arrays"
        raise ValueError(msg)
    if not np.isfinite(np.concatenate((tau, r, q))).all():
        msg = "Stored curve inputs must contain only finite values"
        raise ValueError(msg)
    if np.any(tau <= 0.0) or np.any(np.diff(tau) <= 0.0):
        msg = "Stored maturities must be positive and strictly increasing"
        raise ValueError(msg)

    return LinearEquityParams(spot=spot, tau=tau, r=r, q=q)


def _model_from_params(params: LinearEquityParams) -> LinearEquityMarket:
    """Reconstruct a linear-equity market from validated parameters."""
    return LinearEquityMarket(
        spot=params.spot,
        disc_curve=make_raw_disc_curve(tau=params.tau, r=params.r),
        cont_carry_curve=make_raw_interpolator(tau=params.tau, r=params.q),
    )


class LinearModelStore:
    """Persist and retrieve linear-equity calibration artifacts in SQLite."""

    def __init__(self, database_path: Path = DATABASE_PATH, *, timeout_seconds: float = 30.0) -> None:
        self.database_path = Path(database_path)
        self._dao = CalibrationArtifactDAO(
            database_path=self.database_path,
            timeout_seconds=timeout_seconds,
        )

    def initialize(self) -> None:
        """Create or upgrade the calibration artifact schema."""
        self._dao.initialize_schema()

    def contains(
        self,
        ticker: str,
        calibration_date: date | datetime | str,
        run_id: str,
    ) -> bool:
        """Return whether the requested linear-model artifact exists."""
        ticker = _normalize_identifier(ticker, "ticker")
        run_id = _normalize_identifier(run_id, "run_id")
        date_text = _normalize_date(calibration_date).isoformat()
        with self._dao.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM linear_market WHERE ticker = ? AND date = ? AND run_id = ?",
                (ticker, date_text, run_id),
            ).fetchone()
        return row is not None

    def save(
        self,
        *,
        ticker: str,
        calibration_date: date | datetime | str,
        run_id: str,
        params: LinearEquityParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
        overwrite: bool = False,
    ) -> None:
        """Persist one calibration, optionally replacing the same primary key."""
        ticker = _normalize_identifier(ticker, "ticker")
        run_id = _normalize_identifier(run_id, "run_id")
        date_text = _normalize_date(calibration_date).isoformat()
        config_json, config_hash = _canonical_config(config)
        values = (
            ticker,
            date_text,
            run_id,
            sqlite3.Binary(_pickle_bytes(asdict(params))),
            sqlite3.Binary(_pickle_bytes(dict(stats))) if stats is not None else None,
            config_json,
            config_hash,
        )
        insert_sql = """
            INSERT INTO linear_market (
                ticker, date, run_id, params, stats, config, config_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        upsert_sql = (
            insert_sql
            + """
            ON CONFLICT(ticker, date, run_id) DO UPDATE SET
                params = excluded.params,
                stats = excluded.stats,
                config = excluded.config,
                config_hash = excluded.config_hash,
                update_time = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """
        )
        try:
            with self._dao.connect() as connection:
                connection.execute(upsert_sql if overwrite else insert_sql, values)
        except sqlite3.IntegrityError as error:
            msg = f"Linear model already exists for ticker={ticker!r}, date={date_text!r}, run_id={run_id!r}"
            raise ValueError(msg) from error

    def load(
        self,
        ticker: str,
        calibration_date: date | datetime | str,
        run_id: str,
    ) -> LinearModelArtifact:
        """Retrieve one artifact and reconstruct its linear-equity model."""
        ticker = _normalize_identifier(ticker, "ticker")
        run_id = _normalize_identifier(run_id, "run_id")
        normalized_date = _normalize_date(calibration_date)
        with self._dao.connect() as connection:
            row = connection.execute(
                """
                SELECT params, stats, config, config_hash, update_time
                FROM linear_market
                WHERE ticker = ? AND date = ? AND run_id = ?
                """,
                (ticker, normalized_date.isoformat(), run_id),
            ).fetchone()
        if row is None:
            msg = (
                f"No linear model found for ticker={ticker!r}, date={normalized_date.isoformat()!r}, run_id={run_id!r}"
            )
            raise KeyError(msg)

        config_json = cast("str", row["config"])
        config_hash = cast("str", row["config_hash"])
        actual_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        if actual_hash != config_hash:
            msg = "Stored linear-model configuration hash does not match its payload"
            raise ValueError(msg)

        raw_params = _unpickle_bytes(bytes(row["params"]), "parameters")
        raw_stats = None if row["stats"] is None else _unpickle_bytes(bytes(row["stats"]), "statistics")
        raw_config = json.loads(config_json)
        if not isinstance(raw_params, Mapping) or not all(isinstance(key, str) for key in raw_params):
            msg = "Stored linear-model parameters must be a string-keyed mapping"
            raise ValueError(msg)
        if raw_stats is not None and not isinstance(raw_stats, Mapping):
            msg = "Stored linear-model statistics must be a mapping"
            raise ValueError(msg)
        if not isinstance(raw_config, dict):
            msg = "Stored linear-model configuration must be a JSON object"
            raise TypeError(msg)

        params = _params_from_mapping(cast("Mapping[str, object]", raw_params))
        stats = None if raw_stats is None else dict(cast("Mapping[object, object]", raw_stats))
        config = cast("dict[str, object]", raw_config)
        return LinearModelArtifact(
            ticker=ticker,
            calibration_date=normalized_date,
            run_id=run_id,
            model=_model_from_params(params),
            params=params,
            stats=stats,
            config=config,
            config_hash=config_hash,
            update_time=cast("str", row["update_time"]),
        )
