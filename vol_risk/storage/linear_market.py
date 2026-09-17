"""SQLite persistence for linear-equity calibration artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from vol_risk.models.linear import (
    LinearEquityMarket,
    LinearEquityParams,
    make_raw_disc_curve,
    make_raw_interpolator,
)
from vol_risk.protocols import CalibrationArtifactStore
from vol_risk.storage.codec import (
    canonical_config,
    decode_numpy_fields,
    encode_numpy_fields,
    pickle_bytes,
    unpickle_bytes,
)
from vol_risk.storage.misc import (
    normalize_date,
    normalize_identifier,
    register_calib_spec,
)
from vol_risk.storage.sqlite import initialize_schema

if TYPE_CHECKING:
    from datetime import date, datetime

MODEL_NAME = "linear_equity"
_PARAMS_MAGIC = b"vol-risk.linear-params.numpy.v1\0"


@dataclass(frozen=True)
class LinearModelArtifact:
    """A stored linear-equity calibration reconstructed for use."""

    ticker: str
    calibration_date: date
    calibration_id: str
    algorithm_version: str
    model: LinearEquityMarket
    params: LinearEquityParams
    stats: dict[object, object] | None
    config: dict[str, object]
    update_time: str


def _params_from_mapping(raw: Mapping[str, object]) -> LinearEquityParams:
    """Rebuild stored curve parameters without validating them again."""
    required = {"spot", "tau", "r", "q"}
    missing = required.difference(raw)
    if missing:
        msg = f"Stored linear-model parameters are missing fields: {sorted(missing)}"
        raise ValueError(msg)

    return LinearEquityParams(
        spot=float(cast("float", raw["spot"])),
        tau=np.asarray(raw["tau"]),
        r=np.asarray(raw["r"]),
        q=np.asarray(raw["q"]),
        validate=False,
    )


def _encode_params(params: LinearEquityParams) -> bytes:
    """Validate linear parameters and encode their numeric fields."""
    validated = LinearEquityParams(
        spot=params.spot,
        tau=params.tau,
        r=params.r,
        q=params.q,
        validate=True,
    )
    fields = {
        "spot": np.asarray(validated.spot, dtype=np.float64),
        "tau": np.asarray(validated.tau, dtype=np.float64),
        "r": np.asarray(validated.r, dtype=np.float64),
        "q": np.asarray(validated.q, dtype=np.float64),
    }
    return _PARAMS_MAGIC + encode_numpy_fields(fields)


def _decode_params(value: bytes) -> LinearEquityParams:
    """Decode linear parameters from the stored NumPy format."""
    if not value.startswith(_PARAMS_MAGIC):
        msg = "Stored linear-model parameters do not use the supported NumPy encoding"
        raise ValueError(msg)
    return _params_from_mapping(decode_numpy_fields(value[len(_PARAMS_MAGIC) :]))


def _model_from_params(params: LinearEquityParams) -> LinearEquityMarket:
    """Reconstruct a linear-equity market from validated parameters."""
    return LinearEquityMarket(
        spot=params.spot,
        disc_curve=make_raw_disc_curve(tau=params.tau, r=params.r),
        cont_carry_curve=make_raw_interpolator(tau=params.tau, r=params.q),
    )


class LinearModelStore(CalibrationArtifactStore[LinearEquityParams, LinearModelArtifact]):
    """Persist linear-equity artifacts using a caller-managed SQLite transaction."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def initialize(self) -> None:
        """Create the calibration specification and observation tables."""
        initialize_schema(self.connection)

    def contains(
        self,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
    ) -> bool:
        """Return whether the requested linear-model artifact exists."""
        ticker = normalize_identifier(ticker, "ticker")
        calibration_id = normalize_identifier(calibration_id, "calibration_id")
        date_text = normalize_date(calibration_date).isoformat()
        row = self.connection.execute(
            "SELECT 1 FROM linear_market WHERE ticker = ? AND date = ? AND calibration_id = ?",
            (ticker, date_text, calibration_id),
        ).fetchone()
        return row is not None

    def write(
        self,
        *,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
        algorithm_version: str,
        params: LinearEquityParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
    ) -> None:
        """Persist one new observation and its calibration specification."""
        self._write(
            ticker=ticker,
            calibration_date=calibration_date,
            calibration_id=calibration_id,
            algorithm_version=algorithm_version,
            params=params,
            stats=stats,
            config=config,
            overwrite=False,
        )

    def overwrite(
        self,
        *,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
        algorithm_version: str,
        params: LinearEquityParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
    ) -> None:
        """Replace one observation after verifying its calibration specification."""
        self._write(
            ticker=ticker,
            calibration_date=calibration_date,
            calibration_id=calibration_id,
            algorithm_version=algorithm_version,
            params=params,
            stats=stats,
            config=config,
            overwrite=True,
        )

    def _write(
        self,
        *,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
        algorithm_version: str,
        params: LinearEquityParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
        overwrite: bool,
    ) -> None:
        """Write an observation while maintaining specification consistency."""
        ticker = normalize_identifier(ticker, "ticker")
        calibration_id = normalize_identifier(calibration_id, "calibration_id")
        algorithm_version = normalize_identifier(algorithm_version, "algorithm_version")
        date_text = normalize_date(calibration_date).isoformat()
        if not isinstance(params, LinearEquityParams):
            msg = "params must be a LinearEquityParams instance"
            raise TypeError(msg)
        config_json = canonical_config(config)
        params_blob = sqlite3.Binary(_encode_params(params))
        stats_blob = sqlite3.Binary(pickle_bytes(dict(stats))) if stats is not None else None

        if not overwrite and (
            self.connection.execute(
                "SELECT 1 FROM linear_market WHERE ticker = ? AND date = ? AND calibration_id = ?",
                (ticker, date_text, calibration_id),
            ).fetchone()
            is not None
        ):
            msg = (
                f"Linear model already exists for ticker={ticker!r}, "
                f"date={date_text!r}, calibration_id={calibration_id!r}"
            )
            raise ValueError(msg)

        try:
            register_calib_spec(
                self.connection,
                calibration_id=calibration_id,
                model=MODEL_NAME,
                algorithm_version=algorithm_version,
                config_json=config_json,
            )
            if overwrite:
                self.connection.execute(
                    """
                    INSERT INTO linear_market (
                        ticker, date, calibration_id, params, stats
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, date, calibration_id) DO UPDATE SET
                        params = excluded.params,
                        stats = excluded.stats,
                        update_time = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    """,
                    (ticker, date_text, calibration_id, params_blob, stats_blob),
                )
            else:
                self.connection.execute(
                    """
                    INSERT INTO linear_market (
                        ticker, date, calibration_id, params, stats
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticker, date_text, calibration_id, params_blob, stats_blob),
                )
        except sqlite3.IntegrityError as error:
            msg = (
                f"Linear model already exists for ticker={ticker!r}, "
                f"date={date_text!r}, calibration_id={calibration_id!r}"
            )
            raise ValueError(msg) from error

    def load(
        self,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
    ) -> LinearModelArtifact:
        """Retrieve one artifact and reconstruct its linear-equity model."""
        ticker = normalize_identifier(ticker, "ticker")
        calibration_id = normalize_identifier(calibration_id, "calibration_id")
        normalized_date = normalize_date(calibration_date)
        row = self.connection.execute(
            """
            SELECT calibration_specs.model, linear_market.params, linear_market.stats,
                   calibration_specs.config,
                   calibration_specs.algorithm_version, linear_market.update_time
            FROM linear_market
            JOIN calibration_specs USING (calibration_id)
            WHERE linear_market.ticker = ?
              AND linear_market.date = ?
              AND linear_market.calibration_id = ?
            """,
            (ticker, normalized_date.isoformat(), calibration_id),
        ).fetchone()
        if row is None:
            msg = (
                f"No linear model found for ticker={ticker!r}, date={normalized_date.isoformat()!r}, "
                f"calibration_id={calibration_id!r}"
            )
            raise KeyError(msg)

        stored_model, params_blob, stats_blob, config_json, algorithm_version, update_time = row
        if stored_model != MODEL_NAME:
            msg = f"Stored calibration specification {calibration_id!r} is not a linear-equity model"
            raise ValueError(msg)

        params = _decode_params(bytes(params_blob))
        raw_stats = None if stats_blob is None else unpickle_bytes(bytes(stats_blob), "linear-model statistics")
        raw_config = json.loads(config_json)
        if raw_stats is not None and not isinstance(raw_stats, Mapping):
            msg = "Stored linear-model statistics must be a mapping"
            raise ValueError(msg)
        if not isinstance(raw_config, dict):
            msg = "Stored linear-model configuration must be a JSON object"
            raise TypeError(msg)

        stats = None if raw_stats is None else dict(cast("Mapping[object, object]", raw_stats))
        config = cast("dict[str, object]", raw_config)
        return LinearModelArtifact(
            ticker=ticker,
            calibration_date=normalized_date,
            calibration_id=calibration_id,
            algorithm_version=algorithm_version,
            model=_model_from_params(params),
            params=params,
            stats=stats,
            config=config,
            update_time=update_time,
        )
