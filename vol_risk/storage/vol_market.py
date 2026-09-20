"""SQLite persistence for log-normal mixture volatility-surface artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

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
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, LogNormMixSurfaceParams, _make_smile_fun
from vol_risk.vol_surface.surface import VolSurface

if TYPE_CHECKING:
    from datetime import date, datetime

    from vol_risk.models.linear import LinearEquityMarket

_PARAMS_MAGIC = b"vol-risk.mixture-params.numpy.v1\0"


@dataclass(frozen=True)
class VolMarketArtifact:
    """A stored mixture-surface calibration reconstructed for use."""

    ticker: str
    calibration_date: date
    calibration_id: str
    model_id: str
    algorithm_version: str
    params: LogNormMixSurfaceParams
    stats: dict[object, object] | None
    config: dict[str, object]
    update_time: str

    def build_surface(self, le: LinearEquityMarket) -> VolSurface:
        """Reconstruct the mixture volatility surface against a linear-equity market."""
        taus = np.asarray(self.params.taus, dtype=float)
        smiles = [
            _make_smile_fun(params=slice_params, le=le, tau=float(tau))
            for tau, slice_params in zip(taus, self.params.slices, strict=True)
        ]
        return VolSurface(taus=taus, smiles=smiles, linear_model=le)


def _encode_params(params: LogNormMixSurfaceParams) -> bytes:
    """Validate mixture-surface parameters and encode their numeric fields."""
    taus = np.asarray(params.taus, dtype=np.float64)
    slices = tuple(params.slices)
    if taus.ndim != 1 or taus.size == 0:
        msg = "Surface maturities must be a non-empty 1-D array"
        raise ValueError(msg)
    if taus.size != len(slices):
        msg = "Number of maturities must match the number of mixture slices"
        raise ValueError(msg)
    if not np.isfinite(taus).all() or np.any(np.diff(taus) <= 0.0):
        msg = "Surface maturities must be finite and strictly increasing"
        raise ValueError(msg)

    component_counts = np.array([np.asarray(smile.w).size for smile in slices], dtype=np.int64)
    fields = {
        "taus": taus,
        "component_counts": component_counts,
        "w": np.concatenate([np.asarray(smile.w, dtype=np.float64) for smile in slices]),
        "fwd_scale": np.concatenate([np.asarray(smile.fwd_scale, dtype=np.float64) for smile in slices]),
        "sigma": np.concatenate([np.asarray(smile.sigma, dtype=np.float64) for smile in slices]),
    }
    return _PARAMS_MAGIC + encode_numpy_fields(fields)


def _params_from_fields(fields: Mapping[str, np.ndarray]) -> LogNormMixSurfaceParams:
    """Rebuild mixture-surface parameters from decoded NumPy fields."""
    required = {"taus", "component_counts", "w", "fwd_scale", "sigma"}
    missing = required.difference(fields)
    if missing:
        msg = f"Stored mixture-surface parameters are missing fields: {sorted(missing)}"
        raise ValueError(msg)

    taus = np.asarray(fields["taus"], dtype=float)
    counts = np.asarray(fields["component_counts"]).astype(int)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    w = np.asarray(fields["w"], dtype=float)
    fwd_scale = np.asarray(fields["fwd_scale"], dtype=float)
    sigma = np.asarray(fields["sigma"], dtype=float)
    slices = tuple(
        LogNormMixParams(
            w=w[offsets[index] : offsets[index + 1]],
            fwd_scale=fwd_scale[offsets[index] : offsets[index + 1]],
            sigma=sigma[offsets[index] : offsets[index + 1]],
        )
        for index in range(counts.size)
    )
    return LogNormMixSurfaceParams(taus=taus, slices=slices)


def _decode_params(value: bytes) -> LogNormMixSurfaceParams:
    """Decode mixture-surface parameters from the stored NumPy format."""
    if not value.startswith(_PARAMS_MAGIC):
        msg = "Stored mixture-surface parameters do not use the supported NumPy encoding"
        raise ValueError(msg)
    return _params_from_fields(decode_numpy_fields(value[len(_PARAMS_MAGIC) :]))


class VolMarketStore(CalibrationArtifactStore[LogNormMixSurfaceParams, VolMarketArtifact]):
    """Persist mixture-surface artifacts using a caller-managed SQLite transaction."""

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
        """Return whether the requested vol-market artifact exists."""
        ticker = normalize_identifier(ticker, "ticker")
        calibration_id = normalize_identifier(calibration_id, "calibration_id")
        date_text = normalize_date(calibration_date).isoformat()
        row = self.connection.execute(
            "SELECT 1 FROM vol_market WHERE ticker = ? AND date = ? AND calibration_id = ?",
            (ticker, date_text, calibration_id),
        ).fetchone()
        return row is not None

    def write(
        self,
        *,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
        model_id: str,
        algorithm_version: str,
        params: LogNormMixSurfaceParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
    ) -> None:
        """Persist one new observation and its calibration specification."""
        self._write(
            ticker=ticker,
            calibration_date=calibration_date,
            calibration_id=calibration_id,
            model_id=model_id,
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
        model_id: str,
        algorithm_version: str,
        params: LogNormMixSurfaceParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
    ) -> None:
        """Replace one observation after verifying its calibration specification."""
        self._write(
            ticker=ticker,
            calibration_date=calibration_date,
            calibration_id=calibration_id,
            model_id=model_id,
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
        model_id: str,
        algorithm_version: str,
        params: LogNormMixSurfaceParams,
        stats: Mapping[object, object] | None,
        config: Mapping[str, object],
        overwrite: bool,
    ) -> None:
        """Write an observation while maintaining specification consistency."""
        ticker = normalize_identifier(ticker, "ticker")
        calibration_id = normalize_identifier(calibration_id, "calibration_id")
        model_id = normalize_identifier(model_id, "model_id")
        algorithm_version = normalize_identifier(algorithm_version, "algorithm_version")
        date_text = normalize_date(calibration_date).isoformat()
        if not isinstance(params, LogNormMixSurfaceParams):
            msg = "params must be a MixtureSurfaceParams instance"
            raise TypeError(msg)
        config_json = canonical_config(config)
        params_blob = sqlite3.Binary(_encode_params(params))
        stats_blob = sqlite3.Binary(pickle_bytes(dict(stats))) if stats is not None else None

        if not overwrite and (
            self.connection.execute(
                "SELECT 1 FROM vol_market WHERE ticker = ? AND date = ? AND calibration_id = ?",
                (ticker, date_text, calibration_id),
            ).fetchone()
            is not None
        ):
            msg = (
                f"Vol market already exists for ticker={ticker!r}, "
                f"date={date_text!r}, calibration_id={calibration_id!r}"
            )
            raise ValueError(msg)

        try:
            register_calib_spec(
                self.connection,
                calibration_id=calibration_id,
                model_id=model_id,
                algorithm_version=algorithm_version,
                config_json=config_json,
            )
            if overwrite:
                self.connection.execute(
                    """
                    INSERT INTO vol_market (
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
                    INSERT INTO vol_market (
                        ticker, date, calibration_id, params, stats
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticker, date_text, calibration_id, params_blob, stats_blob),
                )
        except sqlite3.IntegrityError as error:
            msg = (
                f"Vol market already exists for ticker={ticker!r}, "
                f"date={date_text!r}, calibration_id={calibration_id!r}"
            )
            raise ValueError(msg) from error

    def load(
        self,
        ticker: str,
        calibration_date: date | datetime | str,
        calibration_id: str,
    ) -> VolMarketArtifact:
        """Retrieve one mixture-surface artifact and its calibration metadata."""
        ticker = normalize_identifier(ticker, "ticker")
        calibration_id = normalize_identifier(calibration_id, "calibration_id")
        normalized_date = normalize_date(calibration_date)
        row = self.connection.execute(
            """
            SELECT calibration_specs.model_id, vol_market.params, vol_market.stats,
                   calibration_specs.config, calibration_specs.algorithm_version,
                   vol_market.update_time
            FROM vol_market
            JOIN calibration_specs USING (calibration_id)
            WHERE vol_market.ticker = ?
              AND vol_market.date = ?
              AND vol_market.calibration_id = ?
            """,
            (ticker, normalized_date.isoformat(), calibration_id),
        ).fetchone()
        if row is None:
            msg = (
                f"No vol market found for ticker={ticker!r}, date={normalized_date.isoformat()!r}, "
                f"calibration_id={calibration_id!r}"
            )
            raise KeyError(msg)

        stored_model_id, params_blob, stats_blob, config_json, algorithm_version, update_time = row

        params = _decode_params(bytes(params_blob))
        raw_stats = None if stats_blob is None else unpickle_bytes(bytes(stats_blob), "vol-market statistics")
        raw_config = json.loads(config_json)
        if raw_stats is not None and not isinstance(raw_stats, Mapping):
            msg = "Stored vol-market statistics must be a mapping"
            raise ValueError(msg)
        if not isinstance(raw_config, dict):
            msg = "Stored vol-market configuration must be a JSON object"
            raise TypeError(msg)

        stats = None if raw_stats is None else dict(cast("Mapping[object, object]", raw_stats))
        config = cast("dict[str, object]", raw_config)
        return VolMarketArtifact(
            ticker=ticker,
            calibration_date=normalized_date,
            calibration_id=calibration_id,
            model_id=stored_model_id,
            algorithm_version=algorithm_version,
            params=params,
            stats=stats,
            config=config,
            update_time=update_time,
        )
