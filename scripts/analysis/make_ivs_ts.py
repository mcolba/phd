from __future__ import annotations

import datetime as dt
import logging
import shelve
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandera import pandas

from vol_risk.models.linear import LinearEquityMarket, make_raw_disc_curve, make_raw_interpolator
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, _make_smile_fun
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY
from vol_risk.vol_surface.surface import VolSurface

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHELVE_PATH = PROJECT_ROOT / "data" / "derived" / "mixture"
OUTPUT_PATH = PROJECT_ROOT / "data" / "derived" / "mixture" / "spx_ivs_ts.npz"

TICKER = "SPX"
MONEYNESS_TYPE = "delta"
MONEYNESS_GRID = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
TAU_GRID = np.array([30.0, 90.0, 180.0, 365.0, 730.0], dtype=float) / 365.0


@dataclass(frozen=True)
class IvsTimeSeries:
    """Fixed-grid implied-volatility time series and market curves."""

    dates: np.ndarray
    tau_grid: np.ndarray
    moneyness_grid: np.ndarray
    iv: np.ndarray
    disc: np.ndarray
    fwd: np.ndarray

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return pd.DataFrame(
            data=self.iv.reshape(self.dates.size, -1),
            index=self.dates,
            columns=pd.MultiIndex.from_product([self.tau_grid, self.moneyness_grid], names=["tau", "moneyness"]),
        )


def _build_market_and_surface(entry: Mapping[str, object]) -> tuple[LinearEquityMarket, VolSurface]:
    linear_params, ivs_params = entry["params"]
    tau = linear_params["tau"]
    r = linear_params["r"]
    q = linear_params["q"]

    #TODO (Marco): save spot in shelve!
    alpha = [v["coeff"][0] for _, v in entry["stats"][0].items() if not v["excluded"]]
    spot = (alpha / np.exp(-q * tau))[0]

    market = LinearEquityMarket(
        spot=spot,
        disc_curve=make_raw_disc_curve(tau=tau, r=r),
        cont_carry_curve=make_raw_interpolator(tau=tau, r=q),
    )

    smiles = [
        _make_smile_fun(
            params=LogNormMixParams(
                w=np.asarray(v["params"]["w"], dtype=float),
                mu=np.asarray(v["params"]["mu"], dtype=float),
                sigma=np.asarray(v["params"]["sigma"], dtype=float),
            ),
            le=market,
            tau=v["tau"],
        )
        for v in ivs_params.values()
    ]

    taus = np.array([v["tau"] for v in ivs_params.values()], dtype=float)
    return VolSurface(taus=taus, smiles=smiles, linear_model=market)


def make_ivs_from_shelve(
    ticker: str,
    path: Path,
    moneyness_type: str,
    moneyness_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> IvsTimeSeries:
    if moneyness_type not in MONEYNESS_REGISTRY and moneyness_type != "k":
        msg = f"Unsupported moneyness_type={moneyness_type!r}."
        raise ValueError(msg)

    grid_m = np.sort(np.asarray(moneyness_grid, dtype=float))
    grid_t = np.asarray(tau_grid, dtype=float)
    if grid_m.ndim != 1 or grid_m.size == 0 or not np.all(np.isfinite(grid_m)):
        msg = "moneyness_grid must be a non-empty 1-D finite array."
        raise ValueError(msg)
    if grid_t.ndim != 1 or grid_t.size == 0 or np.any(grid_t <= 0.0):
        msg = "tau_grid must be a non-empty 1-D positive array."
        raise ValueError(msg)

    with shelve.open(str(path), flag="r") as db:
        keys = sorted(k for k in db if k.startswith(f"{ticker}_"))
        if not keys:
            msg = f"No entries found for ticker={ticker!r} in shelve={path}."
            raise ValueError(msg)

        iv = np.full((len(keys), grid_t.size, grid_m.size), np.nan, dtype=float)
        disc = np.full((len(keys), grid_t.size), np.nan, dtype=float)
        fwd = np.full((len(keys), grid_t.size), np.nan, dtype=float)
        dates = np.empty(len(keys), dtype="datetime64[D]")

        for i, key in enumerate(keys):
            row = db[key]
            dates[i] = np.datetime64(row["date"])

            surface = _build_market_and_surface(row)
            disc[i] = np.asarray(surface._linear_model.df(grid_t), dtype=float)
            fwd[i] = np.asarray(surface._linear_model.fwd(grid_t), dtype=float)

            if not np.all(np.isfinite(fwd[i])):
                msg = f"Non-finite forward curve for key={key}, date={row['date']}."
                raise ValueError(msg)

            for j, tau in enumerate(grid_t):
                moneyness = MONEYNESS_REGISTRY[moneyness_type](le=surface._linear_model)
                kwargs = {"moneyness": grid_m, "tau": tau, "sigma": surface.atmf_vol(np.atleast_1d(tau))}
                strikes = moneyness.invert(**kwargs)
                iv[i, j] = np.asarray(surface.vol(strikes, np.full_like(strikes, tau, dtype=float)), dtype=float)

    return IvsTimeSeries(dates=dates, tau_grid=grid_t, moneyness_grid=grid_m, iv=iv, disc=disc, fwd=fwd)


if __name__ == "__main__":
    ts = out = make_ivs_from_shelve(
        ticker=TICKER,
        path=SHELVE_PATH,
        moneyness_type=MONEYNESS_TYPE,
        moneyness_grid=MONEYNESS_GRID,
        tau_grid=TAU_GRID,
    )
    df = ts.df
