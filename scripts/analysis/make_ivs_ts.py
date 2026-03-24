"""Extract calibrated IVS time series from shelved mixture models → tabular CSV.

Pipeline: extract from shelve → forward-fill → clean IV outliers → write CSV.
Output columns: anchor, type (IVS|FWD|DISC), strike, tau, value.
"""

from __future__ import annotations

import logging
import shelve
from pathlib import Path

import numpy as np
import pandas as pd

from vol_risk.models.linear import LinearEquityMarket, make_raw_disc_curve, make_raw_interpolator
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, _make_smile_fun
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY
from vol_risk.vol_surface.surface import VolSurface

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHELVE_PATH = PROJECT_ROOT / "data" / "derived" / "mixture"
OUTPUT_PATH = SHELVE_PATH.with_name("ivs_ts.csv")

TICKERS = ("SPX",)
MONEYNESS_TYPE = "delta"
MONEYNESS_GRID = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
TAU_GRID = np.array([30.0, 90.0, 180.0, 365.0, 730.0]) / 365.0
OUTLIER_JUMP_STD = 2.0
OUTLIER_REVERSION_STD = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_market_and_surface(entry: dict) -> tuple[LinearEquityMarket, VolSurface]:
    """Reconstruct market model and vol surface from a shelved calibration entry."""
    linear_params, ivs_params = entry["params"]
    tau, r, q = linear_params["tau"], linear_params["r"], linear_params["q"]

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
    return market, VolSurface(taus=taus, smiles=smiles, linear_model=market)


def forward_fill_2d(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 0 of a 2-D array."""
    return pd.DataFrame(arr).ffill().to_numpy(dtype=float)


def remove_transient_outliers(
    series: np.ndarray,
    jump_std: float = OUTLIER_JUMP_STD,
    reversion_std: float = OUTLIER_REVERSION_STD,
) -> tuple[np.ndarray, int]:
    """NaN-out isolated one-day spikes in a 1-D series (spike + immediate reversion)."""
    cleaned = np.asarray(series, dtype=float).copy()
    if cleaned.size < 3:
        return cleaned, 0

    diffs = np.diff(cleaned)
    diffs = diffs[np.isfinite(diffs)]
    diff_std = float(np.std(diffs))
    if not np.isfinite(diff_std) or diff_std <= 0.0:
        return cleaned, 0

    jump_thr = jump_std * diff_std
    rev_thr = reversion_std * diff_std
    mask = np.zeros(cleaned.size, dtype=bool)

    for i in range(1, cleaned.size - 1):
        left, cur, right = cleaned[i - 1], cleaned[i], cleaned[i + 1]
        if not np.isfinite([left, cur, right]).all():
            continue
        jl, jr = cur - left, cur - right
        if abs(jl) > jump_thr and abs(jr) > jump_thr and abs(right - left) <= rev_thr and np.sign(jl) == np.sign(jr):
            mask[i] = True

    cleaned[mask] = np.nan
    return cleaned, int(mask.sum())


# ── Extract ──────────────────────────────────────────────────────────────────


def extract_ticker(
    ticker: str,
    path: Path,
    moneyness_type: str,
    moneyness_grid: np.ndarray,
    tau_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read shelve entries for one ticker → (dates, iv, disc, fwd) arrays.

    Returns:
        dates (N,), iv (N, T, M), disc (N, T), fwd (N, T).
    """
    with shelve.open(str(path), flag="r") as db:  # noqa: S301
        keys = sorted(k for k in db if k.startswith(f"{ticker}_"))
        if not keys:
            raise ValueError(f"No entries for {ticker!r} in {path}")

        n = len(keys)
        iv = np.full((n, tau_grid.size, moneyness_grid.size), np.nan)
        disc = np.full((n, tau_grid.size), np.nan)
        fwd = np.full((n, tau_grid.size), np.nan)
        dates = np.empty(n, dtype="datetime64[D]")

        for i, key in enumerate(keys):
            row = db[key]
            dates[i] = np.datetime64(row["date"])
            market, surface = _build_market_and_surface(row)
            disc[i] = np.asarray(market.df(tau_grid), dtype=float)
            fwd[i] = np.asarray(market.fwd(tau_grid), dtype=float)

            moneyness_model = MONEYNESS_REGISTRY[moneyness_type](le=market)
            for j, tau in enumerate(tau_grid):
                strikes = moneyness_model.invert(
                    moneyness=moneyness_grid,
                    tau=tau,
                    sigma=surface.atmf_vol(np.atleast_1d(tau)),
                )
                iv[i, j] = np.asarray(
                    surface.vol(strikes, np.full_like(strikes, tau)),
                    dtype=float,
                )

    log.info("%s: extracted %d dates", ticker, n)
    return dates, iv, disc, fwd


# ── Transform ────────────────────────────────────────────────────────────────


def clean_iv(
    iv: np.ndarray,
    jump_std: float = OUTLIER_JUMP_STD,
    reversion_std: float = OUTLIER_REVERSION_STD,
) -> np.ndarray:
    """Forward-fill, remove transient outliers, forward-fill again on (N, T, M) IV array."""
    n, n_tau, n_m = iv.shape
    flat = forward_fill_2d(iv.reshape(n, -1))

    total = 0
    for col in range(flat.shape[1]):
        flat[:, col], count = remove_transient_outliers(
            flat[:, col],
            jump_std=jump_std,
            reversion_std=reversion_std,
        )
        total += count

    if total > 0:
        log.info("Removed %d transient IV outliers", total)
        flat = forward_fill_2d(flat)

    return flat.reshape(n, n_tau, n_m)


# ── Format ───────────────────────────────────────────────────────────────────


def to_long_frame(
    dates: np.ndarray,
    iv: np.ndarray,
    disc: np.ndarray,
    fwd: np.ndarray,
    tau_grid: np.ndarray,
    moneyness_grid: np.ndarray,
) -> pd.DataFrame:
    """Stack arrays into long-format DataFrame: anchor, type, strike, tau, value."""
    n = dates.size

    # IVS rows: (N, T, M) → flat
    d_ix, t_ix, m_ix = np.meshgrid(
        np.arange(n),
        np.arange(tau_grid.size),
        np.arange(moneyness_grid.size),
        indexing="ij",
    )
    ivs = pd.DataFrame(
        {
            "anchor": dates[d_ix.ravel()],
            "type": "IVS",
            "strike": moneyness_grid[m_ix.ravel()],
            "tau": tau_grid[t_ix.ravel()],
            "value": iv.ravel(),
        }
    )

    # DISC and FWD rows: (N, T) → flat
    d_ix2, t_ix2 = np.meshgrid(np.arange(n), np.arange(tau_grid.size), indexing="ij")
    curve_anchor = dates[d_ix2.ravel()]
    curve_tau = tau_grid[t_ix2.ravel()]

    disc_df = pd.DataFrame(
        {
            "anchor": curve_anchor,
            "type": "DISC",
            "strike": np.nan,
            "tau": curve_tau,
            "value": disc.ravel(),
        }
    )
    fwd_df = pd.DataFrame(
        {
            "anchor": curve_anchor,
            "type": "FWD",
            "strike": np.nan,
            "tau": curve_tau,
            "value": fwd.ravel(),
        }
    )

    return (
        pd.concat([ivs, disc_df, fwd_df], ignore_index=True)
        .sort_values(["anchor", "type", "tau", "strike"], na_position="first")
        .reset_index(drop=True)
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────


def main() -> pd.DataFrame:
    frames = []
    for ticker in TICKERS:
        dates, iv, disc, fwd = extract_ticker(
            ticker=ticker,
            path=SHELVE_PATH,
            moneyness_type=MONEYNESS_TYPE,
            moneyness_grid=MONEYNESS_GRID,
            tau_grid=TAU_GRID,
        )
        iv = clean_iv(iv)
        disc = forward_fill_2d(disc)
        fwd = forward_fill_2d(fwd)
        frames.append(to_long_frame(dates, iv, disc, fwd, TAU_GRID, MONEYNESS_GRID))

    df = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Wrote %d rows to %s", len(df), OUTPUT_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
