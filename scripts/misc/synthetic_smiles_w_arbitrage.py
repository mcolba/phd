"""Generates synthetic options data with arbitrage for testing purposes.

This script creates two datasets using the SVI model:
1. A smile with butterfly arbitrage.
2. Two smiles with calendar spread arbitrage.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vol_risk.models.black76 import bsm_price, implied_vol_jackel
from vol_risk.models.linear import LinearEquity
from vol_risk.vol_surface.interpl.ssvi import RawSVIParams, svi_raw

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

plt.style.use("ggplot")


def build_call_slice(params: RawSVIParams, tau: float, strikes: np.ndarray, le: LinearEquity) -> pd.DataFrame:
    """Build a DataFrame slice for a given maturity and SVI parameters."""
    # ensure monotonic strikes
    order = np.argsort(strikes)
    k_sorted = strikes[order]

    svi_total_var = svi_raw(params)
    log_k = np.log(k_sorted / le.fwd(tau))
    w = svi_total_var(log_k)
    iv = np.sqrt(w / tau)
    price = np.array(
        [
            bsm_price(s=le.spot, k=k, t=tau, sigma=vol, r=le.r, q=le.q, is_call=True)
            for k, vol in zip(k_sorted, iv, strict=False)
        ]
    )

    if not (price >= np.maximum(np.exp(-le.q * tau) * le.spot - le.df(tau) * strikes, 0)).all():
        msg = "Negative intrinsic value detected."
        raise ValueError(msg)

    # density = ∂²C/∂K²
    dc_dk = np.gradient(price, k_sorted)
    d2c_dk2 = np.gradient(dc_dk, k_sorted)

    return pd.DataFrame(
        {
            "tau": tau,
            "strike": strikes,
            "implied_vol": iv,
            "total_variance": w,
            "call_price": price,
            "density": d2c_dk2,
        }
    )


def save_dataframe_to_csv(df: pd.DataFrame, lm, path: Path) -> None:
    """Save DataFrame to CSV with custom CBOE-style field names."""
    log.info("Saving DataFrame to %s", path)
    cboe_style_df = pd.DataFrame(
        {
            "underlying_symbol": "NA",
            "tau": df["tau"],
            "strike": df["strike"],
            "option_type": "C",
            "bid_eod": df["call_price"],
            "ask_eod": df["call_price"],
            "underlying_mid_eod": lm.spot,
            "forward_price": lm.fwd(df["tau"]),
            "discount_factor": lm.df(df["tau"]),
        }
    )
    cboe_style_df.to_csv(path, index=False)


def plot_panels(
    axes: Sequence[plt.Axes],
    df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
) -> None:
    """Plot multiple panels with shared x-axis."""
    if len(axes) != len(y_cols):
        msg = "Number of axes and y_cols must match."
        raise ValueError(msg)
    if "tau" not in df.columns:
        msg = "DataFrame must contain 'tau' column for grouping."
        raise ValueError(msg)

    def prettify(name: str) -> str:
        return name.replace("_", " ").capitalize()

    for ax, y_col in zip(axes, y_cols, strict=False):
        for tau_val, slice_df in df.groupby("tau"):
            ax.plot(slice_df[x_col], slice_df[y_col], marker="o", label=f"T={tau_val}")
        ax.legend()
        ax.set_xlabel(x_col)
        ax.set_ylabel(prettify(y_col))


if __name__ == "__main__":
    spot = 100
    r = 0.02
    q = 0.05
    strikes = spot * np.exp(np.linspace(-0.3, 0.3, 20))
    lm = LinearEquity(spot=spot, r=r, q=q)

    # ------------------------------------------------------------------------------------------------------------------
    # Smile with butterfly arbitrage
    # ------------------------------------------------------------------------------------------------------------------
    log.info("Building SVI smile with butterfly arbitrage ...")
    tau = 0.3
    svi_params = RawSVIParams(a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.4)
    df01 = build_call_slice(svi_params, tau, strikes, lm).sort_values("strike").reset_index(drop=True)

    # Introduce strike arbitrage
    bump = 0.5
    idx = df01.index[len(df01) // 2]

    new_prices = df01["call_price"].to_numpy().copy()
    new_prices[idx] += bump

    new_iv = implied_vol_jackel(
        price=new_prices[idx],
        f=lm.fwd(tau),
        k=df01.loc[idx, "strike"],
        t=tau,
        df=lm.df(tau),
        theta=1.0,
    )

    dc_dk = np.gradient(new_prices, df01["strike"].values)
    d2c_dk2 = np.gradient(dc_dk, df01["strike"].values)

    df01.loc[idx, "call_price"] = new_prices[idx]
    df01.loc[idx, "implied_vol"] = new_iv
    df01.loc[idx, "total_variance"] = new_iv**2 * tau
    df01.loc[:, "density"] = d2c_dk2

    if not (df01["density"] < 0).any():
        msg = "Butterfly arbitrage not detected: density is non-negative for all strikes."
        raise AssertionError(msg)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle("SVI Smile with Butterfly Arbitrage", fontsize=16)
    plot_panels(axes, df01, "strike", ["implied_vol", "call_price", "density"])
    plt.show(block=False)

    # Save CSV using custom CBOE-style field names
    path = Path(r".\data\raw\butterfly_arbitrage.csv")
    save_dataframe_to_csv(df01, lm, path)

    # ------------------------------------------------------------------------------------------------------------------
    # Smile with calendar spread arbitrage
    # ------------------------------------------------------------------------------------------------------------------
    tau0, tau1 = 0.25, 0.50
    avi_params_0 = RawSVIParams(a=0.004, b=0.11, rho=-0.20, m=0.15, sigma=0.22)
    avi_params_1 = RawSVIParams(a=0.015, b=0.08, rho=-0.50, m=0.00, sigma=0.35)

    df_t0 = build_call_slice(avi_params_0, tau0, strikes, lm)
    df_t1 = build_call_slice(avi_params_1, tau1, strikes, lm)
    df02 = pd.concat([df_t0, df_t1], ignore_index=True).sort_values(["tau", "strike"])

    # Check for calendar arbitrage: C(T1) < C(T0) for some strikes
    pivot = df02.pivot_table(index="strike", columns="tau", values="call_price")
    if not (pivot[tau1] <= pivot[tau0]).any():
        msg = f"Calendar arbitrage not detected: C(T={tau1}) < C(T={tau0}) for some strikes."
        raise AssertionError(msg)

    # --- One figure, 3 stacked subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    fig.suptitle("SVI Smile with Calendar Arbitrage", fontsize=16)
    plot_panels(axes, df02, x_col="strike", y_cols=["total_variance", "call_price", "density"])
    plt.show(block=False)

    # Save CSV using custom CBOE-style field names
    path = Path(r".\data\raw\calendar_arbitrage.csv")
    save_dataframe_to_csv(df02, lm, path)
