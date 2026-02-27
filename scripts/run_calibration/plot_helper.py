from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vol_risk.calibration.data.option_chain import OptionChain
from vol_risk.models.black76 import implied_vol_jackel
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY, Moneyness
from vol_risk.vol_surface.surface import VolSurface

AXIS_LABELS = {
    "k": "Strike (K)",
    "kf": "Forward moneyness K/F",
    "lkf": "log-forward moneyness log(K/F)",
    "slkf": "std log-forward moneyness log(K/F) / (sqrt(tau) * sigma)",
    "delta": "Call delta",
}

AXIS_LIMITS = {
    "kf": (0.2, 2),
    "lkf": (-0.5, 0.5),
    "slkf": (-5, 3),
    "delta": (0.01, 0.99),
}


def _instantiate_moneyness_models(le: LinearEquityMarket) -> dict[str, Moneyness]:
    return {name: MONEYNESS_REGISTRY[name](le=le) for name in MONEYNESS_REGISTRY.keys()}


def _select_atm_sigma(k_slice: np.ndarray, fwd_value: float, iv_slice: np.ndarray) -> float:
    idx = int(np.nanargmin(np.abs(k_slice - fwd_value)))
    sigma = float(iv_slice[idx]) if np.isfinite(iv_slice[idx]) else float(np.nanmean(iv_slice))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanmean(iv_slice))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.2
    return float(np.clip(sigma, 0.01, 1.0))


def _compute_moneyness(
    coord: str,
    strikes: np.ndarray,
    tau_vec: np.ndarray,
    le: LinearEquityMarket,
    sigma: float | None = None,
) -> np.ndarray:
    if coord == "k":
        return strikes

    model = MONEYNESS_REGISTRY.get(coord)(le)

    if coord == "lkf" or coord == "kf":
        return model.value(strike=strikes, tau=tau_vec)
    if coord == "slkf" or coord == "delta":
        if sigma is None:
            raise ValueError(f"Sigma must be provided for coordinate '{coord}'")
        values = model.value(strike=strikes, tau=tau_vec, sigma=sigma)
    else:
        msg = f"Unsupported moneyness coordinate: {coord}"
        raise ValueError(msg)

    return values


def make_rnd_function(params: LogNormMixParams, tau: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return the risk-neutral density of log-returns for a log-normal mixture.

    The construction follows the usual representation

        S_T = F_T e^X,

    with

        p(X) = sum_k w_k * N( (mu_k - 0.5 * sigma_k^2) * tau, sigma_k^2 * tau ).
    """
    if tau <= 0.0:
        msg = "tau must be positive to define a density for X."
        raise ValueError(msg)

    w = np.asarray(params.w, dtype=float)
    mu = np.asarray(params.mu, dtype=float)
    sigma = np.asarray(params.sigma, dtype=float)

    if not (w.shape == mu.shape == sigma.shape):
        msg = "Parameters 'w', 'mu', and 'sigma' must have the same shape."
        raise ValueError(msg)

    var = sigma**2 * tau
    mean = (mu - 0.5 * sigma**2) * tau

    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    def rnd(x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)[..., None]
        diff = x_arr - mean
        comp_pdf = inv_sqrt_2pi / np.sqrt(var) * np.exp(-0.5 * (diff**2) / var)
        dens = np.sum(w * comp_pdf, axis=-1)
        if np.isscalar(x):
            return float(dens)
        return dens

    return rnd


def plot_mixture_smile(
    ax: plt.Axes,
    chain_slice: OptionChain,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    coord: str = "lkf",
) -> None:
    """Plot market IVs and mixture surface for a single expiry on ax.

    This is a thin wrapper around make_iv_plt_data + plot_iv_slice so that
    the smile styling matches the other slice plots used elsewhere.
    """
    df_plt = make_iv_plt_data(
        sl_raw=chain_slice,
        sl_calib=chain_slice,
        lin_mkt=lin_mkt,
        coord=coord,
    )

    spot = chain_slice.spot
    expiry = chain_slice.df["expiry"].iloc[0]

    plot_iv_slice(
        ax=ax,
        df_plt=df_plt,
        lin_mkt=lin_mkt,
        surface=surface,
        sl=chain_slice,
        coord=coord,
        spot=spot,
        idx=0,
        expiry=expiry,
    )


def plot_mixture_density(
    ax: plt.Axes,
    rnd: Callable[[np.ndarray], np.ndarray],
    x_min: float = -1.0,
    x_max: float = 1.0,
    n: int = 400,
) -> None:
    """Plot the risk-neutral density of log-returns on ax."""
    x = np.linspace(x_min, x_max, n)
    y = rnd(x)

    ax.plot(x, y, color="black", linewidth=1.0)
    ax.set_xlabel("log-return X")
    ax.set_ylabel("Risk-neutral density")
    ax.tick_params(labelleft=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


def plot_mixture_combined(
    chain_slice: OptionChain,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    params: LogNormMixParams,
    coord: str = "lkf",
    x_min: float = -1.0,
    x_max: float = 1.0,
    n: int = 400,
):
    """Create a figure with volatility smile and RND side by side."""
    tau_slice = float(chain_slice.tau[0])
    rnd = make_rnd_function(params=params, tau=tau_slice)

    fig, (ax_smile, ax_rnd) = plt.subplots(1, 2, figsize=(10, 4))
    plot_mixture_smile(ax_smile, chain_slice, lin_mkt, surface, coord=coord)

    # Axis labels for volatility smile
    x_label = AXIS_LABELS.get(coord, "Moneyness")
    ax_smile.set_xlabel(x_label)
    ax_smile.set_ylabel("Implied volatility")

    title_text = ax_smile.get_title()
    if title_text:
        ax_smile.set_title("")
        fig.suptitle(title_text, y=0.95)

    ax_smile.legend()
    plot_mixture_density(ax_rnd, rnd, x_min=x_min, x_max=x_max, n=n)

    fig.tight_layout()
    return fig, (ax_smile, ax_rnd)


def make_iv_plt_data(
    sl_raw: OptionChain,
    sl_calib: OptionChain,
    lin_mkt: LinearEquityMarket,
    coord: str = "lkf",
):
    """Build a DataFrame with IVs and usage flags for one expiry slice.

    Returns a DataFrame with columns:
        iv_mid, iv_bid, iv_ask, is_otm, option_type, is_used.
    """
    tau_slice = float(sl_raw.tau[0])
    k_slice = sl_raw.k
    opt_type_slice = sl_raw.option_type

    df_slice = float(lin_mkt.df(tau_slice))
    fwd_slice = float(lin_mkt.fwd(tau_slice))

    iv_mid = np.empty_like(k_slice, dtype=float)
    iv_bid = np.full_like(k_slice, np.nan, dtype=float)
    iv_ask = np.full_like(k_slice, np.nan, dtype=float)

    bid_arr = sl_raw.bid
    ask_arr = sl_raw.ask

    for i, (mid, bid, ask, k, opt_type) in enumerate(zip(sl_raw.mid, bid_arr, ask_arr, k_slice, opt_type_slice)):
        iv_mid[i] = implied_vol_jackel(
            price=float(mid),
            f=fwd_slice,
            k=float(k),
            t=tau_slice,
            df=df_slice,
            is_call=opt_type == "C",
        )

        if bid > 0:
            try:
                iv_bid[i] = implied_vol_jackel(
                    price=float(bid),
                    f=fwd_slice,
                    k=float(k),
                    t=tau_slice,
                    df=df_slice,
                    is_call=opt_type == "C",
                )
            except Exception:
                iv_bid[i] = np.nan

        if ask > 0:
            try:
                iv_ask[i] = implied_vol_jackel(
                    price=float(ask),
                    f=fwd_slice,
                    k=float(k),
                    t=tau_slice,
                    df=df_slice,
                    is_call=opt_type == "C",
                )
            except Exception:
                iv_ask[i] = np.nan

    # OTM flag under forward measure
    otm_call = (opt_type_slice == "C") & (k_slice >= fwd_slice)
    otm_put = (opt_type_slice == "P") & (k_slice < fwd_slice)
    is_otm = otm_call | otm_put

    df1 = sl_raw.df[["expiry", "strike"]]
    df2 = sl_calib.df[["expiry", "strike"]].drop_duplicates()
    merged = pd.merge(df1, df2, how="left", on=["expiry", "strike"], indicator=True)
    is_used = (merged["_merge"] == "both").to_numpy()

    atm_sigma = _select_atm_sigma(k_slice=k_slice, fwd_value=fwd_slice, iv_slice=iv_mid)

    moneyness_vals = _compute_moneyness(
        coord=coord,
        strikes=k_slice,
        tau_vec=tau_slice,
        le=lin_mkt,
        sigma=atm_sigma,
    )

    out = pd.DataFrame(
        {
            "strike": k_slice,
            "moneyness": moneyness_vals,
            "iv_mid": iv_mid,
            "iv_bid": iv_bid,
            "iv_ask": iv_ask,
            "is_otm": is_otm,
            "option_type": opt_type_slice,
            "is_used": is_used,
        }
    )

    return out


def plot_iv_slice(
    ax: plt.Axes,
    df_plt: pd.DataFrame,
    lin_mkt: LinearEquityMarket,
    surface,
    sl: OptionChain,
    coord: str,
    spot: float,
    idx: int,
    expiry,
) -> None:
    """Plot market IV points with bid/ask brackets and mixture surface for one slice."""
    colour = [
        (~df_plt["is_otm"], "gray", 0.6, "ITM"),
        (df_plt["is_otm"] & ~df_plt["is_used"], "green", 0.8, "OTM - Not Used"),
        (df_plt["is_otm"] & df_plt["is_used"], "blue", 0.8, "OTM - Used"),
    ]

    # Plot bid/ask brackets and mid IVs by category
    for mask, col, alpha, label in colour:
        if np.any(mask):
            valid_bracket = np.isfinite(df_plt.iv_bid) & np.isfinite(df_plt.iv_ask)
            ax.vlines(
                df_plt.moneyness[valid_bracket & mask],
                df_plt.iv_bid[valid_bracket & mask],
                df_plt.iv_ask[valid_bracket & mask],
                color=col,
                linewidth=0.8,
                alpha=alpha,
            )

            ax.scatter(
                df_plt.moneyness[mask],
                df_plt.iv_mid[mask],
                c=col,
                s=3,
                label=label if idx == 0 else None,
            )

    # Mixture surface on same axes
    k_grid = np.linspace(0.2 * spot, 2 * spot, 100)
    t_grid = np.full_like(k_grid, sl.slice_tau, dtype=float)
    iv_model = surface.vol(k_grid, t_grid)

    atm_sigma = _select_atm_sigma(k_slice=k_grid, fwd_value=lin_mkt.fwd(sl.slice_tau), iv_slice=iv_model)
    m_grid = _compute_moneyness(
        coord=coord,
        strikes=k_grid,
        tau_vec=sl.slice_tau,
        le=lin_mkt,
        sigma=atm_sigma,
    )

    ax.plot(
        m_grid,
        iv_model,
        color="red",
        linewidth=1.0,
        label="Mixture surface" if idx == 0 else None,
    )

    if coord in AXIS_LIMITS:
        ax.set_xlim(*AXIS_LIMITS[coord])

    ax.set_ylim(0.0, 0.6)
    ax.set_title(f"T={expiry.date()} (tau={sl.slice_tau:.2f})", fontsize=8)
