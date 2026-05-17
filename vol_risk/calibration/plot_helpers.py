from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.markers import CARETDOWN, CARETUP

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vol_risk.calibration.option_chain import (
        Callable,
        LinearEquityMarket,
        LogNormMixParams,
        OptionChain,
        VolSurface,
    )

from vol_risk.models.black76 import implied_black_vol
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY, Moneyness

AXIS_LABELS = {
    "k": "$K$",
    "kf": "$K/F$",
    "lkf": "$\\ln(K/F)$",
    "lkft": "$\\ln(K/F) / \\sqrt{\\tau}$",
    "slkf": "$\\ln(K/F) / (\\sigma_{ATM} \\sqrt{\\tau})$",
    "delta": "\\Delta_{call}",
}

AXIS_LIMITS = {
    "kf": (0.5, 1.5),
    "lkf": (-0.7, 0.7),
    "lkft": (-1, 1),
    "slkf": (-5, 5),
    "delta": (0.01, 0.99),
}


def _instantiate_moneyness_models(le: LinearEquityMarket) -> dict[str, Moneyness]:
    return {name: MONEYNESS_REGISTRY[name](le=le) for name in MONEYNESS_REGISTRY}


def _select_atm_sigma(k_slice: np.ndarray, fwd_value: float, iv_slice: np.ndarray) -> float:
    idx = int(np.nanargmin(np.abs(k_slice - fwd_value)))
    sigma = float(iv_slice[idx]) if np.isfinite(iv_slice[idx]) else float(np.nanmean(iv_slice))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanmean(iv_slice))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.2
    return float(np.clip(sigma, 0.01, 1.0))


def _invert_moneyness_to_strike(
    coord: str,
    moneyness: np.ndarray,
    tau: float,
    le: LinearEquityMarket,
    surface: VolSurface,
) -> np.ndarray:
    """Invert a moneyness grid to strikes using the registered convention."""
    if coord == "k":
        return np.asarray(moneyness, dtype=float)

    model_cls = MONEYNESS_REGISTRY.get(coord)
    if model_cls is None:
        msg = f"Unsupported moneyness coordinate: {coord}"
        raise ValueError(msg)

    invert_kwargs = {
        "moneyness": np.asarray(moneyness, dtype=float),
        "tau": tau,
    }
    if model_cls.requires_sigma:
        sigma_atm = float(
            surface.vol(
                np.asarray([le.fwd(tau)], dtype=float),
                np.asarray([tau], dtype=float),
            )[0]
        )
        invert_kwargs["sigma"] = sigma_atm

    return np.asarray(model_cls(le).invert(**invert_kwargs), dtype=float)


def make_mixture_rnd_function(params: LogNormMixParams, tau: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return the risk-neutral density of log-returns for a log-normal mixture."""
    if tau <= 0.0:
        msg = "tau must be positive to define a density for X."
        raise ValueError(msg)

    w = np.asarray(params.w, dtype=float)
    mu = np.asarray(params.mu(tau), dtype=float)
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


def _make_iv_plt_data(
    sl_raw: OptionChain,
    sl_calib: OptionChain,
    lin_mkt: LinearEquityMarket,
    coord: str = "lkf",
) -> pd.DataFrame:
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
        iv_mid[i] = float(
            implied_black_vol(
                price=float(mid),
                fwd=fwd_slice,
                strike=float(k),
                tau=tau_slice,
                disc=df_slice,
                is_call=opt_type == "C",
            )
        )

        sign = 1.0 if opt_type == "C" else -1.0
        time_value = max(sign * (lin_mkt.spot - k), 0.0)
        if bid > 0 and bid > time_value:
            try:
                iv_bid[i] = float(
                    implied_black_vol(
                        price=float(bid),
                        fwd=fwd_slice,
                        strike=float(k),
                        tau=tau_slice,
                        disc=df_slice,
                        is_call=opt_type == "C",
                    )
                )
            except Exception:
                iv_bid[i] = np.nan
        elif bid == 0 or bid <= time_value:
            iv_bid[i] = 0.0

        if ask > 0:
            try:
                iv_ask[i] = float(
                    implied_black_vol(
                        price=float(ask),
                        fwd=fwd_slice,
                        strike=float(k),
                        tau=tau_slice,
                        disc=df_slice,
                        is_call=opt_type == "C",
                    )
                )
            except Exception:
                iv_ask[i] = np.nan

    # OTM flag under forward measure
    otm_call = (opt_type_slice == "C") & (k_slice >= fwd_slice)
    otm_put = (opt_type_slice == "P") & (k_slice < fwd_slice)
    is_otm = otm_call | otm_put

    df1 = sl_raw.df[["expiry", "strike"]]
    df2 = sl_calib.df[["expiry", "strike"]].drop_duplicates()
    merged = df1.merge(df2, how="left", on=["expiry", "strike"], indicator=True)
    is_used = (merged["_merge"] == "both").to_numpy()
    synthetic_series = sl_raw.df.get("synthetic", pd.Series(False, index=sl_raw.df.index))
    is_synthetic = synthetic_series.fillna(value=False).to_numpy(dtype=bool)

    moneyness_converter = MONEYNESS_REGISTRY.get(coord)(le=lin_mkt)

    atm_sigma = None
    if moneyness_converter and moneyness_converter.requires_sigma:
        atm_sigma = _select_atm_sigma(k_slice=k_slice, fwd_value=fwd_slice, iv_slice=iv_mid)

    moneyness_vals = moneyness_converter.value(strike=k_slice, tau=tau_slice, sigma=atm_sigma)

    return pd.DataFrame(
        {
            "strike": k_slice,
            "moneyness": moneyness_vals,
            "iv_mid": iv_mid,
            "iv_bid": iv_bid,
            "iv_ask": iv_ask,
            "is_otm": is_otm,
            "option_type": opt_type_slice,
            "is_used": is_used,
            "is_synthetic": is_synthetic,
        }
    )


def _make_bounds_plt_data(
    bounds_df: pd.DataFrame,
    tau: float,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    coord: str = "lkf",
) -> pd.DataFrame:
    """Build plotting data for no-arbitrage upper and lower bounds."""
    if bounds_df.empty:
        return pd.DataFrame(columns=["strike", "moneyness", "iv_ub", "iv_lb"])

    strike = bounds_df["strike"].to_numpy(dtype=float)
    tau_arr = np.full_like(strike, tau, dtype=float)
    disc = float(lin_mkt.df(tau))
    fwd = float(lin_mkt.fwd(tau))

    moneyness_converter = MONEYNESS_REGISTRY.get(coord)(le=lin_mkt)
    atm_sigma = None
    if moneyness_converter.requires_sigma:
        atm_sigma = float(
            surface.vol(
                np.asarray([fwd], dtype=float),
                np.asarray([tau], dtype=float),
            )[0]
        )

    moneyness_vals = moneyness_converter.value(strike=strike, tau=tau_arr, sigma=atm_sigma)

    def _compute_bound_iv(column: str) -> np.ndarray:
        iv = np.full_like(strike, np.nan, dtype=float)
        if column not in bounds_df.columns:
            return iv

        price_norm = bounds_df[column].to_numpy(dtype=float)
        valid = np.isfinite(price_norm) & (price_norm > 0.0)
        valid_idx = np.flatnonzero(valid)
        for current_idx in valid_idx:
            try:
                iv[current_idx] = float(
                    implied_black_vol(
                        price=float(price_norm[current_idx] * fwd * disc),
                        fwd=fwd,
                        strike=float(strike[current_idx]),
                        tau=tau,
                        disc=disc,
                        is_call=True,
                    )
                )
            except ValueError:
                iv[current_idx] = np.nan

        return iv

    return pd.DataFrame(
        {
            "strike": strike,
            "moneyness": moneyness_vals,
            "iv_ub": _compute_bound_iv("price_norm_ub"),
            "iv_lb": _compute_bound_iv("price_norm_lb"),
        }
    )


def _plot_no_arb_bounds(
    ax: plt.Axes,
    bounds_df: pd.DataFrame,
    tau: float,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    coord: str,
    idx: int,
) -> None:
    """Overlay no-arbitrage bounds as faint upper and lower markers."""
    if bounds_df.empty:
        return

    df_bounds = _make_bounds_plt_data(
        bounds_df=bounds_df,
        tau=tau,
        lin_mkt=lin_mkt,
        surface=surface,
        coord=coord,
    )

    for iv_col, marker, colour, label in [
        ("iv_ub", CARETDOWN, "black", "Upper bound"),
        ("iv_lb", CARETUP, "black", "Lower bound"),
    ]:
        mask = np.isfinite(df_bounds["moneyness"]) & np.isfinite(df_bounds[iv_col])
        if coord in AXIS_LIMITS:
            lower_bound, upper_bound = AXIS_LIMITS[coord]
            mask &= df_bounds["moneyness"].between(lower_bound, upper_bound)

        if not np.any(mask):
            continue

        ax.scatter(
            df_bounds.loc[mask, "moneyness"],
            df_bounds.loc[mask, iv_col],
            c=colour,
            s=10,
            marker=marker,
            alpha=0.65,
            # edgecolors=colour,
            linewidths=0.2,
            label=label if idx == 0 else None,
            zorder=3,
        )


_DENSITY_COORDS = {"lkf", "lkft", "kf"}


def plot_mixture_density(
    ax: plt.Axes,
    params: LogNormMixParams,
    tau: float,
    coord: str = "lkf",
    x_min: float | None = None,
    x_max: float | None = None,
    n: int = 400,
    component_alpha: float = 0.35,
) -> None:
    """Plot the risk-neutral density on *ax*.

    Parameters
    ----------
    coord : {"lkf", "lkft", "kf"}
        ``"lkf"`` -- density of log-forward moneyness ln(K/F)
        (mixture of normals).
        ``"lkft"`` -- density of log-forward moneyness standardized
        by sqrt(T), ln(K/F) / sqrt(T).
        ``"kf"``  -- density of forward moneyness K/F
        (mixture of log-normals, includes 1/m Jacobian).
    x_min, x_max : float | None
        Override axis limits.  When *None* the limits are taken from
        ``AXIS_LIMITS[coord]``.
    """
    if coord not in _DENSITY_COORDS:
        msg = f"coord must be one of {sorted(_DENSITY_COORDS)}, got {coord!r}"
        raise ValueError(msg)

    # Resolve axis limits
    lo, hi = AXIS_LIMITS[coord]
    x_min = x_min if x_min is not None else lo
    x_max = x_max if x_max is not None else hi

    w = np.asarray(params.w, dtype=float)
    mu = np.asarray(params.mu(tau), dtype=float)
    sigma = np.asarray(params.sigma, dtype=float)
    var = sigma**2 * tau
    mean = (mu - 0.5 * sigma**2) * tau
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    grid = np.linspace(x_min, x_max, n)

    # ln(K/F) values used for the normal kernel
    if coord == "lkf":
        x = grid
        jacobian = 1.0
    elif coord == "lkft":
        x = grid * np.sqrt(tau)
        jacobian = np.sqrt(tau)
    else:  # kf -- grid values are m = K/F > 0
        x = np.log(grid)
        jacobian = 1.0 / grid

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mixture_density = np.zeros(n, dtype=float)

    for j in range(len(w)):
        comp_pdf = w[j] * inv_sqrt_2pi / np.sqrt(var[j]) * np.exp(-0.5 * (x - mean[j]) ** 2 / var[j]) * jacobian
        mixture_density += comp_pdf
        color = colors[j % len(colors)]
        ax.fill_between(
            grid,
            comp_pdf,
            alpha=component_alpha,
            color=color,
            label=f"comp {j + 1} (w={w[j]:.2f})",
        )
        ax.plot(grid, comp_pdf, color=color, linewidth=0.7, alpha=0.6)

    ax.plot(grid, mixture_density, color="black", linewidth=1.0, label="mixture")

    ax.set_xlabel(AXIS_LABELS.get(coord, "Moneyness"))
    ax.set_ylabel("Risk-neutral density")
    ax.tick_params(labelleft=False)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.5)


def plot_total_variance_curves(
    ax: plt.Axes,
    expiries_sorted: Sequence[object],
    params_ivs: Mapping[object, Mapping[str, object]],
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    coord: str = "kf",
    ca_tolerance: float = 1e-4,
) -> np.ndarray:
    """Plot total variance curves across maturities and return the values."""
    if coord not in ("kf", "lkf"):
        msg = "coord must be either 'kf' or 'lkf' when plotting total variance curves."
        raise ValueError(msg)

    x_min, x_max = AXIS_LIMITS[coord]
    m_common = np.linspace(x_min, x_max, 100)
    model_cls = MONEYNESS_REGISTRY[coord]
    moneyness_model = model_cls(lin_mkt)

    total_var = []
    calendar_label_added = False
    for expiry in expiries_sorted:
        tau = float(params_ivs[expiry]["tau"])
        if model_cls.requires_sigma:
            sigma_atm = float(
                surface.vol(
                    np.asarray([lin_mkt.fwd(tau)], dtype=float),
                    np.asarray([tau], dtype=float),
                )[0]
            )
            k_grid = moneyness_model.invert(moneyness=m_common, tau=tau, sigma=sigma_atm)
        else:
            k_grid = moneyness_model.invert(moneyness=m_common, tau=tau)
        iv_model = surface.vol(k_grid, np.full_like(k_grid, tau, dtype=float))
        total_var_t = iv_model**2 * tau
        ax.plot(m_common, total_var_t, label=f"tau={tau:.2f}")

        if total_var:
            prev_total_var = total_var[-1]
            calendar_mask = (
                np.isfinite(total_var_t) & np.isfinite(prev_total_var) & (total_var_t < prev_total_var - ca_tolerance)
            )
            if np.any(calendar_mask):
                ax.scatter(
                    m_common[calendar_mask],
                    total_var_t[calendar_mask],
                    color="black",
                    marker="x",
                    s=18,
                    linewidths=0.8,
                    label="Calendar arbitrage" if not calendar_label_added else None,
                    zorder=4,
                )
                calendar_label_added = True

        total_var.append(total_var_t)

    ax.set_xlabel(AXIS_LABELS.get(coord, "Moneyness"))
    if coord in AXIS_LIMITS:
        ax.set_xlim(*AXIS_LIMITS[coord])
    ax.set_ylabel("Total Variance")
    ax.legend()
    ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.5)
    return np.asarray(total_var)


def plot_smile_and_mkt_grid(
    chain_raw: OptionChain,
    chain_calib: OptionChain,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    coord: str = "lkf",
    n_col: int = 3,
    title: str | None = None,
    bounds: Mapping[object, pd.DataFrame | None] | None = None,
    dpi: float = 120.0,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot market implied vols against the fitted surface across expiries."""
    expiries = np.unique(chain_raw.expiry)
    n = expiries.size
    n_row = n // n_col + int(n % n_col > 0)
    fig, axes = plt.subplots(
        n_row,
        n_col,
        sharex=True,
        sharey=True,
        figsize=(4 * n_col, 3 * n_row),
        dpi=dpi,
    )
    axes = np.atleast_1d(axes).ravel()

    axis_label = AXIS_LABELS[coord]

    sigma_max = 0.0
    for idx, (expiry, sl) in enumerate(chain_raw):
        ax = axes[idx]
        df_plt = _make_iv_plt_data(sl, chain_calib, lin_mkt, coord)
        plot_iv_slice(ax, df_plt, lin_mkt, surface, sl, coord, chain_raw.spot, idx, expiry)
        bounds_df = None if bounds is None else bounds.get(expiry)
        if bounds_df is not None:
            _plot_no_arb_bounds(
                ax=ax,
                bounds_df=bounds_df,
                tau=float(sl.tau[0]),
                lin_mkt=lin_mkt,
                surface=surface,
                coord=coord,
                idx=idx,
            )
        sigma_max = max(sigma_max, df_plt.loc[df_plt["moneyness"].between(*AXIS_LIMITS[coord]), "iv_mid"].max())

    for j in range(idx + 1, axes.size):
        axes[j].axis("off")

    visible_axes = axes[: min(n, axes.size)]
    for ax in visible_axes:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        ax.set_ylim(0.0, min(sigma_max + 0.1, 1.5))

    for col_idx in range(n_col):
        col_axes = visible_axes[col_idx::n_col]
        if col_axes.size == 0:
            continue
        bottom_ax = col_axes[-1]
        bottom_ax.set_xlabel(axis_label)
        bottom_ax.tick_params(labelbottom=True)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        fig.tight_layout()

    return fig, axes


def plot_mixture_smile_and_rnd(
    chain_calib: OptionChain,
    chain_raw: OptionChain,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    params: LogNormMixParams,
    coord: str = "lkf",
    density_coord: str | None = None,
    n: int = 400,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a figure with volatility smile and RND side by side.

    Parameters
    ----------
    density_coord : str | None
        Coordinate for the density panel (``"lkf"``, ``"lkft"``, or ``"kf"``).
        Defaults to *coord* when it is one of these, otherwise ``"lkf"``.
    """
    assert chain_calib.slice_tau == chain_raw.slice_tau
    tau_slice = float(chain_calib.tau[0])

    if density_coord is None:
        density_coord = coord if coord in _DENSITY_COORDS else "lkf"

    fig, (ax_smile, ax_rnd) = plt.subplots(1, 2, figsize=(10, 4))
    # Prepare DataFrame for plotting
    df_plt = _make_iv_plt_data(
        sl_raw=chain_raw,
        sl_calib=chain_calib,
        lin_mkt=lin_mkt,
        coord=coord,
    )
    spot = chain_calib.spot
    expiry = chain_calib.df["expiry"].iloc[0]
    plot_iv_slice(
        ax=ax_smile,
        df_plt=df_plt,
        lin_mkt=lin_mkt,
        surface=surface,
        sl=chain_calib,
        coord=coord,
        spot=spot,
        idx=0,
        expiry=expiry,
    )

    # Axis labels for volatility smile
    x_label = AXIS_LABELS.get(coord, "Moneyness")
    ax_smile.set_xlabel(x_label)
    ax_smile.set_ylabel("Implied volatility")

    title_text = ax_smile.get_title()
    if title_text:
        ax_smile.set_title("")
        fig.suptitle(title_text, y=0.95)

    ax_smile.legend()
    plot_mixture_density(
        ax_rnd,
        params=params,
        tau=tau_slice,
        coord=density_coord,
        n=n,
    )

    fig.tight_layout()
    return fig, (ax_smile, ax_rnd)


def plot_iv_3d_surface(
    ax: plt.Axes,
    coord: str,
    surface: VolSurface,
    chain: OptionChain,
    lin_mkt: LinearEquityMarket,
    n_maturities: int = 80,
    n_strikes: int = 80,
    cmap: str = "viridis",
    scatter_size: float = 8.0,
) -> None:
    """Plot a 3D implied-vol surface and overlay market IV points."""
    axis_limits = AXIS_LIMITS.get(coord)
    tau_observed = np.unique(chain.tau)
    if tau_observed.size == 0:
        msg = "chain must contain at least one option quote."
        raise ValueError(msg)

    tau_max = max(float(np.nanmax(tau_observed)), 3.0)
    tau_values = np.linspace(0.05, tau_max, n_maturities)

    strike_grid = None
    coord_grid = None
    moneyness_converter = MONEYNESS_REGISTRY.get(coord)(le=lin_mkt)
    if axis_limits is None or coord == "k":
        strike_min = float(np.nanmin(chain.k))
        strike_max = float(np.nanmax(chain.k))
        if not np.isfinite(strike_min) or not np.isfinite(strike_max) or strike_min >= strike_max:
            msg = f"Unable to construct a strike grid within bounds for coordinate '{coord}'."
            raise ValueError(msg)
        strike_grid = np.linspace(strike_min, strike_max, n_strikes)
    else:
        coord_grid = np.linspace(axis_limits[0], axis_limits[1], n_strikes)

    tau_mesh = np.repeat(tau_values[:, None], n_strikes, axis=1)
    coord_mesh = np.empty_like(tau_mesh, dtype=float)
    iv_mesh = np.empty_like(tau_mesh, dtype=float)

    for row_idx, tau in enumerate(tau_values):
        if coord_grid is None:
            strike_row = strike_grid
        else:
            strike_row = _invert_moneyness_to_strike(
                coord=coord,
                moneyness=coord_grid,
                tau=tau,
                le=lin_mkt,
                surface=surface,
            )

        tau_row = np.full(n_strikes, tau, dtype=float)
        iv_row = surface.vol(strike_row, tau_row)
        iv_mesh[row_idx, :] = iv_row

        if coord_grid is not None:
            coord_mesh[row_idx, :] = coord_grid
        else:
            atmf_sigma = None
            if moneyness_converter and moneyness_converter.requires_sigma:
                fwd_value = float(lin_mkt.fwd(tau))
                atmf_sigma = surface.vol(fwd_value, tau)

            coord_mesh[row_idx, :] = moneyness_converter.value(
                strike=strike_row,
                tau=tau_row,
                sigma=atmf_sigma,
            )

    valid_surface = np.isfinite(tau_mesh) & np.isfinite(coord_mesh) & np.isfinite(iv_mesh) & (iv_mesh > 0.0)
    if not np.any(valid_surface):
        msg = "surface produced no finite positive implied volatilities on the plotting grid."
        raise ValueError(msg)

    tau_mesh = np.ma.masked_where(~valid_surface, tau_mesh)
    coord_mesh = np.ma.masked_where(~valid_surface, coord_mesh)
    iv_mesh = np.ma.masked_where(~valid_surface, iv_mesh)

    ax.plot_surface(
        tau_mesh,
        coord_mesh,
        iv_mesh,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )

    chain_slices = [chain] if hasattr(chain, "slice_tau") else [chain_slice for _, chain_slice in chain]

    market_frames = [
        _make_iv_plt_data(
            sl_raw=chain_slice,
            sl_calib=chain_slice,
            lin_mkt=lin_mkt,
            coord=coord,
        ).assign(tau=chain_slice.slice_tau)
        for chain_slice in chain_slices
    ]

    market_df = pd.concat(market_frames, ignore_index=True)
    valid_market = np.isfinite(market_df["moneyness"]) & np.isfinite(market_df["iv_mid"])
    if axis_limits is not None:
        lower_bound, upper_bound = axis_limits
        valid_market &= (market_df["moneyness"] >= lower_bound) & (market_df["moneyness"] <= upper_bound)
    synthetic_mask = market_df.get("is_synthetic", pd.Series(False, index=market_df.index)).to_numpy(dtype=bool)
    point_styles = [
        (~synthetic_mask, "black"),
        (synthetic_mask, "gray"),
    ]
    for mask, colour in point_styles:
        scatter_mask = valid_market & mask
        if not np.any(scatter_mask):
            continue
        ax.scatter(
            market_df.loc[scatter_mask, "tau"],
            market_df.loc[scatter_mask, "moneyness"],
            market_df.loc[scatter_mask, "iv_mid"],
            c=colour,
            s=scatter_size,
            alpha=1.0,
            edgecolors=colour,
            linewidths=0.2,
            depthshade=False,
        )

    ax.set_xlabel("Expiration")
    ax.set_ylabel(AXIS_LABELS.get(coord, "Moneyness"))
    ax.set_zlabel("Implied volatility")

    if axis_limits is not None:
        ax.set_ylim(*axis_limits)

    z_max = float(np.ma.max(iv_mesh))
    if np.any(valid_market):
        z_max = max(z_max, float(np.nanmax(market_df.loc[valid_market, "iv_mid"])))
    ax.set_zlim(0.0, 1.05 * z_max)

    if "anchor" in chain.df.columns:
        anchor = pd.Timestamp(chain.df["anchor"].iloc[0]).date()
        ax.set_title(str(anchor))


def plot_iv_slice(
    ax: plt.Axes,
    df_plt: pd.DataFrame,
    lin_mkt: LinearEquityMarket,
    surface: VolSurface,
    sl: OptionChain,
    coord: str,
    spot: float,
    idx: int,
    expiry: object,
) -> None:
    """Plot market IV points with bid/ask brackets and mixture surface for one slice."""
    synthetic_mask = df_plt.get("is_synthetic", pd.Series(False, index=df_plt.index)).to_numpy(dtype=bool)
    colour = [
        (synthetic_mask, "gray", 0.8, "Synthetic"),
        (~synthetic_mask & ~df_plt["is_otm"], "lightgray", 0.3, "ITM"),
        (~synthetic_mask & df_plt["is_otm"] & ~df_plt["is_used"], "green", 0.3, "OTM - Not Used"),
        (~synthetic_mask & df_plt["is_otm"] & df_plt["is_used"], "blue", 0.8, "OTM - Used"),
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
    k_grid = np.linspace(0.2 * spot, 4 * spot, 100)
    t_grid = np.full_like(k_grid, sl.slice_tau, dtype=float)
    iv_model = surface.vol(k_grid, t_grid)

    atm_sigma = None
    moneyness_converter = MONEYNESS_REGISTRY.get(coord)(le=lin_mkt)
    if moneyness_converter and moneyness_converter.requires_sigma:
        atm_sigma = _select_atm_sigma(k_slice=k_grid, fwd_value=lin_mkt.fwd(sl.slice_tau), iv_slice=iv_model)
    m_grid = moneyness_converter.value(strike=k_grid, tau=t_grid, sigma=atm_sigma)

    ax.plot(
        m_grid,
        iv_model,
        color="red",
        linewidth=1.0,
        label="Mixture surface" if idx == 0 else None,
    )

    if coord in AXIS_LIMITS:
        ax.set_xlim(*AXIS_LIMITS[coord])

    ax.set_ylim(0.0, 0.8)
    ax.set_title(f"T={expiry.date()} (tau={sl.slice_tau:.2f})", fontsize=8)
