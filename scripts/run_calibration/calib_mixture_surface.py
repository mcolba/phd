from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_calibration.plot_helper import (
    _instantiate_moneyness_models,
    make_iv_plt_data,
    plot_iv_slice,
    plot_mixture_combined,
)
from vol_risk.calibration.data.option_chain import OptionChain
from vol_risk.calibration.data.transformers import liquidity_filter, make_otm_to_call
from vol_risk.models.linear import calib_linear_equity_market
from vol_risk.utils.calendar import Actual365Fixed
from vol_risk.vol_surface.interpl.mixture import calib_mixture_ivs

plt.style.use("ggplot")

N_COMPONENTS = 3
MONEYNESS = "lkf"

DEFAULT_MONEYNESS = "slkf"

# Project root (phd/) and output directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    "delta": (0.02, 0.98),
}


if __name__ == "__main__":
    plots_coord = MONEYNESS

    # 1. Import option chain data
    input_data_path = Path().resolve() / "data" / "test" / "CBOE_EOD_2023-08-25.csv"

    df = (
        pd.read_csv(input_data_path)
        .rename(
            columns={
                "quote_date": "anchor",
                "expiration": "expiry",
                "trade_volume": "volume",
                "bid_eod": "bid",
                "ask_eod": "ask",
            }
        )
        .assign(
            spot=lambda x: 0.5 * (x["underlying_bid_eod"] + x["underlying_ask_eod"]),
            anchor=lambda x: pd.to_datetime(x["anchor"], format="%Y-%m-%d", errors="raise"),
            expiry=lambda x: pd.to_datetime(x["expiry"], format="%Y-%m-%d", errors="raise"),
            mid=lambda x: 0.5 * (x["bid"] + x["ask"]),
        )
    )

    def liq_filter(chain):
        return liquidity_filter(
            chain,
            oi_min=3,
            bid_min=0.001,
            mid_min=0.02,
        )

    spx_mask = (df["underlying_symbol"] == "^SPX") & (df["root"] == "SPX")
    check1 = (df["ask"] - df["bid"]) >= 0
    check2 = df["mid"] > 0
    df_spx = df.loc[spx_mask & check1 & check2]
    chain_all = liq_filter(OptionChain(df_spx, Actual365Fixed))
    chain_liq = liq_filter(chain_all)

    # 2. Calibrate linear market model
    lin_mkt, _ = calib_linear_equity_market(chain_liq)

    # cutoff_moneyness = MONEYNESS_REGISTRY["delta"](lin_mkt)
    # chain_otm = apply_cutoffs(chain_liq, moneyness=cutoff_moneyness, bounds=(0.01, 0.99))
    chain_otm = make_otm_to_call(chain_liq, lin_mkt)

    # 3. Calibrate log-normal mixture for each slice
    surface, stats = calib_mixture_ivs(
        opt=chain_otm,
        mkt=lin_mkt,
        n_components=N_COMPONENTS,
        lw_type="vega",
        transform_method="totvar_simplex",
        pdef=0.0,
    )

    mix_by_expiry = {expiry: stats[expiry]["params"] for expiry in stats}
    expiries_sorted = sorted(mix_by_expiry)

    # 4. Plot log-normal mixture parameters by maturity
    moneyness_models = _instantiate_moneyness_models(le=lin_mkt)

    if expiries_sorted:
        taus = np.array([stats[e]["tau"] for e in expiries_sorted], dtype=float)
        n_expiries = len(expiries_sorted)
        comp_count = N_COMPONENTS

        w_mat = np.empty((n_expiries, comp_count), dtype=float)
        mu_mat = np.empty((n_expiries, comp_count), dtype=float)
        sigma_mat = np.empty((n_expiries, comp_count), dtype=float)

        for i, expiry in enumerate(expiries_sorted):
            params = mix_by_expiry[expiry]
            w_mat[i, :] = params.w
            mu_mat[i, :] = params.mu
            sigma_mat[i, :] = params.sigma

        _, axes_params = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        param_sets = [
            (w_mat, "Mixture weights", "w"),
            (mu_mat, "Component means", "mu"),
            (sigma_mat, "Component std devs", "sigma"),
        ]

        for ax_idx, (ax, (mat, title, prefix)) in enumerate(zip(axes_params, param_sets, strict=False)):
            for comp_idx in range(comp_count):
                label = f"{prefix}_{comp_idx + 1}" if ax_idx == 0 else None
                ax.plot(taus, mat[:, comp_idx], marker="o", linewidth=1.0, label=label)
            ax.set_title(title)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_ylabel("Value")

        axes_params[-1].set_xlabel("tau (years)")
        axes_params[0].legend(title="Component", fontsize=8)

    # 5. Plot total variance curves for all maturities
    if expiries_sorted:
        plt.figure(figsize=(10, 6))
        spot = chain_all.spot
        axis_label = AXIS_LABELS["kf"]
        m_common = np.linspace(0.2, 2, 100)

        total_var = []
        for expiry in expiries_sorted:
            tau = stats[expiry]["tau"]
            fwd_value = float(lin_mkt.fwd(tau))
            k_grid = m_common * fwd_value
            t_grid = np.full_like(k_grid, tau, dtype=float)
            iv_model = surface.vol(k_grid, t_grid)
            total_var_t = iv_model**2 * tau
            plt.plot(m_common, total_var_t, label=f"tau={tau:.2f}")
            total_var.append(total_var_t)

        plt.title("Total Variance Curves")
        plt.xlabel(axis_label)
        plt.ylabel("Total Variance")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.show()

        total_var = np.array(total_var)
        tv_diff = total_var[1:, :] - total_var[:-1, :]

        if np.any(tv_diff < -1e-3):
            print("Calendar arbitrage detected!")

        plt.plot(m_common, total_var[10, :], label=f"tau={tau:.2f}")
        plt.plot(m_common, total_var[11, :], label=f"tau={tau:.2f}")
        plt.show()

    # 6. Plot market implied vols vs mixture surface (by expiry)
    expiries = np.unique(chain_all.expiry)
    n = expiries.size
    n_col = 2
    n_row = n // n_col + int(n % n_col > 0)
    fig, axes = plt.subplots(n_row, n_col, sharex=True, sharey=True, figsize=(10, 25))
    axes = np.atleast_1d(axes).ravel()

    spot = chain_all.spot
    axis_label = AXIS_LABELS[plots_coord]

    for idx, (expiry, sl) in enumerate(chain_all):
        ax = axes[idx]

        df_plt = make_iv_plt_data(sl, chain_otm, lin_mkt, plots_coord)
        plot_iv_slice(ax, df_plt, lin_mkt, surface, sl, plots_coord, spot, idx, expiry)

    # Hide any unused axes
    for j in range(idx + 1, axes.size):
        axes[j].axis("off")

    # Common labels
    for ax in axes[: min(n, axes.size)]:
        ax.set_xlabel(axis_label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

    # 7. Example: plot smile and risk-neutral density side by side
    def _plot_nth_smile(idx: int, file_name: Path) -> None:
        example_expiry = expiries_sorted[idx]
        example_params = mix_by_expiry[example_expiry]

        example_slice = None
        for expiry, sl in chain_all:
            if expiry == example_expiry:
                example_slice = sl
                break

        if example_slice is not None:
            fig, _ = plot_mixture_combined(
                chain_slice=example_slice,
                lin_mkt=lin_mkt,
                surface=surface,
                params=example_params,
                coord=plots_coord,
            )
            fig.savefig(file_name, dpi=300, bbox_inches="tight")
            plt.close(fig)

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_nth_smile(0, output_dir / f"gmm_{N_COMPONENTS}d_short.png")
    _plot_nth_smile(5, output_dir / f"gmm_{N_COMPONENTS}d_mid.png")
    _plot_nth_smile(14, output_dir / f"gmm_{N_COMPONENTS}d_long.png")
