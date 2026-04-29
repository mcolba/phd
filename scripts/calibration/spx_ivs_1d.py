# %matplotlib ipympl
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as ds

from vol_risk.calibration.data_loaders import make_optionmetrics_chain
from vol_risk.calibration.mixture_pipeline import (
    ChainCutoff,
    ChainFilter,
    MixtureCalibConfig,
    run_mixture_pipeline,
)
from vol_risk.calibration.plot_helpers import (
    _instantiate_moneyness_models,
    plot_iv_3d_surface,
    plot_mixture_smile_and_rnd,
    plot_smile_and_mkt_grid,
    plot_total_variance_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

plt.style.use("ggplot")

N_COMPONENTS = 3
MONEYNESS = "lkf"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIONS_DIR = Path(r"D:\option_metrics\parquet")

EOD_DATE = "2007-12-31 00:00:00.000000000"
TICKER = "SPX"


if __name__ == "__main__":
    plots_coord = MONEYNESS

    # 1. Load OptionMetrics data for a single EOD date
    logger.info("Opening parquet dataset at %s", OPTIONS_DIR)
    dataset = ds.dataset(str(OPTIONS_DIR), format="parquet", partitioning="hive")

    df = dataset.to_table(filter=ds.field("date") == EOD_DATE).to_pandas()
    if df.empty:
        msg = f"No data found for date={EOD_DATE}"
        raise RuntimeError(msg)

    logger.info("Loaded %d rows for %s", len(df), EOD_DATE)
    chain_all = make_optionmetrics_chain(df)

    # 2. Run calibration pipeline
    epsilon = 1e-8
    cutoff_cfg = ChainCutoff("delta", (epsilon, 1.0 - epsilon))
    config = MixtureCalibConfig(
        n_components=N_COMPONENTS,
        t0_start_guess="smirk",
        lw_type="vega_and_spread",
        transform_method="totvar_simplex",
        lambda_tm1_params=(1e-3, 1e-2, 1e-3),
        lambda_smoothing=0,
        min_k_per_slice=10,
        repair_arbitrage=True,
        soft_liquidity_filter=True,
        remove_short_span_slices=True,
        use_calendar_arb_bounds=True,
        liquidity_filter=ChainFilter(
            oi_min=1,
            bid_min=0.01,
            mid_min=0.02,
            rel_bid_ask_max=3,
            min_ttm=10,
        ),
        moneyness_cutoff=cutoff_cfg,
    )
    result = run_mixture_pipeline(chain_all, config)

    surface = result.surface
    params_ivs = result.params[1]
    lin_mkt = result.lin_mkt
    _, chain_ivs = result.chains

    mix_by_expiry = {expiry: params_ivs[expiry]["params"] for expiry in params_ivs}
    expiries_sorted = sorted(mix_by_expiry)

    # Plot linear market: zero rate, zero dividend yield, forward curve, and discount factor
    if expiries_sorted:
        taus_plot = np.array([params_ivs[e]["tau"] for e in expiries_sorted], dtype=float)
        zero_rate = lin_mkt.zero_rate(taus_plot)
        zero_dvd_yield = lin_mkt.zero_dvd_yield(taus_plot)
        forward_curve = lin_mkt.fwd(taus_plot)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(taus_plot, zero_rate, label="Zero Rate", color="tab:blue")
        ax1.plot(taus_plot, zero_dvd_yield, label="Zero Dividend Yield", color="tab:orange")
        ax1.scatter(taus_plot, zero_rate, color="tab:blue", marker="o", s=60, zorder=3)
        ax1.scatter(taus_plot, zero_dvd_yield, color="tab:orange", marker="o", s=60, zorder=3)
        ax1.set_xlabel("tau (years)")
        ax1.set_ylabel("Rate / Yield")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(taus_plot, forward_curve, label="Forward Curve", color="tab:green", linestyle="--")
        ax2.axhline(lin_mkt.spot, color="tab:gray", linestyle=":", label="Spot Price")
        ax2.set_ylabel("Forward Price")
        ax2.legend(loc="upper right")

        plt.title(f"Linear Market: Zero Rate, Zero Dividend Yield, Forward Curve — {TICKER} {EOD_DATE}")
        plt.tight_layout()
        plt.show()

    # 3. Plot log-normal mixture parameters by maturity
    moneyness_models = _instantiate_moneyness_models(le=lin_mkt)

    if expiries_sorted:
        taus = np.array([params_ivs[e]["tau"] for e in expiries_sorted], dtype=float)
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
            ax.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_ylabel("Value")

        axes_params[-1].set_xlabel("tau (years)")
        axes_params[0].legend(title="Component", fontsize=8)
        plt.suptitle(f"Mixture Parameters — {TICKER} {EOD_DATE}")
        plt.tight_layout()
        plt.show()

    # 4. Plot total variance curves for all maturities
    if expiries_sorted:
        fig, ax = plt.subplots(figsize=(10, 6))
        total_var = plot_total_variance_curves(
            ax=ax,
            expiries_sorted=expiries_sorted,
            params_ivs=params_ivs,
            lin_mkt=lin_mkt,
            surface=surface,
            coord="lkf",
        )
        ax.set_title(f"Total Variance Curves — {TICKER} {EOD_DATE}")
        plt.show()

        tv_diff = total_var[1:, :] - total_var[:-1, :]
        if np.any(tv_diff < -1e-3):
            logger.warning("Calendar arbitrage detected!")

    # 5. Plot market implied vols vs mixture surface (by expiry)
    plot_smile_and_mkt_grid(
        chain_raw=chain_all,
        chain_calib=chain_ivs,
        lin_mkt=lin_mkt,
        surface=surface,
        coord=plots_coord,
        n_col=3,
        title=f"Market IV vs Mixture Surface — {TICKER} {EOD_DATE}",
    )
    plt.show()

    # 6. Plot 3D surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    plot_iv_3d_surface(
        ax=ax,
        coord=plots_coord,
        surface=surface,
        chain=chain_ivs,
        lin_mkt=lin_mkt,
    )
    ax.set_title(f"Mixture IV Surface — {TICKER} {EOD_DATE}")
    plt.tight_layout()
    plt.show()

    # 7. Plot smile and risk-neutral density side by side for selected expiries
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _plot_nth_smile(idx: int, file_name: Path = None) -> None:
        if idx >= len(expiries_sorted):
            logger.warning(
                "Skipping plot for index %d — only %d expiries available.",
                idx,
                len(expiries_sorted),
            )
            return

        key = expiries_sorted[idx]
        example_params = mix_by_expiry[key]

        fig, _ = plot_mixture_smile_and_rnd(
            chain_raw=chain_all[key],
            chain_calib=chain_ivs[key],
            lin_mkt=lin_mkt,
            surface=surface,
            params=example_params,
            coord=plots_coord,
        )
        if file_name:
            fig.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    date_tag = EOD_DATE.replace("-", "")
    _plot_nth_smile(0)
    _plot_nth_smile(min(5, len(expiries_sorted) - 1))
    _plot_nth_smile(min(14, len(expiries_sorted) - 1))
