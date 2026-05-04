"""Wrapper for mixture surface calibration pipeline.

Exposes a single entry point :func:`run_mixture_pipeline` that can be called
in a loop with a config object and a raw option-chain DataFrame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from vol_risk.calibration.transformers import (
    apply_cutoffs,
    compose,
    get_calendar_arb_upper_bounds,
    liquidity_filter,
    make_otm_to_call,
    min_strikes_per_slice_filter,
    remove_short_span_slices,
    repair_arbitrage,
    soft_liquidity_filter,
)
from vol_risk.models.black76 import implied_vol
from vol_risk.models.linear import LinearEquityMarket, LinearEquityParams, calib_linear_equity_market
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, VolSurface, calib_mixture_ivs
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY

if TYPE_CHECKING:
    from vol_risk.calibration.option_chain import OptionChain

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChainCutoff:
    """Parameters passed to :func:`apply_cutoffs`."""

    moneyness_type: str
    bounds: tuple[float, float]


@dataclass(frozen=True)
class ChainFilter:
    """Parameters for liquidity and moneyness cutoffs."""

    oi_min: int = 50
    bid_min: float = 0.01
    mid_min: float = 0.02
    rel_bid_ask_max: float | None = None
    min_ttm: int | None = None
    min_k_per_slice: int = 5


@dataclass(frozen=True)
class MixtureCalibConfig:
    """Configuration for the mixture surface calibration pipeline."""

    # Option chain filters & transforms
    min_k_per_slice: int = 10
    repair_arbitrage: bool = False
    liquidity_filter: ChainFilter | None = None
    moneyness_cutoff: ChainCutoff | None = None
    soft_liquidity_filter: bool = False
    remove_short_span_slices: bool = False

    # Mixture configuration
    n_components: int = 3
    pdef: float = 0.0
    lambda_smoothing: float = 0.0
    lambda_tm1_params: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lw_type: str = "vega_and_spread"
    transform_method: str = "totvar_simplex"
    t0_start_guess: str = "uninformative"
    use_calendar_arb_bounds: bool = False
    lambda_ca_bounds: float = 0.0


@dataclass(frozen=True)
class MixtureCalibResult:
    """Output of :func:`run_mixture_pipeline`."""

    lin_mkt: LinearEquityMarket
    surface: VolSurface
    params: tuple[LinearEquityParams, list[LogNormMixParams]]
    stats: tuple[dict, dict]
    chains: tuple[OptionChain, OptionChain]


def run_mixture_pipeline(
    chain: OptionChain,
    config: MixtureCalibConfig | None = None,
) -> MixtureCalibResult:
    """Run the log-normal mixture surface calibration pipeline."""
    start_time = time.time()

    # 1. Apply liquidity filter
    initial_filter = []

    if config.liquidity_filter is not None:
        initial_filter.append(
            partial(
                liquidity_filter,
                oi_min=config.liquidity_filter.oi_min,
                bid_min=config.liquidity_filter.bid_min,
                mid_min=config.liquidity_filter.mid_min,
                rel_bid_ask_max=config.liquidity_filter.rel_bid_ask_max,
                min_ttm=config.liquidity_filter.min_ttm,
            )
        )

    if config.min_k_per_slice > 1:
        initial_filter.append(partial(min_strikes_per_slice_filter, n=config.min_k_per_slice))

    chain_lm = compose(*initial_filter)(chain)
    log.info("Options used in the linear market calibration: %d", len(chain_lm))

    # 2. Calibrate linear equity market (rates / dividends)
    lin_mkt, lin_mkt_params, lin_stats = calib_linear_equity_market(chain_lm)
    log.debug("Linear market calibration stats: %s", lin_stats)

    # 3. Convert to OTM calls
    post_lm_filters = [
        partial(make_otm_to_call, le=lin_mkt),
    ]

    # 3. Apply moneyness cutoffs
    if config.moneyness_cutoff is not None:
        post_lm_filters.append(
            partial(
                apply_cutoffs,
                moneyness=MONEYNESS_REGISTRY[config.moneyness_cutoff.moneyness_type](le=lin_mkt),
                bounds=config.moneyness_cutoff.bounds,
            )
        )

    # apply soft liquidity filters
    if config.soft_liquidity_filter:
        post_lm_filters.append(
            partial(
                soft_liquidity_filter,
                lin_mkt=lin_mkt,
                oi_soft_min=50,
                rel_bid_ask_soft_max=0.20,
                min_lk_distance=0.04,
                max_lk_distance=0.01,
            )
        )

    # Ensure each slice has at least min_k_per_slice strikes
    if config.min_k_per_slice is not None:
        post_lm_filters.append(
            partial(
                min_strikes_per_slice_filter,
                n=config.min_k_per_slice,
            )
        )

    if config.remove_short_span_slices:
        post_lm_filters.append(remove_short_span_slices)

    # Remove arbitrage in the option quotes
    if config.repair_arbitrage:
        post_lm_filters.append(
            partial(
                repair_arbitrage,
                market=lin_mkt,
                tolerance=1e-6,
                min_price=1e-3,
                synthetic_weight=None,
            )
        )

    chain_vol = compose(*post_lm_filters)(chain_lm)
    log.info("Options used in the IVS calibration: %d", len(chain_vol))

    calendar_arb_bounds = None
    if config.use_calendar_arb_bounds:
        calendar_arb_bounds = get_calendar_arb_upper_bounds(chain_vol, lin_mkt, (-0.7, 0.7))

    # 6. Calibrate log-normal mixture for each expiry slice
    surface, ivs_params, ivs_stats = calib_mixture_ivs(
        opt=chain_vol,
        mkt=lin_mkt,
        n_components=config.n_components,
        lw_type=config.lw_type,
        transform_method=config.transform_method,
        pdef=config.pdef,
        lambda_tm1_params=config.lambda_tm1_params,
        lambda_smoothing=config.lambda_smoothing,
        calendar_arb_bounds=calendar_arb_bounds,
        lambda_ca_bounds=config.lambda_ca_bounds,
        t0_start_guess=config.t0_start_guess,
    )

    elapsed = time.time() - start_time
    msg = (
        f"Calibration complete. Expiries calibrated: {len(ivs_params)}, "
        f"MAE: {ivs_stats['iv_mae_approx']:.4f}, "
        f"elapsed time: {elapsed:.2f} seconds."
    )
    log.info(msg)

    return MixtureCalibResult(
        surface=surface,
        lin_mkt=lin_mkt,
        params=(lin_mkt_params, ivs_params),
        stats=(lin_stats, ivs_stats),
        chains=(chain_lm, chain_vol),
    )
