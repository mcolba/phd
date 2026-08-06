"""Wrapper for mixture surface calibration pipeline.

Exposes a single entry point :func:`run_mixture_pipeline` that can be called
in a loop with a config object and a raw option-chain DataFrame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from vol_risk.market_data.opt_chain_transformers import (
    ChainCutoff,
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
from vol_risk.models.black76 import black76_price
from vol_risk.models.linear import LinearEquityMarket, LinearEquityParams, calib_linear_equity_market
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, VolSurface, calib_mixture_ivs
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY

if TYPE_CHECKING:
    from vol_risk.market_data.opt_chain import OptionChain

log = logging.getLogger(__name__)

_CALENDAR_ARB_LKF_GRID = np.linspace(-0.5, 0.5, 25, dtype=float)
_CALENDAR_ARB_TAU_GRID = np.array([1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]) * 30.0 / 365.0
_CALENDAR_ARB_TOL = 1e-6


class MixturePipelineReturnCode(IntEnum):
    """Return codes for :func:`run_mixture_pipeline`."""

    OK = 0
    WARNING = 1


class _WarningCaptureHandler(logging.Handler):
    """Capture warning records emitted during a pipeline run."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _calendar_arb_tau_grid(chain: OptionChain) -> np.ndarray:
    """Build the maturity grid used by the surface calendar-arbitrage check."""
    tau_min = float(np.min(chain.tau))
    tau_max = float(np.max(chain.tau))

    in_range = np.logical_and(
        np.greater_equal(_CALENDAR_ARB_TAU_GRID, tau_min),
        np.less_equal(_CALENDAR_ARB_TAU_GRID, tau_max),
    )
    fixed_grid = _CALENDAR_ARB_TAU_GRID[in_range]
    if fixed_grid.size >= 2:
        return fixed_grid

    return np.unique(chain.tau.astype(float))


def _normalized_call_prices(
    surface: VolSurface,
    lin_mkt: LinearEquityMarket,
    tau_grid: np.ndarray,
    lkf_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate normalized call prices on a fixed log-forward-moneyness grid."""
    prices = np.empty((tau_grid.size, lkf_grid.size), dtype=float)

    for i, tau in enumerate(tau_grid):
        fwd = float(lin_mkt.fwd(tau))
        disc = float(lin_mkt.df(tau))
        strike = fwd * np.exp(lkf_grid)
        sigma = surface.vol(k=strike, t=tau)
        call_price = black76_price(
            fwd=fwd,
            strike=strike,
            tau=tau,
            sigma=sigma,
            disc=disc,
            is_call=True,
        )
        prices[i] = np.asarray(call_price, dtype=float) / (disc * fwd)

    return prices


def _detect_calendar_arbitrage(
    surface: VolSurface,
    lin_mkt: LinearEquityMarket,
    chain: OptionChain,
    lkf_grid: np.ndarray = _CALENDAR_ARB_LKF_GRID,
    tolerance: float = _CALENDAR_ARB_TOL,
) -> int:
    """Check calendar monotonicity of normalized call prices on a fixed grid."""
    tau_grid = _calendar_arb_tau_grid(chain)
    if tau_grid.size < 2:
        return 0

    price_norm = _normalized_call_prices(
        surface=surface,
        lin_mkt=lin_mkt,
        tau_grid=tau_grid,
        lkf_grid=lkf_grid,
    )
    diff = np.diff(price_norm, axis=0)
    violations = diff < -tolerance
    return int(np.count_nonzero(violations))


def _classify_return_code(
    ivs_stats: dict,
    warning_messages: list[str],
) -> MixturePipelineReturnCode:
    """Map diagnostics to a single return code."""
    _ = ivs_stats
    if warning_messages:
        return MixturePipelineReturnCode.WARNING

    return MixturePipelineReturnCode.OK


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
    moneyness_cutoff: list[ChainCutoff] | None = None
    soft_liquidity_filter: bool = False
    remove_short_span_slices: bool = False

    # Mixture configuration
    n_components: int = 3
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
    return_code: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def run_mixture_pipeline(
    chain: OptionChain,
    config: MixtureCalibConfig | None = None,
) -> MixtureCalibResult:
    """Run the log-normal mixture surface calibration pipeline."""
    config = config or MixtureCalibConfig()
    start_time = time.time()
    warning_handler = _WarningCaptureHandler()
    package_logger = logging.getLogger("vol_risk")
    package_logger.addHandler(warning_handler)

    try:
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
                    validate_chain=False,
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
            partial(make_otm_to_call, le=lin_mkt, validate_chain=False),
        ]

        # 3. Apply moneyness cutoffs
        if config.moneyness_cutoff is not None:
            post_lm_filters.append(
                partial(
                    apply_cutoffs,
                    cutoffs=config.moneyness_cutoff,
                    lin_mkt=lin_mkt,
                    validate_chain=False,
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
                    validate_chain=False,
                )
            )

        # Ensure each slice has at least min_k_per_slice strikes
        if config.min_k_per_slice is not None:
            post_lm_filters.append(
                partial(
                    min_strikes_per_slice_filter,
                    n=config.min_k_per_slice,
                    validate_chain=False,
                )
            )

        if config.remove_short_span_slices:
            post_lm_filters.append(partial(remove_short_span_slices, validate_chain=False))

        # Remove arbitrage in the option quotes
        if config.repair_arbitrage:
            post_lm_filters.append(
                partial(
                    repair_arbitrage,
                    market=lin_mkt,
                    tolerance=1e-6,
                    min_price=1e-3,
                    synthetic_weight=None,
                    validate_chain=False,
                )
            )

        chain_vol = compose(*post_lm_filters)(chain_lm)
        log.info("Options used in the IVS calibration: %d", len(chain_vol))

        calendar_arb_bounds = None
        if config.use_calendar_arb_bounds:
            calendar_arb_bounds = get_calendar_arb_upper_bounds(
                chain_vol, lin_mkt, (_CALENDAR_ARB_LKF_GRID.min() - 0.05, _CALENDAR_ARB_LKF_GRID.max() + 0.05)
            )

        # 6. Calibrate log-normal mixture for each expiry slice
        surface, ivs_params, ivs_stats = calib_mixture_ivs(
            opt=chain_vol,
            mkt=lin_mkt,
            n_components=config.n_components,
            lw_type=config.lw_type,
            transform_method=config.transform_method,
            lambda_tm1_params=config.lambda_tm1_params,
            lambda_smoothing=config.lambda_smoothing,
            calendar_arb_bounds=calendar_arb_bounds,
            lambda_ca_bounds=config.lambda_ca_bounds,
            t0_start_guess=config.t0_start_guess,
        )

        calendar_arb_count = _detect_calendar_arbitrage(
            surface=surface,
            lin_mkt=lin_mkt,
            chain=chain_vol,
        )
        ivs_stats["calendar_arb_num_violations"] = calendar_arb_count
        has_calendar_arb = calendar_arb_count > 0

        if has_calendar_arb:
            log.warning(
                "Calendar arbitrage detected on normalized price grid: %d violation(s).",
                calendar_arb_count,
            )

        return_code = _classify_return_code(
            ivs_stats=ivs_stats,
            warning_messages=warning_handler.messages,
        )
        ivs_stats["warning_messages"] = tuple(warning_handler.messages)
        ivs_stats["return_code"] = int(return_code)

        elapsed = time.time() - start_time
        msg = (
            f"Calibration complete. Expiries calibrated: {len(ivs_params)}, "
            f"MAE: {ivs_stats['iv_mae_approx']:.4f}, "
            f"return code: {int(return_code)}, "
            f"elapsed time: {elapsed:.2f} seconds."
        )
        log.info(msg)

        return MixtureCalibResult(
            surface=surface,
            lin_mkt=lin_mkt,
            params=(lin_mkt_params, ivs_params),
            stats=(lin_stats, ivs_stats),
            chains=(chain_lm, chain_vol),
            warnings=tuple(warning_handler.messages),
            return_code=int(return_code),
        )
    finally:
        package_logger.removeHandler(warning_handler)
