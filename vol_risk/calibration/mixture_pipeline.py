"""Wrapper for mixture surface calibration pipeline.

Exposes a single entry point :func:`run_mixture_pipeline` that can be called
in a loop with a config object and a raw option-chain DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vol_risk.calibration.transformers import (
    append_synthetic_quotes,
    apply_cutoffs,
    detect_arbitrage,
    liquidity_filter,
    make_otm_to_call,
    repair_arbitrage,
)
from vol_risk.models.linear import LinearEquityMarket, LinearEquityParams, calib_linear_equity_market
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, VolSurface, calib_mixture_ivs
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY

if TYPE_CHECKING:
    from vol_risk.calibration.option_chain import OptionChain

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChainCutoff:
    """Parameters passed to :func:`apply_cutoffs`.

    Attributes:
        moneyness_type: Key in ``MONEYNESS_REGISTRY`` selecting the moneyness measure (e.g. "delta", "lkf", "slkf").
        bounds: (lower, upper) bounds in the chosen moneyness coordinate.
    """

    moneyness_type: str
    bounds: tuple[float, float]


@dataclass(frozen=True)
class ChainFilter:
    """Parameters for liquidity and moneyness cutoffs.

    Attributes:
        oi_min: Minimum open interest for liquidity filter.
        bid_min: Minimum bid price for liquidity filter.
        mid_min: Minimum mid price for liquidity filter.
        rel_bid_ask_max: Maximum relative bid-ask spread for liquidity filter.
        min_ttm: Minimum business days between anchor and expiry.
        cutoff: Optional moneyness cutoff config. None disables cutoffs.
    """

    oi_min: int = 3
    bid_min: float = 0.01
    mid_min: float = 0.02
    rel_bid_ask_max: float | None = None
    min_ttm: int | None = None
    cutoff: ChainCutoff | None = None
    min_k_per_slice: int = 3


@dataclass(frozen=True)
class ThinPlateSmilePreprocess:
    """Configuration for synthetic thin-plate smile augmentation before calibration."""

    lkf_bounds: tuple[float, float] = (-0.7, 0.7)
    grid_size: int = 41
    spline_smoothing: float = 0.0
    min_obs_per_slice: int = 3
    min_total_variance: float = 1e-8
    synthetic_weight: float = 10.0
    repair_min_price: float | None = None
    repair_tolerance: float = 0.0
    solver: str = "glpk"


@dataclass(frozen=True)
class MixtureCalibConfig:
    """Configuration for the mixture surface calibration pipeline.

    Attributes:
        n_components: Number of log-normal mixture components.
        lw_type: Loss-weight scheme passed to ``calib_mixture_ivs`` ("vega" or "uniform").
        transform_method: Bijection method passed to ``calib_mixture_ivs`` (e.g. "simplex", "totvar_simplex").
        pdef: Probability-of-default parameter.
        filters: Option filter settings (liquidity and cutoff).
        thin_plate_preprocess: Optional synthetic quote augmentation before calibration.
    """

    n_components: int = 3
    lw_type: str = "vega"
    transform_method: str = "totvar_simplex"
    pdef: float = 0.0
    filters: ChainFilter = field(default_factory=ChainFilter)
    thin_plate_preprocess: ThinPlateSmilePreprocess | None = None
    repair_arbitrage: bool = False


@dataclass(frozen=True)
class MixtureCalibResult:
    """Output of :func:`run_mixture_pipeline`.

    Attributes:
        surface: Calibrated :class:`VolSurface` object.
        stats: Per-expiry calibration statistics returned by
            ``calib_mixture_ivs``.
        lin_mkt: Calibrated linear equity market model.
        chain_otm: Filtered OTM market option chain after cutoffs, before synthetic augmentation.
        chain_calib: Option chain actually used during calibration.
    """

    lin_mkt: LinearEquityMarket
    surface: VolSurface
    params: tuple[LinearEquityParams, list[LogNormMixParams]]
    stats: tuple[dict, dict]
    chain_otm: OptionChain
    chain_calib: OptionChain


def run_mixture_pipeline(
    chain: OptionChain,
    config: MixtureCalibConfig | None = None,
) -> MixtureCalibResult:
    """Run the log-normal mixture surface calibration pipeline.

    Parameters
    ----------
    chain:
        Option chain to calibrate. Use
        :func:`vol_risk.calibration.data_loaders.cboe_to_option_chain` to build
        one from a raw CBOE EOD DataFrame.
    config:
        Calibration configuration.  Defaults to :class:`MixtureCalibConfig` with all default values when ``None``.

    Returns:
    -------
    MixtureCalibResult
        Calibrated surface, per-expiry stats, linear market model and the filtered OTM chain.
    """
    if config is None:
        config = MixtureCalibConfig()

    # 1. Apply liquidity filter
    filt_cfg = config.filters
    chain_liq = liquidity_filter(
        chain,
        oi_min=filt_cfg.oi_min,
        bid_min=filt_cfg.bid_min,
        mid_min=filt_cfg.mid_min,
        rel_bid_ask_max=filt_cfg.rel_bid_ask_max,
        min_ttm=filt_cfg.min_ttm,
        min_k_per_slice=filt_cfg.min_k_per_slice,
    )

    log.info("Options after liquidity filter: %d", len(chain_liq))

    # 2. Calibrate linear equity market (rates / dividends)
    lin_mkt, lin_mkt_params, lin_stats = calib_linear_equity_market(chain_liq)
    log.debug("Linear market calibration stats: %s", lin_stats)

    # 3. Convert to OTM calls
    chain_otm = make_otm_to_call(chain_liq, lin_mkt)
    log.info("Options after OTM conversion: %d", len(chain_otm))

    # 4. Optionally apply moneyness cutoffs
    if filt_cfg.cutoff is not None:
        cutoff_cfg = filt_cfg.cutoff
        if cutoff_cfg.moneyness_type not in MONEYNESS_REGISTRY:
            msg = f"Unknown moneyness_type {cutoff_cfg.moneyness_type!r}. Available: {list(MONEYNESS_REGISTRY)}"
            raise ValueError(msg)
        moneyness = MONEYNESS_REGISTRY[cutoff_cfg.moneyness_type](le=lin_mkt)
        chain_otm = apply_cutoffs(chain_otm, moneyness=moneyness, bounds=cutoff_cfg.bounds)
        log.info("Options after cutoff filter: %d", len(chain_otm))

    # 5. Optionally augment each smile with synthetic thin-plate quotes
    chain_calib = chain_otm
    if config.thin_plate_preprocess is not None:
        preprocess_cfg = config.thin_plate_preprocess
        chain_calib = append_synthetic_quotes(
            chain=chain_otm,
            market=lin_mkt,
            k_min=preprocess_cfg.lkf_bounds[0],
            k_max=preprocess_cfg.lkf_bounds[1],
            grid_size=preprocess_cfg.grid_size,
            spline_smoothing=preprocess_cfg.spline_smoothing,
            min_obs_per_slice=preprocess_cfg.min_obs_per_slice,
            min_total_variance=preprocess_cfg.min_total_variance,
            synthetic_weight=preprocess_cfg.synthetic_weight,
        )
        synthetic_count = int(chain_calib.df["synthetic"].sum()) if "synthetic" in chain_calib.df.columns else 0
        log.info(
            "Options after thin-plate augmentation: %d (%d synthetic)",
            len(chain_calib),
            synthetic_count,
        )

    if config.repair_arbitrage is not None:
        chain_calib = repair_arbitrage(
            chain=chain_calib,
            market=lin_mkt,
            tolerance=1e-6,
            min_price=1e-6,
            synthetic_weight=None,
        )
        log.info("Options after arbitrage repair: %d", len(chain_calib))

    arbitrage_stats = detect_arbitrage(
        chain=chain_calib,
        market=lin_mkt,
        repair_tolerance=1e-6,
        min_price=1e-6,
    )
    if arbitrage_stats["n_breaches"] > 0:
        log.warning(
            "Static arbitrage check after preprocessing found %d breaches across %d constraints for %d quotes.",
            arbitrage_stats["n_breaches"],
            arbitrage_stats["n_constraints"],
            arbitrage_stats["n_quotes"],
        )

    # 6. Calibrate log-normal mixture for each expiry slice
    surface, ivs_params = calib_mixture_ivs(
        opt=chain_calib,
        mkt=lin_mkt,
        n_components=config.n_components,
        lw_type=config.lw_type,
        transform_method=config.transform_method,
        pdef=config.pdef,
    )

    log.info("Calibration complete. Expiries calibrated: %d", len(ivs_params))

    return MixtureCalibResult(
        surface=surface,
        lin_mkt=lin_mkt,
        params=(lin_mkt_params, ivs_params),
        stats=lin_stats,
        chain_otm=chain_otm,
        chain_calib=chain_calib,
    )
