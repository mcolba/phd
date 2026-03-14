"""Wrapper for mixture surface calibration pipeline.

Exposes a single entry point :func:`run_mixture_pipeline` that can be called
in a loop with a config object and a raw option-chain DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from vol_risk.calibration.data.option_chain import OptionChain
from vol_risk.calibration.data.transformers import apply_cutoffs, liquidity_filter, make_otm_to_call
from vol_risk.models.linear import LinearEquityMarket, LinearEquityParams, calib_linear_equity_market
from vol_risk.vol_surface.interpl.mixture import LogNormMixParams, VolSurface, calib_mixture_ivs
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY

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
        rel_bid_ask: Minimum relative bid-ask spread for liquidity filter.
        cutoff: Optional moneyness cutoff config. None disables cutoffs.
    """

    oi_min: int = 3
    bid_min: float = 0.01
    mid_min: float = 0.02
    rel_bid_ask_max: float | None = None
    cutoff: ChainCutoff | None = None
    min_k_per_slice: int = 3


@dataclass(frozen=True)
class MixtureCalibConfig:
    """Configuration for the mixture surface calibration pipeline.

    Attributes:
        n_components: Number of log-normal mixture components.
        lw_type: Loss-weight scheme passed to ``calib_mixture_ivs`` ("vega" or "uniform").
        transform_method: Bijection method passed to ``calib_mixture_ivs`` (e.g. "totvar", "simplex").
        pdef: Probability-of-default parameter.
        filters: Option filter settings (liquidity and cutoff).
    """

    n_components: int = 3
    lw_type: str = "vega"
    transform_method: str = "totvar"
    pdef: float = 0.0
    filters: ChainFilter = field(default_factory=ChainFilter)


@dataclass(frozen=True)
class MixtureCalibResult:
    """Output of :func:`run_mixture_pipeline`.

    Attributes:
        surface: Calibrated :class:`VolSurface` object.
        stats: Per-expiry calibration statistics returned by
            ``calib_mixture_ivs``.
        lin_mkt: Calibrated linear equity market model.
        chain_otm: Filtered OTM option chain used during calibration.
    """

    lin_mkt: LinearEquityMarket
    surface: VolSurface
    params: tuple[LinearEquityParams, list[LogNormMixParams]]
    stats: tuple[dict, dict]
    chain_otm: OptionChain


def run_mixture_pipeline(
    chain: OptionChain,
    config: MixtureCalibConfig | None = None,
) -> MixtureCalibResult:
    """Run the log-normal mixture surface calibration pipeline.

    Parameters
    ----------
    chain:
        Option chain to calibrate.  Use :func:`vol_risk.calibration.data.loaders.cboe_to_option_chain` to build one from a raw CBOE EOD DataFrame.
    config:
        Calibration configuration.  Defaults to :class:`MixtureCalibConfig` with all default values when ``None``.

    Returns:
    -------
    MixtureCalibResult
        Calibrated surface, per-expiry stats, linear market model and the filtered OTM chain.
    """
    if config is None:
        config = MixtureCalibConfig()

    log.info("Starting mixture calibration: n_components=%d", config.n_components)

    # 1. Apply liquidity filter
    filt_cfg = config.filters
    chain_liq = liquidity_filter(
        chain,
        oi_min=filt_cfg.oi_min,
        bid_min=filt_cfg.bid_min,
        mid_min=filt_cfg.mid_min,
        rel_bid_ask_max=filt_cfg.rel_bid_ask_max,
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

    # 5. Calibrate log-normal mixture for each expiry slice
    surface, ivs_params = calib_mixture_ivs(
        opt=chain_otm,
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
    )
