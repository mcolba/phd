from dataclasses import dataclass

from vol_risk.calibration.mixture_pipeline import ChainCutoff, ChainFilter, MixtureCalibConfig

DELTA_CUTOFF_EPSILON = 1e-8


@dataclass(frozen=True)
class MixtureCalibIndexConfig(MixtureCalibConfig):
    """Default configuration for the SPX IVS calibration pipeline."""

    # Global filters (applies to both linear equity and mixture IVS calibration)
    liquidity_filter = ChainFilter(
        oi_min=1,
        bid_min=0.01,
        mid_min=0.02,
        rel_bid_ask_max=2.5,
        min_ttm=10,
    )

    # Mixture filters (applies to mixture IVS calibration)
    moneyness_cutoff = ChainCutoff("delta", (DELTA_CUTOFF_EPSILON, 1.0 - DELTA_CUTOFF_EPSILON))
    soft_liquidity_filter = True
    min_k_per_slice = 20
    remove_short_span_slices = True
    repair_arbitrage = True

    # Mixture configuration
    n_components = 3
    lambda_smoothing = 0.0
    lambda_tm1_params = (1e-3, 1e-2, 1e-3)
    t0_start_guess = "smirk"
    lw_type = "vega_and_spread"
    transform_method = "totvar_simplex"
    use_calendar_arb_bounds = True
    lambda_ca_bounds = 10
