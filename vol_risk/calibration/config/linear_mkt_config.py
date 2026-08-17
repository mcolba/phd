"""Default configuration for index linear-equity calibration."""

from dataclasses import dataclass

from vol_risk.calibration.config.mixture_config import MixtureCalibIndexConfig
from vol_risk.calibration.mixture_pipeline import ChainFilter


@dataclass(frozen=True)
class LinearModelCalibConfig:
    """Configuration shared with the linear stage of index IVS calibration."""

    liquidity_filter = ChainFilter(
        oi_min=1,
        bid_min=0.01,
        mid_min=0.02,
        rel_bid_ask_max=2.5,
        min_ttm=10,
    )
    min_k_per_slice = 10

