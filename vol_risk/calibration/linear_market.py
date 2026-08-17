"""Linear-equity calibration pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from vol_risk.calibration.config.linear_mkt_config import LinearModelCalibConfig
from vol_risk.market_data.opt_chain_transformers import (
    compose,
    liquidity_filter,
    min_strikes_per_slice_filter,
)
from vol_risk.models.linear import (
    LinearEquityMarket,
    LinearEquityParams,
    calib_linear_equity_market,
)

if TYPE_CHECKING:
    from vol_risk.market_data.opt_chain import OptionChain

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearModelCalibResult:
    """Output of the linear-equity calibration pipeline."""

    model: LinearEquityMarket
    params: LinearEquityParams
    stats: dict[object, object]
    chain: OptionChain


def run_linear_model_pipeline(
    chain: OptionChain,
    config: LinearModelCalibConfig | None = None,
) -> LinearModelCalibResult:
    """Filter an option chain and calibrate only its linear-equity model."""
    config = config or LinearModelCalibConfig()
    transforms = []

    if config.liquidity_filter is not None:
        chain_filter = config.liquidity_filter
        transforms.append(
            partial(
                liquidity_filter,
                oi_min=chain_filter.oi_min,
                bid_min=chain_filter.bid_min,
                mid_min=chain_filter.mid_min,
                rel_bid_ask_max=chain_filter.rel_bid_ask_max,
                min_ttm=chain_filter.min_ttm,
                validate_chain=False,
            )
        )

    if config.min_k_per_slice > 1:
        transforms.append(
            partial(
                min_strikes_per_slice_filter,
                n=config.min_k_per_slice,
                validate_chain=False,
            )
        )

    calibration_chain = compose(*transforms)(chain)
    log.info("Options used in the linear market calibration: %d", len(calibration_chain))
    model, params, stats = calib_linear_equity_market(calibration_chain)
    return LinearModelCalibResult(
        model=model,
        params=params,
        stats=stats,
        chain=calibration_chain,
    )
