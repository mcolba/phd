"""Unit tests for linear-equity market curve evaluation."""

import numpy as np

from vol_risk.models.linear import LinearEquityMarket, make_raw_disc_curve, make_raw_interpolator


def test_linear_market_interpolates_and_extrapolates_curves() -> None:
    """Use linear total rates between nodes and flat zero rates beyond the last node."""
    spot = 100.0
    node_maturities = np.array([0.5, 1.0])
    rates = np.array([0.02, 0.04])
    dividend_yields = np.array([0.01, 0.015])
    market = LinearEquityMarket(
        spot=spot,
        disc_curve=make_raw_disc_curve(node_maturities, rates),
        cont_carry_curve=make_raw_interpolator(node_maturities, dividend_yields),
    )

    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.25])
    integrated_rates = np.array([0.005, 0.01, 0.025, 0.04, 0.05])
    integrated_dividend_yields = np.array([0.0025, 0.005, 0.01, 0.015, 0.01875])

    np.testing.assert_allclose(market.zero_rate(maturities), integrated_rates / maturities, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        market.zero_dvd_yield(maturities), integrated_dividend_yields / maturities, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(market.disc(maturities), np.exp(-integrated_rates), rtol=1e-12)
    np.testing.assert_allclose(
        market.fwd(maturities), spot * np.exp(integrated_rates - integrated_dividend_yields), rtol=1e-12
    )
