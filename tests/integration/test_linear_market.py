"""Synthetic recovery test for the linear-equity calibration pipeline."""

# ruff: noqa: S101

import numpy as np
import pandas as pd
from scipy.special import ndtr

from vol_risk.calibration.linear_market import run_linear_model_pipeline
from vol_risk.market_data.opt_chain import OptionChain
from vol_risk.utils.calendar import Actual365Fixed


def test_linear_market_recovers_synthetic_parameters() -> None:
    """Recover known rate and dividend-yield nodes from independently priced option pairs."""
    spot = 100.0
    sigma = 0.25
    anchor = pd.Timestamp("2025-01-01")
    expiries = (pd.Timestamp("2025-07-02"), pd.Timestamp("2026-01-01"))
    rates = np.array([0.0225, 0.035])
    dividend_yields = np.array([0.011, 0.017])
    maturities = np.array([(expiry - anchor).days / 365.0 for expiry in expiries])
    strikes = np.linspace(80.0, 120.0, 11)

    rows = []
    for expiry, tau, rate, dividend_yield in zip(expiries, maturities, rates, dividend_yields, strict=True):
        discount = np.exp(-rate * tau)
        forward = spot * np.exp((rate - dividend_yield) * tau)
        total_vol = sigma * np.sqrt(tau)

        for strike in strikes:
            d1 = (np.log(forward / strike) + 0.5 * total_vol**2) / total_vol
            d2 = d1 - total_vol
            call = discount * (forward * ndtr(d1) - strike * ndtr(d2))
            put = discount * (strike * ndtr(-d2) - forward * ndtr(-d1))

            for option_type, mid in (("C", call), ("P", put)):
                rows.append(
                    {
                        "anchor": anchor,
                        "expiry": expiry,
                        "spot": spot,
                        "strike": strike,
                        "option_type": option_type,
                        "mid": mid,
                        "bid": mid - 0.01,
                        "ask": mid + 0.01,
                        "volume": 10,
                        "open_interest": 100,
                    }
                )

    chain = OptionChain(pd.DataFrame(rows), Actual365Fixed)
    result = run_linear_model_pipeline(chain)

    assert len(result.chain) == len(rows)
    assert result.params.spot == spot
    np.testing.assert_allclose(result.params.tau, maturities, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.params.r, rates, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.params.q, dividend_yields, rtol=1e-10, atol=1e-12)
    for diagnostics in result.stats.values():
        assert diagnostics["n_obs"] == len(strikes)
        assert not diagnostics["excluded"]
        assert np.all(diagnostics["in_bid_ask"])
