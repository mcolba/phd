from __future__ import annotations

import pathlib
import unittest
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from vol_risk.models.heston import (
    HestonParams,
    heston_calibrator,
    heston_delta,
    heston_jacobian,
    heston_price_cui,
)
from vol_risk.models.linear import LinearEquityMarket, make_simple_linear_market

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "heston_prices_quantlib.csv"


class TestHestonModel(unittest.TestCase):
    """Numerical regression tests for the Heston DLL bindings."""

    PARAM_COLS: ClassVar[list[str]] = ["kappa", "v_inf", "vol_of_vol", "rho", "v0"]

    @classmethod
    def setUpClass(cls) -> None:
        cls.ql_data = pd.read_csv(DATA_PATH)

    @staticmethod
    def _make_market(x: pd.Series) -> LinearEquityMarket:
        return make_simple_linear_market(
            s=float(x["S"]),
            r=float(x["r"]),
            q=float(x["q"]),
        )

    @classmethod
    def _make_params(cls, row: pd.Series) -> HestonParams:
        return HestonParams(*(float(row[col]) for col in cls.PARAM_COLS))

    def test_heston_price_cui_against_quantlib(self) -> None:
        sample = self.ql_data
        params = self._make_params(sample.iloc[0])
        market = self._make_market(sample.iloc[0])
        strikes = sample["K"].to_numpy()
        maturities = sample["tau"].to_numpy(dtype=float)

        expected_prices = sample["price"].to_numpy()
        prices = heston_price_cui(params, market, strikes, maturities)

        abs_error = np.abs(prices - expected_prices)
        rel_error = abs_error / expected_prices
        idx_small_value = expected_prices < 0.01

        self.assertLessEqual(float(np.max(abs_error)), 1e-5)
        self.assertLessEqual(float(np.max(rel_error[~idx_small_value])), 1e-3)

    def test_heston_delta_matches_finite_difference(self) -> None:
        params = HestonParams(
            kappa=1.5,
            v_inf=0.04,
            vol_of_vol=0.5,
            rho=-0.6,
            v0=0.04,
        )
        spot = 100.0
        strike = np.array([100.0], dtype=float)
        maturity = np.array([0.5], dtype=float)
        r = 0.05
        q = 0.02
        market = make_simple_linear_market(s=spot, r=r, q=q)
        rel_bump = 1e-4

        dll_delta = float(heston_delta(params, market, strike, maturity))

        market_up = make_simple_linear_market(s=spot * (1 + rel_bump), r=r, q=q)
        market_dn = make_simple_linear_market(s=spot * (1 - rel_bump), r=r, q=q)

        price_up = float(heston_price_cui(params, market_up, strike, maturity))
        price_dn = float(heston_price_cui(params, market_dn, strike, maturity))
        fd_delta = (price_up - price_dn) / (2 * spot * rel_bump)

        self.assertAlmostEqual(dll_delta, fd_delta, places=4)

    def test_heston_jacobian_against_finite_difference(self) -> None:
        params = HestonParams(
            kappa=1.5,
            v_inf=0.04,
            vol_of_vol=0.5,
            rho=-0.6,
            v0=0.04,
        )
        spot = 100.0
        r = 0.05
        q = 0.02
        market = make_simple_linear_market(s=spot, r=r, q=q)

        strikes = np.array([90.0, 100.0, 110.0], dtype=float)
        maturities = np.full(strikes.shape, 0.25, dtype=float)

        rel_bump = 1e-4
        analytic_jac = heston_jacobian(params, market, strikes, maturities)

        base_values = np.array(
            [params.kappa, params.v_inf, params.vol_of_vol, params.rho, params.v0],
            dtype=float,
        )

        for idx in range(base_values.size):
            plus = base_values.copy()
            minus = base_values.copy()
            plus[idx] *= 1 + rel_bump
            minus[idx] *= 1 - rel_bump

            price_plus = heston_price_cui(HestonParams(*plus), market, strikes, maturities)
            price_minus = heston_price_cui(HestonParams(*minus), market, strikes, maturities)
            h = base_values[idx] * rel_bump if base_values[idx] != 0.0 else rel_bump
            numeric_grad = (price_plus - price_minus) / (2 * h)

            assert_allclose(analytic_jac[:, idx], numeric_grad, rtol=5e-3, atol=5e-2)

    def test_heston_calibrator_recovers_parameters(self) -> None:
        params_true = HestonParams(
            kappa=1.5,
            v_inf=0.04,
            vol_of_vol=0.5,
            rho=-0.6,
            v0=0.04,
        )
        spot = 100.0
        r = 0.05
        q = 0.02
        market = make_simple_linear_market(s=spot, r=r, q=q)

        x0 = HestonParams(
            kappa=params_true.kappa * 0.9,
            v_inf=params_true.v_inf * 1.1,
            vol_of_vol=params_true.vol_of_vol * 0.9,
            rho=params_true.rho * 0.9,
            v0=params_true.v0 * 1.1,
        )

        strikes = np.linspace(80.0, 120.0, 50)
        maturities = np.full(strikes.shape, 1, dtype=float)
        prices = heston_price_cui(params_true, market, strikes, maturities)

        par_opt, stats = heston_calibrator(x0, market, prices, strikes, maturities)

        self.assertLess(stats["error_l2_final"], stats["error_l2_initial"])
        self.assertIn(stats["stop_code"], {1, 2, 6})
        assert_allclose(
            [
                par_opt.kappa,
                par_opt.v_inf,
                par_opt.vol_of_vol,
                par_opt.rho,
                par_opt.v0,
            ],
            [
                params_true.kappa,
                params_true.v_inf,
                params_true.vol_of_vol,
                params_true.rho,
                params_true.v0,
            ],
            rtol=0.01,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
