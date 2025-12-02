"""Data-driven unit tests for Black76 using the provided CSV dataset.

The dataset is treated as the source of truth for expected values.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vol_risk.models.black76 import (
    black76_fwd_delta,
    black76_fwd_delta_to_strike,
    black76_price,
    black76_vega,
    bsm_price,
    bsm_spot_delta,
    implied_vol_jackel,
    implied_vol_simple,
)


class TestBlack76(unittest.TestCase):
    """Validate Black76 and BSM functions against the CSV dataset."""

    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "data" / "vanilla_opt.csv"
        df = pd.read_csv(path)
        df["is_call"] = df["type"].map({"C": True, "P": False})
        cls.df = df

    def test_black76_price(self) -> None:
        expected = self.df.price
        result = black76_price(
            df=self.df.DF,
            f=self.df.F,
            k=self.df.K,
            t=self.df.tau,
            sigma=self.df.sigma,
            is_call=self.df.is_call,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_bsm_price(self) -> None:
        expected = self.df.price
        result = bsm_price(
            s=self.df.S,
            k=self.df.K,
            t=self.df.tau,
            sigma=self.df.sigma,
            r=self.df.r,
            q=self.df.q,
            is_call=self.df.is_call,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_black76_fwd_delta(self) -> None:
        adj = np.exp((self.df.q - self.df.r) * self.df.tau)
        expected = self.df.delta * adj
        result = black76_fwd_delta(
            f=self.df.F,
            k=self.df.K,
            t=self.df.tau,
            r=self.df.r,
            sigma=self.df.sigma,
            is_call=self.df.is_call,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_bsm_spot_delta(self) -> None:
        expected = self.df.delta
        result = bsm_spot_delta(
            s=self.df.S,
            k=self.df.K,
            t=self.df.tau,
            r=self.df.r,
            q=self.df.q,
            sigma=self.df.sigma,
            is_call=self.df.is_call,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_black76_vega(self) -> None:
        # Dataset vega is per 1 vol point (i.e., 0.01 sigma).
        expected = self.df.vega
        result = 0.01 * black76_vega(
            df=self.df.DF,
            f=self.df.F,
            k=self.df.K,
            t=self.df.tau,
            sigma=self.df.sigma,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)

    def test_implied_vol_newton(self) -> None:
        expected = self.df.sigma
        result = implied_vol_simple(
            df=self.df.DF,
            f=self.df.F,
            k=self.df.K,
            t=self.df.tau,
            p=self.df.price,
            is_call=self.df.is_call,
            x0=np.full_like(self.df.price, 0.2),
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_implied_vol_jaeckel(self) -> None:
        for row in self.df.itertuples(index=False):
            iv = implied_vol_jackel(
                price=row.price,
                f=row.F,
                k=row.K,
                t=row.tau,
                df=row.DF,
                is_call=row.is_call,
            )
            np.testing.assert_allclose(
                iv,
                row.sigma,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"row id={row.id}",
            )

    def test_delta(self) -> None:
        adj = np.exp((self.df.q - self.df.r) * self.df.tau)
        expected = self.df.K
        result = black76_fwd_delta_to_strike(
            delta=self.df.delta * adj,
            f=self.df.F,
            t=self.df.tau,
            r=self.df.r,
            sigma=self.df.sigma,
            is_call=self.df.is_call,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
