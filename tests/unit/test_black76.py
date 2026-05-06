"""Data-driven unit tests for Black76 using the provided CSV dataset.

The dataset is treated as the source of truth for expected values.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_risk.models.black76 import (
    black76_fwd_delta,
    black76_price,
    black76_undisc_fwd_delta,
    black76_undisc_fwd_delta_to_strike,
    black76_vega,
    bsm_price,
    bsm_spot_delta,
    implied_black_vol,
    implied_vol_simple,
)


def _load_df() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[1] / "data" / "vanilla_opt.csv"
    data = pd.read_csv(path)
    data["is_call"] = data["type"].map({"C": True, "P": False})
    return data


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _load_df()


def test_black76_price(df: pd.DataFrame) -> None:
    expected = df.price
    result = black76_price(
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        sigma=df.sigma,
        disc=df.DF,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_bsm_price(df: pd.DataFrame) -> None:
    expected = df.price
    result = bsm_price(
        spot=df.S,
        strike=df.K,
        tau=df.tau,
        sigma=df.sigma,
        r=df.r,
        q=df.q,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_black76_fwd_delta(df: pd.DataFrame) -> None:
    adj = np.exp((df.q - df.r) * df.tau)
    expected = df.delta * adj
    undisc_delta = black76_undisc_fwd_delta(
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        sigma=df.sigma,
        is_call=df.is_call,
    )
    result = black76_fwd_delta(
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        sigma=df.sigma,
        disc=df.DF,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, df.DF * undisc_delta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_black76_undisc_fwd_delta(df: pd.DataFrame) -> None:
    disc = np.exp(-df.r * df.tau)
    adj = np.exp((df.q - df.r) * df.tau)
    expected = df.delta * adj / disc
    result = black76_undisc_fwd_delta(
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        sigma=df.sigma,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_bsm_spot_delta(df: pd.DataFrame) -> None:
    expected = df.delta
    result = bsm_spot_delta(
        spot=df.S,
        strike=df.K,
        tau=df.tau,
        r=df.r,
        q=df.q,
        sigma=df.sigma,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_black76_vega(df: pd.DataFrame) -> None:
    # Dataset vega is per 1 vol point (i.e., 0.01 sigma).
    expected = df.vega
    result = 0.01 * black76_vega(
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        sigma=df.sigma,
        disc=df.DF,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_implied_vol_newton(df: pd.DataFrame) -> None:
    expected = df.sigma
    result = implied_vol_simple(
        price=df.price,
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        disc=df.DF,
        is_call=df.is_call,
        x0=np.full_like(df.price, 0.2),
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_implied_vol_scalar(df: pd.DataFrame) -> None:
    for row in df.itertuples(index=False):
        iv = implied_black_vol(
            price=row.price,
            fwd=row.F,
            strike=row.K,
            tau=row.tau,
            disc=row.DF,
            is_call=row.is_call,
        )
        assert isinstance(iv, np.ndarray)
        assert iv.shape == ()
        np.testing.assert_allclose(
            iv,
            row.sigma,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"row id={row.id}",
        )


def test_implied_vol_vector(df: pd.DataFrame) -> None:
    expected = df.sigma
    result = implied_black_vol(
        price=df.price,
        fwd=df.F,
        strike=df.K,
        tau=df.tau,
        disc=df.DF,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)


def test_delta_to_strike(df: pd.DataFrame) -> None:
    spot_to_fwd_adj = np.exp((df.q - df.r) * df.tau)
    disc = np.exp(-df.r * df.tau)
    undisc_fwd_delta = df.delta * spot_to_fwd_adj / disc
    expected = df.K
    result = black76_undisc_fwd_delta_to_strike(
        delta=undisc_fwd_delta,
        fwd=df.F,
        tau=df.tau,
        sigma=df.sigma,
        is_call=df.is_call,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
