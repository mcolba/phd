"""Unit tests for Black model functions from letsberational package."""

import pytest
from letsberational import black_price, implied_black_vol


def _roundtrip_implied_vol(
    *,
    forward: float,
    strike: float,
    original_sigma: float,
    maturity: float,
    option_type: float,
) -> float:
    """Calculate the option price using the original volatility and return the implied volatility."""
    price = black_price(f=forward, k=strike, sigma=original_sigma, t=maturity, option_type=option_type)
    return implied_black_vol(p=price, f=forward, k=strike, t=maturity, option_type=option_type)


@pytest.mark.parametrize(
    ("option_type", "expected"),
    [
        pytest.param(1.0, 10.90558, id="call"),
        pytest.param(-1.0, 5.90558, id="put"),
    ],
)
def test_black_price(option_type: float, expected: float) -> None:
    """Test Black price calculation for calls and puts."""
    forward = 105.0
    strike = 100.0
    sigma = 0.2
    maturity = 1.0

    price = black_price(f=forward, k=strike, sigma=sigma, t=maturity, option_type=option_type)

    assert price == pytest.approx(expected, abs=0.5 * 10**-4)


@pytest.mark.parametrize(
    ("option_type",),
    [
        pytest.param(1.0, id="call"),
        pytest.param(-1.0, id="put"),
    ],
)
def test_implied_vol_roundtrip(option_type: float) -> None:
    """Test that implied volatility calculation retrieves original volatility."""
    forward = 105.0
    strike = 100.0
    original_sigma = 0.25
    maturity = 0.5

    implied_sigma = _roundtrip_implied_vol(
        forward=forward,
        strike=strike,
        original_sigma=original_sigma,
        maturity=maturity,
        option_type=option_type,
    )

    assert implied_sigma == pytest.approx(
        original_sigma,
        abs=0.5 * 10**-6,
        rel=0.0,
    )
