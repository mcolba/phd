"""Unit tests for Black model functions from letsberational package."""

import unittest
import math
from letsberational import black_price, implied_black_vol


class TestImpliedVolFunctions(unittest.TestCase):
    """Unit tests for the Black model price and implied volatility functions."""

    def test_black_price_call(self):
        """Test Black price calculation for a call option."""
        forward = 105.0
        strike = 100.0
        sigma = 0.2
        maturity = 1.0
        option_type = 1.0

        price = black_price(f=forward, k=strike, sigma=sigma, t=maturity, option_type=option_type)
        expected = 10.90558

        self.assertAlmostEqual(price, expected, places=4)

    def test_black_price_put(self):
        """Test Black price calculation for a put option."""
        forward = 105.0
        strike = 100.0
        sigma = 0.2
        maturity = 1.0
        option_type = -1

        price = black_price(f=forward, k=strike, sigma=sigma, t=maturity, option_type=option_type)
        expected = 5.90558

        self.assertAlmostEqual(price, expected, places=4)

    def test_implied_vol_call(self):
        """Test that implied volatility calculation retrieves original volatility for call."""
        forward = 105.0
        strike = 100.0
        original_sigma = 0.25
        maturity = 0.5
        option_type = 1.0

        # Step 1: Calculate option price using original volatility
        price = black_price(f=forward, k=strike, sigma=original_sigma, t=maturity, option_type=option_type)

        # Step 2: Retrieve implied volatility from the calculated price
        implied_sigma = implied_black_vol(p=price, f=forward, k=strike, t=maturity, option_type=option_type)

        # Step 3: Verify the implied volatility matches the original
        self.assertAlmostEqual(
            implied_sigma, original_sigma, places=6, msg="Implied volatility should match original volatility"
        )

    def test_implied_vol_put(self):
        """Test that implied volatility calculation retrieves original volatility for put."""
        forward = 105.0
        strike = 100.0
        original_sigma = 0.25
        maturity = 0.5
        option_type = -1.0

        # Step 1: Calculate option price using original volatility
        price = black_price(f=forward, k=strike, sigma=original_sigma, t=maturity, option_type=option_type)

        # Step 2: Retrieve implied volatility from the calculated price
        implied_sigma = implied_black_vol(p=price, f=forward, k=strike, t=maturity, option_type=option_type)

        # Step 3: Verify the implied volatility matches the original
        self.assertAlmostEqual(
            implied_sigma, original_sigma, places=6, msg="Implied volatility should match original volatility"
        )


if __name__ == "__main__":
    unittest.main()
