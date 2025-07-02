"""Unit tests for Black model functions from letsberational package."""

import unittest
import math
from letsberational import black_price, implied_black_vol


class TestBlackFunctions(unittest.TestCase):
    """Unit tests for the Black model price and implied volatility functions."""

    def test_black_price_call_option(self):
        """Test Black price calculation for a call option."""
        forward = 100.0  # forward price
        strike = 100.0  # strike price (ATM)
        sigma = 0.2  # 20% volatility
        maturity = 1.0  # 1 year to expiry
        option_type = 1.0  # call option

        price = black_price(forward, strike, sigma, maturity, option_type)

        # For ATM call, price should be approximately F * N(sigma*sqrt(T)/2) * 2
        # where N is the cumulative normal distribution
        expected_price = forward * 2 * (1 / math.sqrt(2 * math.pi)) * math.sqrt(maturity) * sigma / 2

        # Price should be positive and reasonable for ATM call
        self.assertGreater(price, 0)
        self.assertLess(price, forward)  # Call price should be less than forward for ATM

    def test_black_price_put_option(self):
        """Test Black price calculation for a put option."""
        forward = 100.0  # forward price
        strike = 100.0  # strike price (ATM)
        sigma = 0.2  # 20% volatility
        maturity = 1.0  # 1 year to expiry
        option_type = -1.0  # put option

        price = black_price(forward, strike, sigma, maturity, option_type)

        # Price should be positive and reasonable for ATM put
        self.assertGreater(price, 0)
        self.assertLess(price, strike)  # Put price should be less than strike for ATM

    def test_implied_vol_roundtrip_call(self):
        """Test that implied volatility calculation retrieves original volatility for call."""
        forward = 100.0  # forward price
        strike = 105.0  # strike price (OTM call)
        original_sigma = 0.25  # 25% volatility
        maturity = 0.5  # 6 months to expiry
        option_type = 1.0  # call option

        # Step 1: Calculate option price using original volatility
        price = black_price(forward, strike, original_sigma, maturity, option_type)

        # Step 2: Retrieve implied volatility from the calculated price
        implied_sigma = implied_black_vol(price, forward, strike, maturity, option_type)

        # Step 3: Verify the implied volatility matches the original
        self.assertAlmostEqual(
            implied_sigma, original_sigma, places=6, msg="Implied volatility should match original volatility"
        )

    def test_implied_vol_roundtrip_put(self):
        """Test that implied volatility calculation retrieves original volatility for put."""
        forward = 100.0  # forward price
        strike = 95.0  # strike price (OTM put)
        original_sigma = 0.3  # 30% volatility
        maturity = 0.25  # 3 months to expiry
        option_type = -1.0  # put option

        # Step 1: Calculate option price using original volatility
        price = black_price(forward, strike, original_sigma, maturity, option_type)

        # Step 2: Retrieve implied volatility from the calculated price
        implied_sigma = implied_black_vol(price, forward, strike, maturity, option_type)

        # Step 3: Verify the implied volatility matches the original
        self.assertAlmostEqual(
            implied_sigma, original_sigma, places=6, msg="Implied volatility should match original volatility"
        )

    def test_implied_vol_roundtrip_atm(self):
        """Test implied volatility roundtrip for at-the-money options."""
        forward = 100.0  # forward price
        strike = 100.0  # strike price (ATM)
        original_sigma = 0.2  # 20% volatility
        maturity = 1.0  # 1 year to expiry
        option_type = 1.0  # call option

        # Step 1: Calculate option price using original volatility
        price = black_price(forward, strike, original_sigma, maturity, option_type)

        # Step 2: Retrieve implied volatility from the calculated price
        implied_sigma = implied_black_vol(price, forward, strike, maturity, option_type)

        # Step 3: Verify the roundtrip works perfectly
        self.assertAlmostEqual(
            implied_sigma, original_sigma, places=8, msg="ATM implied volatility roundtrip should be very accurate"
        )

    def test_multiple_strikes_roundtrip(self):
        """Test implied volatility roundtrip across multiple strikes."""
        forward = 100.0  # forward price
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]  # Various strikes
        original_sigma = 0.2  # 20% volatility
        maturity = 1.0  # 1 year to expiry
        option_type = 1.0  # call option

        for strike in strikes:
            with self.subTest(strike=strike):
                # Step 1: Calculate option price
                price = black_price(forward, strike, original_sigma, maturity, option_type)

                # Step 2: Retrieve implied volatility
                implied_sigma = implied_black_vol(price, forward, strike, maturity, option_type)

                # Step 3: Verify roundtrip accuracy
                self.assertAlmostEqual(
                    implied_sigma, original_sigma, places=6, msg=f"Roundtrip failed for strike {strike}"
                )

    def test_different_maturities_roundtrip(self):
        """Test implied volatility roundtrip across different maturities."""
        forward = 100.0  # forward price
        strike = 100.0  # strike price (ATM)
        original_sigma = 0.25  # 25% volatility
        maturities = [0.1, 0.25, 0.5, 1.0, 2.0]  # Various maturities
        option_type = 1.0  # call option

        for maturity in maturities:
            with self.subTest(maturity=maturity):
                # Step 1: Calculate option price
                price = black_price(forward, strike, original_sigma, maturity, option_type)

                # Step 2: Retrieve implied volatility
                implied_sigma = implied_black_vol(price, forward, strike, maturity, option_type)

                # Step 3: Verify roundtrip accuracy
                self.assertAlmostEqual(
                    implied_sigma, original_sigma, places=6, msg=f"Roundtrip failed for maturity {maturity}"
                )


if __name__ == "__main__":
    unittest.main()
