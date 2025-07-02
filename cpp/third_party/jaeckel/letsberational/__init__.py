from . import _letsberational


def black_price(forward: float, strike: float, sigma: float, maturity: float, option_type: float) -> float:
    """Compute the option price using the Black (1976) model.

    This function calculates the theoretical price of an option under the Black (1976) model,
    which is commonly used for pricing options on futures and forwards.

    Args:
        forward (float): Forward price of the underlying asset.
        strike (float): Strike price of the option.
        sigma (float): Volatility of the underlying asset (annualized).
        maturity (float): Time to expiry (in years).
        option_type (float): Option type parameter.
            - For calls: option_type = 1.0
            - For puts: option_type = -1.0

    Returns:
        float: The theoretical option price under the Black model.

    Raises:
        RuntimeError: If the calculation encounters invalid parameters.
    """
    return _letsberational.Black(forward, strike, sigma, maturity, option_type)


def implied_black_vol(price: float, forward: float, strike: float, maturity: float, option_type: float) -> float:
    """Compute the implied volatility under the Black (1976) model.

    This function numerically inverts the Black formula to determine the volatility (`sigma`)
    that would result in the observed option price.

    Args:
        price (float): Observed option price.
        forward (float): Forward price of the underlying asset.
        strike (float): Strike price of the option.
        maturity (float): Time to expiry (in years).
        option_type (float): Option type parameter.
            - For calls: option_type = 1.0
            - For puts: option_type = -1.0

    Returns:
        float: The implied volatility (sigma) that solves the Black equation.

    Raises:
        RuntimeError: If the implied volatility cannot be determined (e.g. price out of bounds).
    """
    return _letsberational.ImpliedBlackVolatility(price, forward, strike, maturity, option_type)
