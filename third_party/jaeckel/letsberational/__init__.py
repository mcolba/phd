from . import _letsberational


def black_price(f: float, k: float, sigma: float, t: float, option_type: float) -> float:
    """Compute the undiscounted option price using the Black (1976) model.

    This function calculates the theoretical price of an option under the Black (1976) model,
    which is commonly used for pricing options on futures and forwards.

    Args:
        f (float): Forward price of the underlying asset.
        k (float): Strike price of the option.
        sigma (float): Volatility of the underlying asset (annualized).
        t (float): Time to expiry (in years).
        option_type (float): Option type parameter. For calls: 1.0, for puts: -1.0.

    Returns:
        float: The theoretical option price under the Black model.
    """
    return _letsberational.Black(f, k, sigma, t, option_type)


def implied_black_vol(p: float, f: float, k: float, t: float, option_type: float) -> float:
    """Compute the implied volatility under the Black (1976) model.

    This function numerically inverts the Black formula to determine the volatility (`sigma`)
    that would result in the observed (undiscounted) option price.

    Args:
        p (float): Observed undiscounted option price.
        f (float): Forward price of the underlying asset.
        k (float): Strike price of the option.
        t (float): Time to expiry (in years).
        option_type (float): Option type parameter. For calls: 1.0, for puts: -1.0.

    Returns:
        float: The implied volatility (sigma) that solves the Black equation.
    """
    return _letsberational.ImpliedBlackVolatility(p, f, k, t, option_type)
