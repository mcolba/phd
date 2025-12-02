import warnings

import letsberational
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import bisect, newton
from scipy.stats import norm


def _gaussian_density(x: float) -> float:
    """Standard normal probability density function."""
    return np.exp(-x * x / 2.0) / np.sqrt(2 * np.pi)


def black76_price(
    df: ArrayLike,
    f: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Black 76 pricing function.

    Args:
        df: Discount factor
        f: Forward
        k: Strike
        t: Time to maturity (year fraction)
        sigma: Volatility
        is_call: call/put flag

    Returns: Contract price
    """
    df, f, k, t, sigma, is_call = map(np.asarray, (df, f, k, t, sigma, is_call))
    sign = np.array([1.0 if x else -1.0 for x in is_call])

    d1 = (np.log(f / k) + (sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))

    return df * sign * (f * norm.cdf(sign * d1) - k * norm.cdf(sign * d2))


def black76_vega(
    df: ArrayLike,
    f: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 vega for european options."""
    df, f, k, t, sigma = map(np.asarray, (df, f, k, t, sigma))

    d1 = (np.log(f / k) + (sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    return df * f * _gaussian_density(d1) * np.sqrt(t)


def black76_fwd_delta(
    f: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 delta for European options."""
    f, k, t, r, sigma, is_call = map(np.asarray, (f, k, t, r, sigma, is_call))

    d1 = (np.log(f / k) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    df = np.exp(-r * t)

    return np.where(is_call, df * norm.cdf(d1), -df * norm.cdf(-d1))


def black76_fwd_delta_to_strike(
    delta: ArrayLike,
    f: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 delta for European options."""
    f, t, r, sigma, is_call = map(np.asarray, (f, t, r, sigma, is_call))

    df = np.exp(-r * t)
    total_vol = sigma * np.sqrt(t)

    return np.where(
        is_call,
        f * np.exp(-total_vol * norm.ppf(delta / df) + 0.5 * total_vol**2),
        f * np.exp(total_vol * norm.ppf(-delta / df) + 0.5 * total_vol**2),
    )


def implied_vol_simple(
    df: float,
    f: float,
    k: float,
    t: float,
    p: float,
    is_call: bool,
    x0: float = 0.3,
) -> float:
    try:
        return newton(
            func=lambda x: black76_price(df, f, k, t, x, is_call) - p,
            fprime=lambda x: black76_vega(df, f, k, t, x),
            x0=x0,
            tol=1e-12,
            rtol=1e-10,
            maxiter=200,
        )
    except Exception as e:
        warnings.warn(
            f"Newton-Raphson did not find a root because of the following exception occurred: {e}. "
            f"Trying bisection next...",
            stacklevel=2,
        )
        try:
            return bisect(
                f=lambda x: black76_price(df, f, k, t, x, is_call) - p,
                a=0.00001,
                b=3,
                xtol=1e-12,
                rtol=1e-10,
            )
        except Exception as e:
            msg = f"Bisection did not find a root because of the following exception occurred: {e}."
            warnings.warn(msg, stacklevel=2)
            return None


def implied_vol_jackel(
    price: float,
    f: float,
    k: float,
    t: float,
    df: float,
    is_call: bool,
) -> float:
    """Implied volatility using Jaeckel's method (letsberational)."""
    theta = 1.0 if is_call else -1.0
    return letsberational.implied_black_vol(p=price / df, f=f, k=k, t=t, option_type=theta)


def bsm_price(
    s: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Black-Scholes-Merton price using Black-76 formula."""
    fwd = s * np.exp((r - q) * t)
    disc = np.exp(-r * t)
    return black76_price(
        df=disc,
        f=fwd,
        k=k,
        t=t,
        sigma=sigma,
        is_call=is_call,
    )


def bsm_spot_delta(
    s: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Black-Scholes-Merton price using Black-76 formula."""
    adj = np.exp((r - q) * t)
    fwd = s * np.exp((r - q) * t)
    return adj * black76_fwd_delta(
        f=fwd,
        k=k,
        t=t,
        r=r,
        sigma=sigma,
        is_call=is_call,
    )
