import warnings

import letsberational
import numpy as np
from numpy.typing import ArrayLike
from scipy import special
from scipy.optimize import bisect, newton


def _gaussian_density(x: ArrayLike) -> ArrayLike:
    """Standard normal probability density function."""
    return np.exp(-x * x / 2.0) / np.sqrt(2 * np.pi)


def black76_price(
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    disc: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Black 76 pricing function.

    Args:
        fwd: Forward
        strike: Strike
        tau: Time to maturity (year fraction)
        sigma: Volatility
        disc: Discount factor
        is_call: call/put flag

    Returns: Contract price
    """
    fwd, strike, tau, sigma, disc, is_call = map(np.atleast_1d, (fwd, strike, tau, sigma, disc, is_call))
    is_call = np.asarray(is_call, dtype=bool)
    sign = 2.0 * is_call.astype(float) - 1.0

    total_vol = sigma * np.sqrt(tau)
    d1 = (np.log(fwd / strike) + 0.5 * sigma**2 * tau) / total_vol
    d2 = d1 - total_vol

    return disc * sign * (fwd * special.ndtr(sign * d1) - strike * special.ndtr(sign * d2))


def black76_vega(
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    disc: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 vega for european options."""
    fwd, strike, tau, sigma, disc = map(np.atleast_1d, (fwd, strike, tau, sigma, disc))

    d1 = (np.log(fwd / strike) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return disc * fwd * _gaussian_density(d1) * np.sqrt(tau)


def black76_fwd_delta(
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    disc: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 delta for European options."""
    fwd, strike, tau, sigma, disc, is_call = map(np.atleast_1d, (fwd, strike, tau, sigma, disc, is_call))
    return disc * black76_undisc_fwd_delta(
        fwd=fwd,
        strike=strike,
        tau=tau,
        sigma=sigma,
        is_call=is_call,
    )


def black76_undisc_fwd_delta(
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 delta for European options."""
    fwd, strike, tau, sigma, is_call = map(np.atleast_1d, (fwd, strike, tau, sigma, is_call))
    is_call = np.asarray(is_call, dtype=bool)
    sign = 2.0 * is_call.astype(float) - 1.0
    d1 = (np.log(fwd / strike) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    return sign * special.ndtr(sign * d1)


def black76_undisc_fwd_delta_to_strike(
    delta: ArrayLike,
    fwd: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Calculate the Black-76 delta for European options."""
    delta, fwd, tau, sigma, is_call = map(np.atleast_1d, (delta, fwd, tau, sigma, is_call))
    is_call = np.asarray(is_call, dtype=bool)
    sign = 2.0 * is_call.astype(float) - 1.0
    total_vol = sigma * np.sqrt(tau)

    return fwd * np.exp(-sign * total_vol * special.ndtri(sign * delta) + 0.5 * total_vol**2)


def implied_vol_simple(
    price: ArrayLike,
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    disc: ArrayLike,
    is_call: ArrayLike,
    x0: ArrayLike = 0.3,
) -> ArrayLike:
    try:
        return newton(
            func=lambda x: black76_price(
                fwd=fwd,
                strike=strike,
                tau=tau,
                sigma=x,
                disc=disc,
                is_call=is_call,
            )
            - price,
            fprime=lambda x: black76_vega(
                fwd=fwd,
                strike=strike,
                tau=tau,
                sigma=x,
                disc=disc,
            ),
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
                f=lambda x: black76_price(
                    fwd=fwd,
                    strike=strike,
                    tau=tau,
                    sigma=x,
                    disc=disc,
                    is_call=is_call,
                )
                - price,
                a=0.00001,
                b=3,
                xtol=1e-12,
                rtol=1e-10,
            )
        except Exception as e:
            msg = f"Bisection did not find a root because of the following exception occurred: {e}."
            warnings.warn(msg, stacklevel=2)
            return None


def _implied_vol_jaeckel_scalar(
    price: float,
    fwd: float,
    strike: float,
    tau: float,
    disc: float,
    is_call: bool,
) -> float:
    """Implied volatility using Jaeckel's method (letsberational)."""
    theta = 1.0 if is_call else -1.0
    return letsberational.implied_black_vol(p=price / disc, f=fwd, k=strike, t=tau, option_type=theta)


def bsm_price(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Black-Scholes-Merton price using Black-76 formula."""
    fwd = spot * np.exp((r - q) * tau)
    disc = np.exp(-r * tau)
    return black76_price(
        fwd=fwd,
        strike=strike,
        tau=tau,
        sigma=sigma,
        disc=disc,
        is_call=is_call,
    )


def bsm_spot_delta(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    sigma: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
) -> ArrayLike:
    """Black-Scholes-Merton price using Black-76 formula."""
    adj = np.exp((r - q) * tau)
    fwd = spot * np.exp((r - q) * tau)
    disc = np.exp(-r * tau)
    return adj * black76_fwd_delta(
        fwd=fwd,
        strike=strike,
        tau=tau,
        sigma=sigma,
        disc=disc,
        is_call=is_call,
    )


def _broadcast_and_flatten(*args, shape: tuple) -> list[np.ndarray]:
    return [np.broadcast_to(np.asarray(x), shape).ravel() for x in args]


def implied_black_vol(
    price: float | ArrayLike,
    fwd: float | ArrayLike,
    strike: float | ArrayLike,
    tau: float | ArrayLike,
    disc: float | ArrayLike,
    is_call: bool | ArrayLike,
) -> np.ndarray:
    """Calculate implied volatilities using Jaeckel's method."""
    price = np.asarray(price, dtype=float)
    flat_price = price.ravel()

    flat_strike, flat_tau, flat_fwd, flat_disc, flat_is_call = _broadcast_and_flatten(
        strike, tau, fwd, disc, is_call, shape=price.shape
    )

    if flat_is_call.dtype != bool:
        msg = "is_call must be a boolean array."
        raise ValueError(msg)

    n = flat_price.size
    iv = np.empty(n, dtype=float)
    for i in range(n):
        iv[i] = _implied_vol_jaeckel_scalar(
            price=flat_price[i],
            fwd=flat_fwd[i],
            strike=flat_strike[i],
            tau=flat_tau[i],
            disc=flat_disc[i],
            is_call=flat_is_call[i],
        )

    return iv.reshape(price.shape)
