"""Black-Scholes control variates for Fourier engines (Joshi & Yang, 2011).

Defines the control-variate contracts consumed by the quadrature and FFT engines and
shared builders that inject NumPy or JAX through the array namespace ``xp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.special import ndtr

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from vol_risk.models.numerical.fourier.fft import JaxCallPriceFunction, JaxCharacteristicFunction
    from vol_risk.models.numerical.fourier.quad import CallPriceFunction, CharacteristicFunction


Numeric = float | np.ndarray | jax.Array


@dataclass(frozen=True, slots=True)
class CallControlVariate:
    """Control variate for a discounted Fourier call engine.

    ``cf`` and ``call_price`` must share spot, rates and maturity with the model being
    priced: equal forwards cancel the Carr-Madan pole at ``v = i*damping``, which is what
    makes small damping usable.
    """

    cf: CharacteristicFunction
    call_price: CallPriceFunction


@dataclass(frozen=True, slots=True)
class JaxCallControlVariate:
    """JAX control variate for a discounted Fourier call engine; see :class:`CallControlVariate`."""

    cf: JaxCharacteristicFunction
    call_price: JaxCallPriceFunction


def make_bs_cf(
    spot: Numeric,
    r: Numeric,
    q: Numeric,
    tau: Numeric,
    sigma: Numeric,
    *,
    xp: ModuleType = np,
) -> Callable[[Numeric | complex], np.ndarray | jax.Array]:
    """Return a Black-Scholes log-stock CF using NumPy or traceable JAX operations."""
    drift = xp.log(spot) + (r - q - 0.5 * sigma * sigma) * tau
    half_var = 0.5 * sigma * sigma * tau

    def cf(u: Numeric | complex) -> np.ndarray | jax.Array:
        u = xp.asarray(u)
        return xp.asarray(xp.exp(1j * u * drift - half_var * u * u))

    return cf


def make_bs_call_price(
    spot: Numeric,
    r: Numeric,
    q: Numeric,
    tau: Numeric,
    sigma: Numeric,
    *,
    xp: ModuleType = np,
) -> Callable[[Numeric], np.ndarray | jax.Array]:
    """Return a Black-Scholes call pricer using NumPy/SciPy or traceable JAX operations."""
    cdf = norm.cdf if xp is jnp else ndtr

    fwd = spot * xp.exp((r - q) * tau)
    disc = xp.exp(-r * tau)
    half_var = 0.5 * sigma * sigma * tau
    sig_sqrt_t = sigma * xp.sqrt(tau)

    def call_price(strike: Numeric) -> np.ndarray | jax.Array:
        k = xp.asarray(strike)
        d1 = (xp.log(fwd / k) + half_var) / sig_sqrt_t
        d2 = d1 - sig_sqrt_t
        return xp.asarray(disc * (fwd * cdf(d1) - k * cdf(d2)))

    return call_price


def make_bs_control_np(spot: Numeric, r: Numeric, q: Numeric, tau: Numeric, sigma: Numeric) -> CallControlVariate:
    """Return a NumPy :class:`CallControlVariate` for volatility ``sigma``."""
    return CallControlVariate(
        cf=make_bs_cf(spot, r, q, tau, sigma),
        call_price=make_bs_call_price(spot, r, q, tau, sigma),
    )


def make_bs_control_jax(spot: Numeric, r: Numeric, q: Numeric, tau: Numeric, sigma: Numeric) -> JaxCallControlVariate:
    """Return a traceable :class:`JaxCallControlVariate` for volatility ``sigma``."""
    return JaxCallControlVariate(
        cf=cast("JaxCharacteristicFunction", make_bs_cf(spot, r, q, tau, sigma, xp=jnp)),
        call_price=cast("JaxCallPriceFunction", make_bs_call_price(spot, r, q, tau, sigma, xp=jnp)),
    )
