"""Model-agnostic Black-Scholes control-variate builders for Fourier engines.

Implements the Joshi & Yang (2011) Black-Scholes control variate: subtract the
BS characteristic function from a model CF inside the Carr-Madan Fourier
integral and add the closed-form BS call price back at the end. With a
variance-matched BS volatility the difference CF decays two orders faster
than the bare model CF (Joshi & Yang 2011, Thm 3.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from vol_risk.models.black76 import black76_price
from vol_risk.models.numerical.fourier.base import CallControlVariate
from vol_risk.models.numerical.fourier.fft_jax import JaxControlVariate

if TYPE_CHECKING:
    from collections.abc import Callable


def make_bs_cf(
    spot: float,
    r: float,
    q: float,
    tau: float,
    sigma: float,
    *,
    xp: Any = np,
) -> Callable[[Any], Any]:
    r"""Return the Black-Scholes log-stock characteristic function on ``xp``."""
    drift = float(np.log(spot)) + (r - q - 0.5 * sigma * sigma) * tau
    half_var = 0.5 * sigma * sigma * tau

    def cf(u: Any) -> Any:
        return xp.exp(1j * u * drift - half_var * u * u)

    return cf


def make_bs_call_price_np(
    spot: float, r: float, q: float, tau: float, sigma: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a NumPy Black-76 call pricer for the BS control leg."""
    fwd = spot * np.exp((r - q) * tau)
    disc = float(np.exp(-r * tau))

    def call_price(strike: np.ndarray) -> np.ndarray:
        k = np.asarray(strike, dtype=float)
        return np.asarray(
            black76_price(fwd=fwd, strike=k, tau=tau, sigma=sigma, disc=disc, is_call=True),
            dtype=float,
        ).reshape(k.shape)

    return call_price


def make_bs_call_price_jax(spot: float, r: float, q: float, tau: float, sigma: float) -> Callable[[Any], Any]:
    """Return a JAX Black-Scholes call pricer for the BS control leg."""
    import jax.numpy as jnp
    from jax.scipy.stats import norm

    fwd = float(spot * np.exp((r - q) * tau))
    disc = float(np.exp(-r * tau))
    sqrt_t = float(np.sqrt(tau))
    sig_sqrt_t = sigma * sqrt_t

    def call_price(strike: Any) -> Any:
        k = jnp.asarray(strike)
        d1 = (jnp.log(fwd / k) + 0.5 * sigma * sigma * tau) / sig_sqrt_t
        d2 = d1 - sig_sqrt_t
        return disc * (fwd * norm.cdf(d1) - k * norm.cdf(d2))

    return call_price


def make_bs_control(
    spot: float, r: float, q: float, tau: float, sigma: float, backend: str = "np"
) -> CallControlVariate:
    """Return a NumPy :class:`CallControlVariate` for variance ``sigma``."""
    xp = np if backend == "np" else jnp
    call_price = (
        make_bs_call_price_np(spot, r, q, tau, sigma)
        if backend == "np"
        else make_bs_call_price_jax(spot, r, q, tau, sigma)
    )
    return CallControlVariate(
        cf=make_bs_cf(spot, r, q, tau, sigma, xp=xp),
        call_price=call_price,
    )


__all__ = [
    "make_bs_call_price_jax",
    "make_bs_call_price_np",
    "make_bs_cf",
    "make_bs_control",
]
