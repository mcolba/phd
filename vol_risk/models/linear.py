from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class LinearEquityMarket:
    """Linear model for equity pricing with forward price calculation."""

    spot: float
    disc_curve: Callable
    cont_dvd_curve: Callable

    def fwd(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the forward price."""
        return self.spot * np.exp(-self.cont_dvd_curve(tau) * tau) / self.disc_curve(tau)

    def df(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the discount factor."""
        return self.disc_curve(tau)

    def zero_rate(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the zero rate."""
        return -np.log(self.disc_curve(tau)) / tau

    def zero_dvd_yield(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the zero dividend yield."""
        return -np.log(self.cont_dvd_curve(tau)) / tau


def make_raw_disc_curve(
    tau: np.ndarray,
    r: np.ndarray,
    add_zero_anchor: bool = True,
) -> Callable[[ArrayLike], ArrayLike]:
    """Create a discount curve using flat forward (linear) extrapolation.

    Parameters:
    - tau: array of maturities (must be increasing)
    - r: array of zero rates
    - add_zero_anchor: add a zero point at tau = 0 if True

    Returns:
    - discount(t): function returning discount factor at time t (scalar or array)
    """
    tau = np.squeeze(np.asarray(tau))
    r = np.squeeze(np.asarray(r))

    if tau.ndim != 1 or r.ndim != 1:
        msg = "tau and r must be 1-dimensional arrays."
        raise ValueError(msg)
    if tau.size != r.size:
        msg = "tau and r must have the same length."
        raise ValueError(msg)
    if np.any(np.diff(tau) <= 0):
        msg = "tau must be strictly increasing."
        raise ValueError(msg)
    if tau.size < 2:
        msg = "Need at least two points for interpolation and extrapolation."
        raise ValueError(msg)

    # Interpolate in r * t = -log D(t)
    rt = tau * r

    if add_zero_anchor and tau[0] > 0:
        tau = np.insert(arr=tau, obj=0, values=0.0)
        rt = np.insert(arr=rt, obj=0, values=0.0)

    # Create interpolator with linear extrapolation
    interp_rt = interp1d(tau, rt, kind="linear", fill_value="extrapolate", assume_sorted=True)

    def _disc(x: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        rt_t = interp_rt(x)
        disc = np.exp(-rt_t)
        return float(disc) if np.ndim(x) == 0 else disc

    return _disc


def make_simple_linear_market(s: float = 100.0, r: float = 0, q: float = 0) -> LinearEquityMarket:
    """Creates a dummy linear market data object."""
    return LinearEquityMarket(
        spot=s,
        disc_curve=lambda tau: np.exp(-r * tau),
        cont_dvd_curve=lambda tau: q,
    )
