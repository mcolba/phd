from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import cache

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import ndtri

from vol_risk.models.black76 import black76_undisc_fwd_delta
from vol_risk.models.linear import LinearEquityMarket


# @dataclass(frozen=True)
class Extrapolator(ABC):
    """Vol extrapolator base class."""

    @abstractmethod
    def __call__(self) -> Callable:
        raise NotImplementedError


# @dataclass(frozen=True)
class FlatExtrapolator(Extrapolator):
    """Flat vol extrapolator."""

    def __init__(self, lower: float = -np.inf, upper: float = np.inf) -> None:
        self._lower = lower
        self._upper = upper

    def __call__(self, f: Callable) -> Callable:
        """Return a decorated function with flat extrapolation."""

        def wrapper(k: np.ndarray) -> np.ndarray:
            k = np.asarray(k)
            return f(np.clip(k, self._lower, self._upper))

        return wrapper


# @dataclass(frozen=True)
class VolSmile:
    """Vol smile object wrapping a VolInterpolator and an extrapolator."""

    def __init__(
        self,
        interpl: Callable,
        extrapl: Extrapolator = lambda x: x,
    ) -> None:
        """Initialize the VolSmile object."""
        self._interpolator = interpl
        self._extrapolator = extrapl

    def vol(self, k: np.ndarray) -> np.ndarray:
        """Get vol for given strikes k and moneyness convention."""
        return self._extrapolator(self._interpolator)(k)


class VolSurface:
    """Vol surface object wrapping multiple VolSmiles slices."""

    def __init__(self, taus: np.array, smiles: Iterable[VolSmile], linear_model: LinearEquityMarket = None) -> None:
        """Initialize the surface with a mapping tau -> VolSmile."""
        self._taus = np.asarray(taus)
        self._smiles = smiles
        self._linear_model = linear_model
        self.__post_init__()

    def __post_init__(self) -> None:
        if self._taus.ndim != 1:
            msg = "taus must be a 1-D array."
            raise ValueError(msg)

        if self._taus.dtype != np.float64:
            msg = "taus must have dtype float64 (double)."
            raise TypeError(msg)

        if not np.all(np.diff(self._taus) > 0.0):
            msg = "taus must be strictly increasing."
            raise ValueError(msg)

        if len(self._smiles) != self._taus.size:
            msg = "Number of smiles must match number of taus."
            raise ValueError(msg)

    def vol(self, k: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Evaluate the surface at strikes and maturities.

        If t is between existing slices, interpolate in total variance. Otherwise, use flat extrapolation in maturity.
        """
        k_arr = np.asarray(k, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        # Scalar maturity: vectorised in strikes.
        if t_arr.ndim == 0:
            return self._vol_at_scalar_maturity(k_arr, float(t_arr))

        if k_arr.shape != t_arr.shape:
            msg = "Shapes of k and t must match when both are arrays."
            raise ValueError(msg)

        vols = np.empty_like(k_arr, dtype=float)
        for i, (ki, ti) in enumerate(zip(k_arr, t_arr, strict=False)):
            vols[i] = float(self._vol_at_scalar_maturity(np.asarray([ki], dtype=float), float(ti))[0])

        return vols

    def atmf_vol(self, t: np.ndarray) -> float:
        """Returns the at-the-money forward vol for given maturities."""
        k = self._linear_model.fwd(t)
        return self.vol(k, t)

    def vol_at_delta(self, delta: ArrayLike, tau: ArrayLike) -> float | np.ndarray:
        """Find strikes whose own surface IV gives the requested Black delta."""
        if self._linear_model is None:
            msg = "linear_model is required for delta inversion."
            raise ValueError(msg)
        delta_arr, tau_arr = np.broadcast_arrays(
            np.asarray(delta, dtype=float),
            np.asarray(tau, dtype=float),
        )
        if np.any(~np.isfinite(delta_arr) | (delta_arr <= 0.0) | (delta_arr >= 1.0)):
            msg = "delta must be finite and strictly between zero and one (forward call delta)."
            raise ValueError(msg)
        if np.any(~np.isfinite(tau_arr) | (tau_arr <= 0.0)):
            msg = "tau must be finite and positive."
            raise ValueError(msg)

        unique_tau, inverse = np.unique(tau_arr.ravel(), return_inverse=True)
        sigma_arr = np.asarray(self.atmf_vol(unique_tau))[inverse].reshape(delta_arr.shape)
        fwd = np.asarray(self._linear_model.fwd(unique_tau), dtype=float)

        if np.any(~np.isfinite(fwd) | (fwd <= 0.0)):
            msg = "Forward prices must be finite and positive for delta inversion."
            raise ValueError(msg)
        if np.any(~np.isfinite(sigma_arr) | (sigma_arr <= 0.0)):
            msg = "The IV starting guess must be finite and positive."
            raise ValueError(msg)

        fwd = fwd[inverse].reshape(delta_arr.shape)

        vols = np.empty_like(delta_arr)
        for index in np.ndindex(delta_arr.shape):
            vols[index] = self._vol_at_scalar_delta(
                delta=float(delta_arr[index]),
                tau=float(tau_arr[index]),
                fwd=float(fwd[index]),
                sigma=float(sigma_arr[index]),
            )
        if vols.ndim == 0:
            return float(vols)
        return vols

    def _vol_at_scalar_delta(self, delta: float, tau: float, fwd: float, sigma: float) -> float:
        """Bracket and solve one delta in log-forward moneyness."""
        sqrt_tau = np.sqrt(tau)
        z = ndtri(delta)
        total_vol = sigma * sqrt_tau
        guess = float((0.5 * total_vol - z) * total_vol)

        @cache
        def evaluate(x: float) -> tuple[float, float]:
            with np.errstate(over="ignore", under="ignore"):
                strike = fwd * np.exp(x)
            if not np.isfinite(strike) or strike <= 0.0:
                msg = "Delta inversion exceeded the finite positive strike range."
                raise ValueError(msg)
            vol = float(self._vol_at_scalar_maturity(np.asarray([strike]), tau)[0])
            s = vol * sqrt_tau
            if not np.isfinite(s) or s <= 0.0:
                msg = f"Surface IV must be finite and positive at strike={strike}, tau={tau}."
                raise ValueError(msg)
            # Solve z - d1 = 0 without evaluating the normal CDF on every step.
            return float(z + x / s - 0.5 * s), vol

        root = guess
        f_guess = evaluate(guess)[0]
        if abs(f_guess) > 1e-12:
            width = max(0.5 * total_vol, 1e-4)
            direction = -1.0 if f_guess > 0.0 else 1.0
            for _ in range(32):
                bound = guess + direction * width
                f_bound = evaluate(bound)[0]
                if f_bound == 0.0 or np.signbit(f_guess) != np.signbit(f_bound):
                    break
                width *= 2.0
            else:
                msg = f"Could not bracket a strike for delta={delta}, tau={tau}."
                raise ValueError(msg)
            lower, upper = sorted((guess, bound))
            root = brentq(lambda x: evaluate(x)[0], lower, upper, xtol=1e-12 * min(total_vol, 1.0), rtol=1e-12)

        strike, vol = float(fwd * np.exp(root)), evaluate(root)[1]
        actual_delta = float(np.asarray(black76_undisc_fwd_delta(fwd, strike, tau, vol, is_call=True)).item())
        if not np.isfinite(actual_delta) or abs(actual_delta - delta) > 1e-10:
            msg = f"Delta inversion did not reach the requested accuracy for delta={delta}, tau={tau}."
            raise ValueError(msg)
        return vol

    def _vol_at_scalar_maturity(self, k: np.ndarray, t: float) -> np.ndarray:
        """Helper: interpolate/extrapolate vols for scalar maturity t."""
        if self._linear_model is None:
            msg = "linear_model is required for forward-moneyness interpolation."
            raise ValueError(msg)

        taus = self._taus
        fwd_t = float(self._linear_model.fwd(t))
        log_moneyness = np.log(k / fwd_t)

        def _slice_vol(idx: int) -> np.ndarray:
            strike = float(self._linear_model.fwd(float(taus[idx]))) * np.exp(log_moneyness)
            return self._smiles[idx].vol(strike)

        # Flat extrapolation for maturities outside the known range.
        if t <= float(taus[0]):
            return _slice_vol(0)

        if t >= float(taus[-1]):
            return _slice_vol(-1)

        # Interpolate in total variance
        hi = int(np.searchsorted(taus, t, side="left"))
        lo = hi - 1

        tau_lo = float(taus[lo])
        tau_hi = float(taus[hi])

        vol_lo = _slice_vol(lo)
        vol_hi = _slice_vol(hi)

        total_var_lo = (vol_lo**2) * tau_lo
        total_var_hi = (vol_hi**2) * tau_hi

        weight = (t - tau_lo) / (tau_hi - tau_lo)
        total_var_t = total_var_lo + (total_var_hi - total_var_lo) * weight

        return np.sqrt(total_var_t / t)
