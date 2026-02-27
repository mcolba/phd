from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import numpy as np


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

    def vol(self, k) -> np.array:
        """Get vol for given strikes k and moneyness convention."""
        return self._extrapolator(self._interpolator)(k)


class VolSurface:
    """Vol surface object wrapping multiple VolSmiles slices."""

    def __init__(self, taus: np.array, smiles: Iterable[VolSmile]) -> None:
        """Initialize the surface with a mapping tau -> VolSmile."""
        self._taus = np.asarray(taus)
        self._smiles = smiles

    def vol(self, k, t) -> np.ndarray:
        """Evaluate the surface at strikes and maturities.

        If t is between existing slices, interpolate in total variance.
        Otherwise, use flat extrapolation in maturity.
        """
        k_arr = np.asarray(k, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        # Scalar maturity: vectorised in strikes.
        if t_arr.ndim == 0:
            return self._vol_at_scalar_maturity(k_arr, float(t_arr))

        # Array maturity: require same shape as strikes and interpolate per point.
        if k_arr.shape != t_arr.shape:
            msg = "Shapes of k and t must match when both are arrays."
            raise ValueError(msg)

        vols = np.empty_like(k_arr, dtype=float)
        for i, (ki, ti) in enumerate(zip(k_arr, t_arr, strict=False)):
            vols[i] = float(self._vol_at_scalar_maturity(np.asarray([ki], dtype=float), float(ti))[0])

        return vols

    def _vol_at_scalar_maturity(self, k: np.ndarray, t: float) -> np.ndarray:
        """Helper: interpolate/extrapolate vols for scalar maturity t."""
        taus = self._taus

        # Flat extrapolation for maturities outside the known range.
        if t <= float(taus[0]):
            return self._smiles[0].vol(k)

        if t >= float(taus[-1]):
            return self._smiles[-1].vol(k)

        # Interpolate in total variance between the two surrounding slices.
        hi = int(np.searchsorted(taus, t, side="left"))
        lo = hi - 1

        tau_lo = float(taus[lo])
        tau_hi = float(taus[hi])

        vol_lo = self._smiles[lo].vol(k)
        vol_hi = self._smiles[hi].vol(k)

        total_var_lo = (vol_lo**2) * tau_lo
        total_var_hi = (vol_hi**2) * tau_hi

        weight = (t - tau_lo) / (tau_hi - tau_lo)
        total_var_t = total_var_lo + (total_var_hi - total_var_lo) * weight

        return np.sqrt(total_var_t / t)
