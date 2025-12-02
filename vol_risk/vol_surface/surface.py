from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from vol_risk.protocols import VolInterpolator


@dataclass(frozen=True)
class Extrapolator(ABC):
    """Vol extrapolator base class."""

    @abstractmethod
    def __call__(self) -> Callable:
        raise NotImplementedError


@dataclass(frozen=True)
class FlatExtrapolator(Extrapolator):
    """Flat vol extrapolator."""

    def __init__(self, lower: float | None, upper: float | None) -> None:
        self._lower = -np.inf if lower is None else lower
        self._upper = np.inf if upper is None else upper

    def __call__(self, f: Callable) -> Callable:
        """Return a decorated function with flat extrapolation."""

        def wrapper(k: np.ndarray) -> np.ndarray:
            k = np.asarray(k)
            return f(np.clip(k, self._lower, self._upper))

        return wrapper


@dataclass(frozen=True)
class VolSmile:
    """Vol smile object wrapping a VolInterpolator and an extrapolator."""

    def __init__(
        self,
        interpl: Iterable[VolInterpolator],
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

    def __init__(self, slices: Iterable[VolSmile]) -> None:
        self._slices = slices

    def vol(self, k, t) -> np.array:
        pass
