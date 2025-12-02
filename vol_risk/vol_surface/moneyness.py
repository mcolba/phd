from abc import ABC
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vol_risk.models.black76 import black76_fwd_delta, black76_fwd_delta_to_strike
from vol_risk.models.linear import LinearEquity

MONEYNESS_REGISTRY = {
    "base": None,
    "log": None,
    "forward": None,
    "log-forward": None,
    "std-log-forward": None,
    "delta": None,
}


@dataclass(frozen=True)
class Moneyness(ABC):
    """Abstract base class for moneyness calculations."""

    le: LinearEquity

    def value(self) -> ArrayLike:
        raise NotImplementedError

    def invert(self) -> ArrayLike:
        raise NotImplementedError


@dataclass(frozen=True)
class SpotMoneyness(Moneyness):
    """Spot moneyness: K/S."""

    def value(self, strike: ArrayLike) -> ArrayLike:
        return strike / self.le.spot

    def invert(self, k: ArrayLike) -> ArrayLike:
        return self.le.spot * k


@dataclass(frozen=True)
class FwdMoneyness(Moneyness):
    """Forward moneyness: K/F."""

    def value(self, strike: ArrayLike, tau: ArrayLike) -> ArrayLike:
        return strike / self.le.fwd(tau)

    def invert(self, k: ArrayLike, tau: ArrayLike) -> ArrayLike:
        return self.le.fwd(tau) * k


@dataclass(frozen=True)
class LogFwdMoneyness(Moneyness):
    """Log-forward moneyness: log(K/F)."""

    def value(self, strike: ArrayLike, tau: ArrayLike) -> ArrayLike:
        return np.log(strike / self.le.fwd(tau))

    def invert(self, k: ArrayLike, tau: ArrayLike) -> ArrayLike:
        return self.le.fwd(tau) * np.exp(k)


@dataclass(frozen=True)
class StdLogFwdMoneyness(Moneyness):
    """Standardized log-forward moneyness: log(K/F) / (sqrt(tau) * sigma)."""

    def value(self, k: ArrayLike, tau: ArrayLike, sigma: ArrayLike) -> ArrayLike:
        scaling = 1 / (np.sqrt(tau) * sigma)
        return np.log(k / self.le.fwd(tau)) / scaling

    def invert(self, m: ArrayLike, tau: ArrayLike, sigma: ArrayLike) -> ArrayLike:
        scaling = 1 / (np.sqrt(tau) * sigma)
        return np.exp(m * scaling) * self.le.fwd(tau)


@dataclass(frozen=True)
class DeltaMoneyness(Moneyness):
    """Forward delta moneyness: delta(K, tau, sigma)."""

    def value(self, strike: ArrayLike, tau: ArrayLike, sigma: ArrayLike) -> ArrayLike:
        return black76_fwd_delta(
            f=self.le.fwd(tau),
            k=strike,
            t=tau,
            r=self.le.r,
            sigma=sigma,
            is_call=True,
        )

    def invert(self, k: ArrayLike, tau: ArrayLike, sigma: ArrayLike) -> ArrayLike:
        return black76_fwd_delta_to_strike(
            delta=k,
            f=self.le.fwd(tau),
            t=tau,
            r=self.le.r,
            sigma=sigma,
            is_call=True,
        )
