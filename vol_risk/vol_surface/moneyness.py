from abc import ABC
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vol_risk.models.black76 import bsm_spot_delta, bsm_spot_delta_to_strike
from vol_risk.models.linear import LinearEquityMarket

MONEYNESS_REGISTRY = {}


def register_moneyness(key: str):
    def decorator(cls):
        MONEYNESS_REGISTRY[key] = cls
        return cls

    return decorator


@register_moneyness("base")
@dataclass(frozen=True)
class Moneyness(ABC):
    """Abstract base class for moneyness calculations."""

    le: LinearEquityMarket

    def value(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

    def invert(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError


@register_moneyness("ks")
@dataclass(frozen=True)
class SpotMoneyness(Moneyness):
    """Spot moneyness: K/S."""

    def value(self, *, strike: ArrayLike, **_) -> ArrayLike:
        return strike / self.le.spot

    def invert(self, *, moneyness: ArrayLike, **_) -> ArrayLike:
        return self.le.spot * moneyness


@register_moneyness("kf")
@dataclass(frozen=True)
class FwdMoneyness(Moneyness):
    """Forward moneyness: K/F."""

    def value(self, *, strike: ArrayLike, tau: ArrayLike, **_) -> ArrayLike:
        return strike / self.le.fwd(tau)

    def invert(self, *, moneyness: ArrayLike, tau: ArrayLike, **_) -> ArrayLike:
        return self.le.fwd(tau) * moneyness


@register_moneyness("lkf")
@dataclass(frozen=True)
class LogFwdMoneyness(Moneyness):
    """Log-forward moneyness: log(K/F)."""

    def value(self, *, strike: ArrayLike, tau: ArrayLike, **_) -> ArrayLike:
        return np.log(strike / self.le.fwd(tau))

    def invert(self, *, moneyness: ArrayLike, tau: ArrayLike, **_) -> ArrayLike:
        return self.le.fwd(tau) * np.exp(moneyness)


@register_moneyness("slkf")
@dataclass(frozen=True)
class StdLogFwdMoneyness(Moneyness):
    """Standardized log-forward moneyness: log(K/F) / (sqrt(tau) * sigma)."""

    def value(self, *, strike: ArrayLike, tau: ArrayLike, sigma: ArrayLike, **_) -> ArrayLike:
        scaling = 1 / (np.sqrt(tau) * sigma)
        return np.log(strike / self.le.fwd(tau)) * scaling

    def invert(self, *, moneyness: ArrayLike, tau: ArrayLike, sigma: ArrayLike, **_) -> ArrayLike:
        scaling = 1 / (np.sqrt(tau) * sigma)
        return np.exp(moneyness / scaling) * self.le.fwd(tau)


@register_moneyness("delta")
@dataclass(frozen=True)
class DeltaMoneyness(Moneyness):
    """Forward delta moneyness: delta(K, tau, sigma)."""

    def value(self, *, strike: ArrayLike, tau: ArrayLike, sigma: ArrayLike, **_) -> ArrayLike:
        return bsm_spot_delta(
            s=self.le.spot,
            k=strike,
            t=tau,
            r=self.le.zero_rate(tau),
            q=self.le.zero_dvd_yield(tau),
            sigma=sigma,
            is_call=True,
        )

    def invert(self, *, moneyness: ArrayLike, tau: ArrayLike, sigma: ArrayLike, **_) -> ArrayLike:
        return bsm_spot_delta_to_strike(
            delta=moneyness,
            s=self.le.spot,
            t=tau,
            r=self.le.zero_rate(tau),
            q=self.le.zero_dvd_yield(tau),
            sigma=sigma,
            is_call=True,
        )
