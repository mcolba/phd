from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vol_risk.models.black76 import (
    black76_undisc_fwd_delta,
    black76_undisc_fwd_delta_to_strike,
)
from vol_risk.models.linear import LinearEquityMarket

MONEYNESS_REGISTRY = {}


def register_moneyness(key: str):
    def decorator(cls):
        MONEYNESS_REGISTRY[key] = cls
        return cls

    return decorator


@dataclass(frozen=True)
class Moneyness(ABC):
    """Abstract base class for moneyness calculations."""

    le: LinearEquityMarket
    requires_sigma: bool = False

    @abstractmethod
    def value(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def invert(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError


@register_moneyness("base")
@register_moneyness("k")
@dataclass(frozen=True)
class Strike(Moneyness):
    """Strike moneyness: K."""

    def value(self, *, strike: ArrayLike, **_) -> ArrayLike:
        return strike

    def invert(self, *, moneyness: ArrayLike, **_) -> ArrayLike:
        return moneyness


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


@register_moneyness("lkft")
@dataclass(frozen=True)
class TauScaledLogFwdMoneyness(Moneyness):
    """Standardized log-forward moneyness: log(K/F) / sqrt(tau)."""

    def value(self, *, strike: ArrayLike, tau: ArrayLike, **_) -> ArrayLike:
        scaling = 1 / np.sqrt(tau)
        return np.log(strike / self.le.fwd(tau)) * scaling

    def invert(self, *, moneyness: ArrayLike, tau: ArrayLike, **_) -> ArrayLike:
        scaling = 1 / np.sqrt(tau)
        return np.exp(moneyness / scaling) * self.le.fwd(tau)


@register_moneyness("slkf")
@dataclass(frozen=True)
class StdLogFwdMoneyness(Moneyness):
    """Standardized log-forward moneyness: log(K/F) / (sqrt(tau) * sigma)."""

    requires_sigma: bool = True

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

    requires_sigma: bool = True

    def value(self, *, strike: ArrayLike, tau: ArrayLike, sigma: ArrayLike, **_) -> ArrayLike:
        return black76_undisc_fwd_delta(
            fwd=self.le.fwd(tau),
            strike=strike,
            tau=tau,
            sigma=sigma,
            is_call=True,
        )

    def invert(self, *, moneyness: ArrayLike, tau: ArrayLike, sigma: ArrayLike, **_) -> ArrayLike:
        return black76_undisc_fwd_delta_to_strike(
            delta=moneyness,
            fwd=self.le.fwd(tau),
            tau=tau,
            sigma=sigma,
            is_call=True,
        )
