"""Carr-Madan quadrature call engine from a log-stock characteristic function."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, InitVar, dataclass, field, replace
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
CharacteristicFunction = Callable[[ComplexArray], ComplexArray]
CallPriceFunction = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True, slots=True)
class CallControlVariate:
    """Control variate for a discounted Fourier call engine."""

    cf: CharacteristicFunction
    call_price: CallPriceFunction


@dataclass(frozen=True, slots=True)
class QuadEngineParams:
    """Parameters for finite-range Carr-Madan call quadrature."""

    damping: float = 0.03
    upper_bound: float = 512.0
    adaptive_trunc: bool = False
    breakpoints: tuple[float, ...] | None = (8.0, 64.0)
    epsabs: float = 1.0e-8
    epsrel: float = 1.0e-8
    limit: int = 200

    validate: InitVar[bool] = True

    def __post_init__(self, validate: bool) -> None:
        if validate:
            _validate_positive_finite("damping", self.damping)
            _validate_positive_finite("upper_bound", self.upper_bound)
            breaks = None if self.breakpoints is None else tuple(float(x) for x in self.breakpoints)
            if breaks is not None:
                if not breaks:
                    msg = "breakpoints must be a non-empty tuple of finite positive values."
                    raise ValueError(msg)
                if any(not np.isfinite(x) or x <= 0.0 for x in breaks):
                    msg = "breakpoints must contain only finite positive values."
                    raise ValueError(msg)
                if any(right <= left for left, right in pairwise(breaks)):
                    msg = "breakpoints must be strictly increasing."
                    raise ValueError(msg)
            if self.epsabs <= 0.0 or not np.isfinite(self.epsabs):
                msg = "epsabs must be finite and positive."
                raise ValueError(msg)
            if self.epsrel <= 0.0 or not np.isfinite(self.epsrel):
                msg = "epsrel must be finite and positive."
                raise ValueError(msg)
            if self.limit <= 0:
                msg = "limit must be positive."
                raise ValueError(msg)
            object.__setattr__(self, "breakpoints", breaks)


def scale_truncation(tau: float, par: QuadEngineParams) -> QuadEngineParams:
    """Return a copy of par with an extended upper_bound for short maturities."""
    _validate_positive_finite("tau", tau)
    if tau < 0.5:
        return replace(par, upper_bound=par.upper_bound / float(np.sqrt(tau)))
    return par


def _validate_positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive."
        raise ValueError(msg)


def _as_strike_array(strike: ArrayLike) -> tuple[FloatArray, FloatArray, bool]:
    strike_arr = np.asarray(strike, dtype=float)
    if np.any(~np.isfinite(strike_arr)) or np.any(strike_arr <= 0.0):
        msg = "strike values must be finite and positive."
        raise ValueError(msg)
    return strike_arr, np.atleast_1d(strike_arr).astype(float, copy=False), strike_arr.ndim == 0


def _validate_disc(disc: float) -> float:
    disc_value = float(disc)
    _validate_positive_finite("disc", disc_value)
    return disc_value


def _integrand(
    v: float,
    log_strike: float,
    disc: float,
    damping: float,
    cf: CharacteristicFunction,
    control_cf: CharacteristicFunction | None,
) -> float:
    """Integrand for the Carr-Madan quadrature."""
    u = np.asarray([v - 1j * (damping + 1.0)], dtype=np.complex128)
    phi = np.asarray(cf(u), dtype=np.complex128).reshape(-1)[0]
    if control_cf is not None:
        phi = phi - np.asarray(control_cf(u), dtype=np.complex128).reshape(-1)[0]

    denominator = damping * damping + damping - v * v + 1j * (2.0 * damping + 1.0) * v
    value = np.exp(-damping * log_strike - 1j * v * log_strike) * disc * phi / denominator / np.pi
    return float(np.real(value))


def _quad_call_price_from_cf(
    cf: CharacteristicFunction,
    strike: ArrayLike,
    disc: float,
    damping: float,
    control: CallControlVariate | None = None,
    limit: int = 100,
    epsabs: float = 1.0e-8,
    epsrel: float = 1.0e-8,
    upper_bound: float = 512.0,
    breakpoints: tuple[float, ...] | None = (8.0, 64.0),
) -> float | FloatArray:
    """Return discounted European call prices by finite Carr-Madan quadrature."""
    strike_arr, strike_1d, scalar_input = _as_strike_array(strike)

    disc_value = _validate_disc(disc)
    log_strikes = np.log(strike_1d)
    control_cf = None if control is None else control.cf

    prices = np.empty(strike_1d.shape, dtype=float)
    for idx, log_strike in enumerate(log_strikes):
        integral, _ = quad(
            _integrand,
            a=0.0,
            b=upper_bound,
            args=(
                float(log_strike),
                disc_value,
                damping,
                cf,
                control_cf,
            ),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            points=breakpoints,
        )
        prices[idx] = integral

    if control is not None:
        prices += np.asarray(control.call_price(strike_1d), dtype=float).reshape(strike_1d.shape)

    out = prices.reshape(strike_arr.shape) if not scalar_input else prices
    return float(out[0]) if scalar_input else out


@dataclass(frozen=True, slots=True)
class QuadCallEngine:
    """Finite-range Carr-Madan quadrature call engine."""

    cf: CharacteristicFunction
    disc: float
    _: KW_ONLY
    tau: float | None = None
    params: QuadEngineParams = field(default_factory=QuadEngineParams)
    control: CallControlVariate | None = None

    def __post_init__(self) -> None:
        if self.params.adaptive_trunc:
            if self.tau is None:
                msg = "tau must be provided when adaptive_trunc is True."
                raise ValueError(msg)
            scaled_params = scale_truncation(self.tau, self.params)
            object.__setattr__(self, "params", scaled_params)

    def __call__(
        self,
        strike: ArrayLike,
    ) -> float | FloatArray:
        return _quad_call_price_from_cf(
            cf=self.cf,
            strike=strike,
            disc=self.disc,
            damping=self.params.damping,
            control=self.control,
            limit=self.params.limit,
            epsabs=self.params.epsabs,
            epsrel=self.params.epsrel,
            upper_bound=self.params.upper_bound,
            breakpoints=self.params.breakpoints,
        )


__all__ = [
    "CallControlVariate",
    "QuadCallEngine",
    "QuadEngineParams",
]
