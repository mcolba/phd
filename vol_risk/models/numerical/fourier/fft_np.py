"""NumPy Carr-Madan FFT call engine from a log-stock characteristic function."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vol_risk.models.numerical.fourier.base import CallControlVariate  # noqa: TC001

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
CharacteristicFunction = Callable[[ComplexArray], ComplexArray]


@dataclass(frozen=True, slots=True)
class FFTEngineParams:
    """Parameters for a NumPy Carr-Madan FFT call engine."""

    damping: float = 1.25
    log_strike_step: float = 0.001
    grid_size: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_finite("damping", self.damping)
        _validate_positive_finite("log_strike_step", self.log_strike_step)
        if self.grid_size is not None:
            _validate_grid_size(self.grid_size)


@dataclass(frozen=True, slots=True)
class FFTCallGrid:
    """Precomputed NumPy FFT grid for Carr-Madan call pricing."""

    damping: float
    log_strike_step: float
    grid_size: int
    log_strike_half_width: float
    frequency_step: float
    frequency: FloatArray
    shifted_frequency: ComplexArray
    log_strike: FloatArray
    strike: FloatArray
    weights: FloatArray
    phase: ComplexArray
    denominator: ComplexArray


def _validate_positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive."
        raise ValueError(msg)


def _validate_grid_size(grid_size: int) -> None:
    if not isinstance(grid_size, int) or grid_size < 2:
        msg = "grid_size must be an integer greater than one."
        raise ValueError(msg)
    if grid_size & (grid_size - 1):
        msg = "grid_size must be a power of two."
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


def _minimum_grid_size(strike: FloatArray, log_strike_step: float) -> int:
    max_abs_log_strike = float(np.max(np.abs(np.log(strike))))
    n_float = 2.0 * (max_abs_log_strike + log_strike_step) / log_strike_step
    return max(2, int(np.ceil(n_float)))


def _next_power_of_two(n: int) -> int:
    return 1 << int(np.ceil(np.log2(n)))


def _make_fft_call_grid(strike: ArrayLike, params: FFTEngineParams) -> FFTCallGrid:
    """Return a reusable NumPy FFT grid covering the requested strikes."""
    _, strike_1d, _ = _as_strike_array(strike)
    grid_size = params.grid_size
    if grid_size is None:
        grid_size = _next_power_of_two(_minimum_grid_size(strike_1d, params.log_strike_step))
    else:
        _validate_grid_size(grid_size)

    log_strike_half_width = params.log_strike_step * grid_size / 2.0
    frequency_step = 2.0 * np.pi / (params.log_strike_step * grid_size)
    j_idx = np.arange(grid_size, dtype=float)
    frequency = frequency_step * j_idx
    log_strike = -log_strike_half_width + params.log_strike_step * j_idx

    weights = np.full(grid_size, 2.0 * frequency_step / 3.0, dtype=float)
    weights[0] = frequency_step / 3.0
    weights[1::2] = 4.0 * frequency_step / 3.0

    denominator = (
        params.damping * params.damping
        + params.damping
        - frequency * frequency
        + 1j * (2.0 * params.damping + 1.0) * frequency
    )

    return FFTCallGrid(
        damping=float(params.damping),
        log_strike_step=float(params.log_strike_step),
        grid_size=grid_size,
        log_strike_half_width=float(log_strike_half_width),
        frequency_step=float(frequency_step),
        frequency=frequency,
        shifted_frequency=frequency - 1j * (params.damping + 1.0),
        log_strike=log_strike,
        strike=np.exp(log_strike),
        weights=weights,
        phase=np.exp(1j * log_strike_half_width * frequency),
        denominator=denominator,
    )


def _check_grid_matches_params(params: FFTEngineParams, grid: FFTCallGrid) -> None:
    if params.damping != grid.damping:
        msg = "grid damping does not match params.damping."
        raise ValueError(msg)
    if params.log_strike_step != grid.log_strike_step:
        msg = "grid log_strike_step does not match params.log_strike_step."
        raise ValueError(msg)
    if params.grid_size is not None and params.grid_size != grid.grid_size:
        msg = "grid grid_size does not match params.grid_size."
        raise ValueError(msg)


def _evaluate_cf(cf: CharacteristicFunction, u: ComplexArray, label: str) -> ComplexArray:
    values = np.asarray(cf(u), dtype=np.complex128)
    if values.shape != u.shape:
        msg = f"{label} must return an array with shape {u.shape}."
        raise ValueError(msg)
    return values


def _fft_call_price_from_cf(
    cf: CharacteristicFunction,
    strike: ArrayLike,
    disc: float,
    params: FFTEngineParams,
    grid: FFTCallGrid,
    control: CallControlVariate | None = None,
) -> float | FloatArray:
    """Return discounted European call prices by NumPy Carr-Madan FFT."""
    strike_arr, strike_1d, scalar_input = _as_strike_array(strike)
    disc_value = _validate_disc(disc)
    _check_grid_matches_params(params=params, grid=grid)

    phi = _evaluate_cf(cf, grid.shifted_frequency, "cf")
    if control is not None:
        phi = phi - _evaluate_cf(control.cf, grid.shifted_frequency, "control.cf")

    fft_input = grid.phase * grid.weights * disc_value * phi / grid.denominator
    fft_result = np.fft.fft(fft_input)
    call_grid = np.exp(-grid.damping * grid.log_strike) * np.real(fft_result) / np.pi
    call_price = np.interp(strike_1d, grid.strike, call_grid)

    if control is not None:
        call_price += np.asarray(control.call_price(strike_1d), dtype=float).reshape(strike_1d.shape)

    out = call_price.reshape(strike_arr.shape) if not scalar_input else call_price
    return float(out[0]) if scalar_input else out


@dataclass(frozen=True, slots=True)
class FFTCallEngine:
    """NumPy Carr-Madan FFT call engine."""

    cf: CharacteristicFunction
    disc: float
    _: KW_ONLY
    control: CallControlVariate | None = None
    params: FFTEngineParams = field(default_factory=FFTEngineParams)

    def __call__(self, strike: ArrayLike) -> float | FloatArray:
        grid = _make_fft_call_grid(strike=strike, params=self.params)
        return _fft_call_price_from_cf(
            cf=self.cf,
            strike=strike,
            disc=self.disc,
            params=self.params,
            grid=grid,
            control=self.control,
        )


__all__ = [
    "FFTCallEngine",
    "FFTCallGrid",
    "FFTEngineParams",
    "_fft_call_price_from_cf",
    "_make_fft_call_grid",
]
