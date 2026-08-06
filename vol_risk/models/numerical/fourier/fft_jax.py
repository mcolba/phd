"""JAX Carr-Madan FFT call engine from a log-stock characteristic function."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

JaxCharacteristicFunction = Callable[[jax.Array], jax.Array]
JaxCallPriceFunction = Callable[[jax.Array], jax.Array]
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _log_x64_disabled_warning() -> None:
    """Log the reduced-precision warning at most once per process."""
    msg = (
        "JAX x64 is disabled; the JAX FFT engine will use reduced precision."
    )
    logger.warning(msg)


@dataclass(frozen=True, slots=True)
class JaxControlVariate:
    """JAX control variate for a discounted Fourier call engine."""

    cf: JaxCharacteristicFunction
    call_price: JaxCallPriceFunction


@dataclass(frozen=True, slots=True)
class JaxFFTCallEngineParams:
    """Parameters for a JAX Carr-Madan FFT call engine."""

    damping: float = 1.25
    log_strike_step: float = 0.001
    grid_size: int | None = None

    def __post_init__(self) -> None:
        _validate_positive_finite("damping", self.damping)
        _validate_positive_finite("log_strike_step", self.log_strike_step)
        if self.grid_size is not None:
            _validate_grid_size(self.grid_size)


@dataclass(frozen=True, slots=True)
class JaxFFTCallGrid:
    """Precomputed JAX FFT grid for Carr-Madan call pricing."""

    damping: float
    log_strike_step: float
    grid_size: int
    log_strike_half_width: float
    frequency_step: float
    frequency: jax.Array
    shifted_frequency: jax.Array
    log_strike: jax.Array
    strike: jax.Array
    weights: jax.Array
    phase: jax.Array
    denominator: jax.Array


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


def _as_positive_strikes(strike: ArrayLike) -> np.ndarray:
    strike_arr = np.asarray(strike, dtype=float)
    if np.any(~np.isfinite(strike_arr)) or np.any(strike_arr <= 0.0):
        msg = "strike values must be finite and positive."
        raise ValueError(msg)
    return np.atleast_1d(strike_arr)


def _minimum_grid_size(strike: np.ndarray, log_strike_step: float) -> int:
    max_abs_log_strike = float(np.max(np.abs(np.log(strike))))
    n_float = 2.0 * (max_abs_log_strike + log_strike_step) / log_strike_step
    return max(2, int(np.ceil(n_float)))


def _next_power_of_two(n: int) -> int:
    return 1 << int(np.ceil(np.log2(n)))


def make_jax_fft_call_grid(strike: ArrayLike, params: JaxFFTCallEngineParams) -> JaxFFTCallGrid:
    """Return a reusable JAX FFT grid covering the requested strikes."""
    if not jax.config.jax_enable_x64:
        _log_x64_disabled_warning()

    strike_1d = _as_positive_strikes(strike)
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

    return JaxFFTCallGrid(
        damping=float(params.damping),
        log_strike_step=float(params.log_strike_step),
        grid_size=grid_size,
        log_strike_half_width=float(log_strike_half_width),
        frequency_step=float(frequency_step),
        frequency=jnp.asarray(frequency),
        shifted_frequency=jnp.asarray(frequency - 1j * (params.damping + 1.0)),
        log_strike=jnp.asarray(log_strike),
        strike=jnp.asarray(np.exp(log_strike)),
        weights=jnp.asarray(weights),
        phase=jnp.asarray(np.exp(1j * log_strike_half_width * frequency)),
        denominator=jnp.asarray(denominator),
    )


def _check_grid_matches_params(params: JaxFFTCallEngineParams, grid: JaxFFTCallGrid) -> None:
    if params.damping != grid.damping:
        msg = "grid damping does not match params.damping."
        raise ValueError(msg)
    if params.log_strike_step != grid.log_strike_step:
        msg = "grid log_strike_step does not match params.log_strike_step."
        raise ValueError(msg)
    if params.grid_size is not None and params.grid_size != grid.grid_size:
        msg = "grid grid_size does not match params.grid_size."
        raise ValueError(msg)


def _fft_call_price_from_cf(
    cf: JaxCharacteristicFunction,
    strike: ArrayLike,
    disc: ArrayLike,
    params: JaxFFTCallEngineParams,
    grid: JaxFFTCallGrid,
    control: JaxControlVariate | None = None,
) -> jax.Array:
    """Return discounted European call prices by JAX Carr-Madan FFT."""
    _check_grid_matches_params(params=params, grid=grid)
    return fft_call_price_on_grid(cf=cf, strike=strike, disc=disc, grid=grid, control=control)


def fft_call_price_on_grid(
    cf: JaxCharacteristicFunction,
    strike: ArrayLike,
    disc: ArrayLike,
    grid: JaxFFTCallGrid,
    control: JaxControlVariate | None = None,
) -> jax.Array:
    """Pure traceable Carr-Madan call pricing on a precomputed JAX grid."""
    strike_arr = jnp.asarray(strike)
    disc_arr = jnp.asarray(disc)
    phi = cf(grid.shifted_frequency)
    if control is not None:
        phi = phi - control.cf(grid.shifted_frequency)

    fft_input = grid.phase * grid.weights * disc_arr * phi / grid.denominator
    fft_result = jnp.fft.fft(fft_input)
    call_grid = jnp.exp(-grid.damping * grid.log_strike) * jnp.real(fft_result) / jnp.pi
    call_price = jnp.interp(strike_arr, grid.strike, call_grid)

    if control is not None:
        call_price = call_price + control.call_price(strike_arr)

    return call_price


@dataclass(frozen=True, slots=True)
class JaxFFTCallEngine:
    """JAX Carr-Madan FFT call engine compatible with ``jax.jit`` and ``jax.vmap``.

    When ``grid`` is provided, ``__call__`` skips grid construction and is
    safe to trace under ``jax.jit``; otherwise the grid is built per call
    from ``strike`` (NumPy-side, not jit-friendly).
    """

    cf: JaxCharacteristicFunction
    disc: ArrayLike
    _: KW_ONLY
    control: JaxControlVariate | None = None
    params: JaxFFTCallEngineParams = field(default_factory=JaxFFTCallEngineParams)
    grid: JaxFFTCallGrid | None = None

    def __call__(self, strike: ArrayLike) -> jax.Array:
        grid = self.grid if self.grid is not None else make_jax_fft_call_grid(
            strike=strike, params=self.params
        )
        return _fft_call_price_from_cf(
            cf=self.cf,
            strike=strike,
            disc=self.disc,
            params=self.params,
            grid=grid,
            control=self.control,
        )


__all__ = [
    "JaxControlVariate",
    "JaxFFTCallEngine",
    "JaxFFTCallEngineParams",
    "JaxFFTCallGrid",
    "_fft_call_price_from_cf",
    "fft_call_price_on_grid",
    "make_jax_fft_call_grid",
]
