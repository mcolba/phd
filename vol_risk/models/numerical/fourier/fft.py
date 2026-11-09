"""JAX Carr-Madan FFT call engine from a log-stock characteristic function."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from functools import lru_cache
from math import ceil, isfinite, log, log2, pi
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from vol_risk.models.numerical.control_variates import JaxCallControlVariate

JaxCharacteristicFunction = Callable[[jax.Array], jax.Array]
JaxCallPriceFunction = Callable[[jax.Array], jax.Array]
MAX_AUTO_GRID_SIZE = 1 << 22
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _log_x64_disabled_warning() -> None:
    """Log the reduced-precision warning at most once per process."""
    msg = "JAX x64 is disabled; the JAX FFT engine will use reduced precision."
    logger.warning(msg)


@dataclass(frozen=True, slots=True)
class JaxFFTCallEngineParams:
    r"""Parameters for a JAX Carr-Madan FFT call engine.

    When ``grid_size`` is ``None`` the grid is sized so that the requested strikes sit at
    least ``margin = -ln(alias_tol) / tail_rate`` inside the log-strike window ``[-b, b)``.
    The FFT returns the ``2b``-periodic summation of the damped call $e^{\alpha k} C(k)$
    (Poisson summation; Lee, 2004, J. Comput. Finance 7(3), gives the sampling-error term of
    the Carr-Madan method), so the alias contribution at a strike is bounded by the damped
    call evaluated ``2b - |k|`` away. Without a control variate the damped call decays only as
    $e^{\alpha k}$ for $k \to -\infty$ (Carr & Madan, 1999), which is the default rate. With a
    Black control sharing the forward, the residual decays as $e^{(1 + G + \alpha) k}$ on the
    left and $e^{-(M - 1 - \alpha) k}$ on the right for a model with exponential log-price
    tail rates $G$, $M$; callers with that knowledge pass ``min`` of the two as ``tail_rate``.

    Attributes:
        damping: Carr-Madan damping $\alpha > 0$.
        log_strike_step: Log-strike grid spacing $\lambda$.
        grid_size: Explicit power-of-two grid size, or ``None`` to size from the strikes.
        alias_tol: Target bound on the relative aliasing error used to size the grid.
        tail_rate: Slowest exponential decay rate of the damped call in log-strike;
            ``None`` uses ``damping`` (the uncontrolled worst case).
    """

    damping: float = 1.25
    log_strike_step: float = 0.001
    grid_size: int | None = None
    alias_tol: float = 1.0e-8
    tail_rate: float | None = None

    def __post_init__(self) -> None:
        _validate_positive_finite("damping", self.damping)
        _validate_positive_finite("log_strike_step", self.log_strike_step)
        if self.grid_size is not None:
            _validate_grid_size(self.grid_size)
        if not isfinite(self.alias_tol) or not 0.0 < self.alias_tol < 1.0:
            msg = "alias_tol must lie in (0, 1)."
            raise ValueError(msg)
        if self.tail_rate is not None:
            _validate_positive_finite("tail_rate", self.tail_rate)

    @property
    def log_strike_margin(self) -> float:
        """Log-strike distance from the window edge required for the aliasing tolerance."""
        rate = self.damping if self.tail_rate is None else self.tail_rate
        return -log(self.alias_tol) / rate


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
    if not isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive."
        raise ValueError(msg)


def _validate_grid_size(grid_size: int) -> None:
    if not isinstance(grid_size, int) or grid_size < 2:
        msg = "grid_size must be an integer greater than one."
        raise ValueError(msg)
    if grid_size & (grid_size - 1):
        msg = "grid_size must be a power of two."
        raise ValueError(msg)


def _as_positive_strikes(strike: ArrayLike | jax.Array) -> jax.Array:
    strike_arr = jnp.asarray(strike, dtype=float)
    if bool(jnp.any(~jnp.isfinite(strike_arr))) or bool(jnp.any(strike_arr <= 0.0)):
        msg = "strike values must be finite and positive."
        raise ValueError(msg)
    return jnp.atleast_1d(strike_arr)


def _minimum_grid_size(strike: jax.Array, log_strike_step: float, margin: float) -> int:
    """Smallest N with every strike at least ``margin`` (and one step) inside ``[-b, b)``."""
    max_abs_log_strike = float(jnp.max(jnp.abs(jnp.log(strike))))
    n_float = 2.0 * (max_abs_log_strike + max(margin, log_strike_step)) / log_strike_step
    return max(2, ceil(n_float))


def _next_power_of_two(n: int) -> int:
    return 1 << ceil(log2(n))


def make_jax_fft_call_grid(strike: ArrayLike | jax.Array, params: JaxFFTCallEngineParams) -> JaxFFTCallGrid:
    """Return a reusable JAX FFT grid covering the requested strikes.

    With ``params.grid_size=None`` the size follows ``params.log_strike_margin``; an explicit
    ``grid_size`` is used as given and the aliasing margin is the caller's responsibility.
    """
    if not jax.config.read("jax_enable_x64"):
        _log_x64_disabled_warning()

    strike_1d = _as_positive_strikes(strike)
    grid_size = params.grid_size
    if grid_size is None:
        minimum = _minimum_grid_size(strike_1d, params.log_strike_step, params.log_strike_margin)
        grid_size = _next_power_of_two(minimum)
        if grid_size > MAX_AUTO_GRID_SIZE:
            msg = (
                f"Automatic FFT grid size {grid_size} exceeds {MAX_AUTO_GRID_SIZE} "
                f"(log-strike margin {params.log_strike_margin:.1f}); increase log_strike_step, "
                "alias_tol or tail_rate, or set grid_size explicitly."
            )
            raise ValueError(msg)
    else:
        _validate_grid_size(grid_size)

    log_strike_half_width = params.log_strike_step * grid_size / 2.0
    frequency_step = 2.0 * pi / (params.log_strike_step * grid_size)
    j_idx = jnp.arange(grid_size, dtype=float)
    frequency = frequency_step * j_idx
    log_strike = -log_strike_half_width + params.log_strike_step * j_idx

    # Trapezoid weights: for integrands analytic near the real axis the trapezoid rule is
    # spectrally accurate, whereas Simpson = (4 T_eta - T_2eta) / 3 inherits the T_2eta error.
    weights = jnp.full(grid_size, frequency_step, dtype=float)
    weights = weights.at[0].set(frequency_step / 2.0)

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
        frequency=frequency,
        shifted_frequency=frequency - 1j * (params.damping + 1.0),
        log_strike=log_strike,
        strike=jnp.exp(log_strike),
        weights=weights,
        phase=jnp.exp(1j * log_strike_half_width * frequency),
        denominator=denominator,
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


def check_strikes_in_grid(strike: ArrayLike | jax.Array, grid: JaxFFTCallGrid) -> None:
    """Raise if concrete strikes fall outside the grid; ``jnp.interp`` would clamp them silently.

    Skipped when the strikes or the grid are traced so it can sit on ``jit``-compatible paths.
    The comparison runs on the host because ``jnp`` ops are staged inside a trace even for
    concrete values.
    """
    if isinstance(strike, jax.core.Tracer) or isinstance(grid.strike, jax.core.Tracer):
        return
    strike_arr = np.asarray(strike, dtype=float)
    if strike_arr.size == 0:
        return
    grid_strike = np.asarray(grid.strike)
    lower, upper = float(grid_strike[0]), float(grid_strike[-1])
    if np.any(strike_arr < lower) or np.any(strike_arr > upper):
        msg = (
            f"strike values must lie within the FFT grid [{lower:.6g}, {upper:.6g}]; "
            f"got [{float(strike_arr.min()):.6g}, {float(strike_arr.max()):.6g}]."
        )
        raise ValueError(msg)


def _fft_call_price_from_cf(
    cf: JaxCharacteristicFunction,
    strike: ArrayLike | jax.Array,
    disc: ArrayLike | jax.Array,
    params: JaxFFTCallEngineParams,
    grid: JaxFFTCallGrid,
    control: JaxCallControlVariate | None = None,
) -> jax.Array:
    """Return discounted European call prices by JAX Carr-Madan FFT."""
    _check_grid_matches_params(params=params, grid=grid)
    check_strikes_in_grid(strike, grid)
    return fft_call_price_on_grid(cf=cf, strike=strike, disc=disc, grid=grid, control=control)


def fft_call_price_on_grid(
    cf: JaxCharacteristicFunction,
    strike: ArrayLike | jax.Array,
    disc: ArrayLike | jax.Array,
    grid: JaxFFTCallGrid,
    control: JaxCallControlVariate | None = None,
) -> jax.Array:
    """Pure traceable Carr-Madan call pricing on a precomputed JAX grid.

    Strikes must lie within ``[grid.strike[0], grid.strike[-1]]``; callers with concrete
    strikes should validate with :func:`check_strikes_in_grid` first.
    """
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
        upper = jnp.clip(jnp.searchsorted(grid.strike, strike_arr, side="right"), 1, grid.grid_size - 1)
        lower_strike, upper_strike = grid.strike[upper - 1], grid.strike[upper]
        weight = jnp.clip((strike_arr - lower_strike) / (upper_strike - lower_strike), 0.0, 1.0)
        lower_control = control.call_price(lower_strike)
        upper_control = control.call_price(upper_strike)
        # Interpolating the control at the same nodes reconstructs interp(residual + control),
        # so the interpolation error scales with the model's own convexity rather than the
        # control's. Adding the exact control price instead leaves an error proportional to the
        # Black density, which exceeds deep-OTM model prices and turns them negative.
        call_price = call_price + (lower_control + weight * (upper_control - lower_control))

    return call_price


@dataclass(frozen=True, slots=True)
class JaxFFTCallEngine:
    """JAX Carr-Madan FFT call engine compatible with ``jax.jit`` and ``jax.vmap``.

    When ``grid`` is provided, ``__call__`` skips grid construction and is
    safe to trace under ``jax.jit``; otherwise the grid is built per call
    from concrete ``strike`` values outside tracing.
    """

    cf: JaxCharacteristicFunction
    disc: ArrayLike | jax.Array
    _: KW_ONLY
    control: JaxCallControlVariate | None = None
    params: JaxFFTCallEngineParams = field(default_factory=JaxFFTCallEngineParams)
    grid: JaxFFTCallGrid | None = None

    def __call__(self, strike: ArrayLike | jax.Array) -> jax.Array:
        grid = self.grid if self.grid is not None else make_jax_fft_call_grid(strike=strike, params=self.params)
        return _fft_call_price_from_cf(
            cf=self.cf,
            strike=strike,
            disc=self.disc,
            params=self.params,
            grid=grid,
            control=self.control,
        )
