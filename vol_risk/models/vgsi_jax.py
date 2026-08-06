r"""JAX-traceable VGSI pricing for Carr-Madan FFT under factor VG dynamics.

JIT- and ``jacfwd``/``jacrev``-friendly counterpart of
:mod:`vol_risk.models.vgsi`. Quadrature is intentionally not supported.

Designed for calibration loops where the FFT grid, strikes, time-to-maturity,
forwards, and VG parameters all vary at run-time: every numeric argument is
traced through ``jnp.asarray`` and no Python-level branch fires on traced
values, so a jitted pricer does not retrace across maturities or smiles as
long as array shapes are stable. No ``jax.lax.cond`` is used; the Brownian
limit ``nu -> 0`` is not handled in the JAX path (caller must pass strictly
positive ``nu``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from vol_risk.models.numerical.fourier.fft_jax import (
    JaxControlVariate,
    JaxFFTCallGrid,
    fft_call_price_on_grid,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

ControlName = Literal["none", "bs"]


@dataclass(frozen=True, slots=True)
class VGSIJaxParams:
    r"""JAX-traceable VG parameters $(\sigma, \nu, \theta)$.

    Registered as a JAX pytree with ``(sigma, nu, theta)`` as children so the
    dataclass can carry tracers and be used as a ``jax.jit`` / ``jax.jacfwd``
    input. Validation is the caller's responsibility inside traced regions.
    """

    sigma: jax.Array
    nu: jax.Array
    theta: jax.Array


jax.tree_util.register_pytree_node(
    VGSIJaxParams,
    lambda p: ((p.sigma, p.nu, p.theta), None),
    lambda _aux, children: VGSIJaxParams(*children),
)


def _vg_mgf_base(order: jax.Array, sigma: jax.Array, nu: jax.Array, theta: jax.Array) -> jax.Array:
    return 1.0 - theta * nu * order - 0.5 * sigma * sigma * nu * order * order


def warn_vgsi_fft_moment_condition(
    *,
    index_sigma: ArrayLike,
    index_nu: ArrayLike,
    index_theta: ArrayLike,
    factor_loading: ArrayLike,
    residual_sigma: ArrayLike,
    residual_nu: ArrayLike,
    residual_theta: ArrayLike,
    damping: float,
    context: str = "VGSI FFT",
) -> None:
    """Warn once when concrete VG parameter arrays do not support the FFT contour.

    This host-only preflight is intended to run before a JIT-compiled pricing
    function is called. It performs no JAX operations and therefore adds
    nothing to the compiled pricing loop.
    """
    if not np.isfinite(damping) or damping <= 0.0:
        msg = "damping must be finite and positive."
        raise ValueError(msg)

    order = damping + 1.0
    beta = np.asarray(factor_loading, dtype=float)
    idx_sigma = beta * np.asarray(index_sigma, dtype=float)
    idx_nu = np.asarray(index_nu, dtype=float)
    idx_theta = beta * np.asarray(index_theta, dtype=float)
    res_sigma = np.asarray(residual_sigma, dtype=float)
    res_nu = np.asarray(residual_nu, dtype=float)
    res_theta = np.asarray(residual_theta, dtype=float)

    index_base = 1.0 - idx_theta * idx_nu * order - 0.5 * idx_sigma * idx_sigma * idx_nu * order * order
    residual_base = 1.0 - res_theta * res_nu * order - 0.5 * res_sigma * res_sigma * res_nu * order * order
    index_base, residual_base = np.broadcast_arrays(index_base, residual_base)
    invalid_index = ~np.isfinite(index_base) | (index_base <= 0.0)
    invalid_residual = ~np.isfinite(residual_base) | (residual_base <= 0.0)
    invalid = invalid_index | invalid_residual
    invalid_count = int(np.count_nonzero(invalid))
    if invalid_count == 0:
        return

    warnings.warn(
        f"{context}: damping {damping:g} requires finite exponential moments of order {order:g}; "
        f"{invalid_count} of {invalid.size} parameter sets violate the condition "
        f"(index={np.count_nonzero(invalid_index)}, residual={np.count_nonzero(invalid_residual)}). "
        "FFT prices may violate no-arbitrage bounds; use a smaller damping or constrain the parameters.",
        RuntimeWarning,
        stacklevel=2,
    )


def _vg_compensator(params: VGSIJaxParams, scale: jax.Array | float = 1.0) -> jax.Array:
    r"""Return the VG martingale compensator $\delta = \log\phi_{VG}(-i)/\nu$.

    The VG branch divides by ``nu``; pass strictly positive ``nu`` (the
    Brownian limit is not supported in the JAX path and would yield NaN).
    """
    sigma = scale * params.sigma
    theta = scale * params.theta
    base = _vg_mgf_base(jnp.asarray(1.0), sigma, params.nu, theta)
    return jnp.log(base) / params.nu


def _vgsi_compensator(
    index_params: VGSIJaxParams,
    factor_loading: jax.Array,
    residual_params: VGSIJaxParams,
) -> jax.Array:
    r"""Carr-Madan VGSI compensator $\delta$ (Eq. 6, p. 323)."""
    return _vg_compensator(index_params, factor_loading) + _vg_compensator(residual_params, 1.0)


def _vg_cf(
    u: jax.Array,
    tau: jax.Array,
    params: VGSIJaxParams,
    scale: jax.Array | float = 1.0,
) -> jax.Array:
    r"""VG component CF $E[e^{iu\,\mathrm{scale}\,X_\tau}]$ with no Python branches."""
    sigma = scale * params.sigma
    theta = scale * params.theta
    base = 1.0 - 1j * theta * params.nu * u + 0.5 * sigma * sigma * params.nu * u * u
    return jnp.power(base, -tau / params.nu)


def vgsi_cf(
    u: jax.Array,
    spot: jax.Array,
    r: jax.Array,
    q: jax.Array,
    tau: jax.Array,
    index_params: VGSIJaxParams,
    factor_loading: jax.Array,
    residual_params: VGSIJaxParams,
) -> jax.Array:
    r"""Return the log-stock CF $\phi(u)$ at already-traced inputs."""
    delta = _vgsi_compensator(index_params, factor_loading, residual_params)
    drift = jnp.log(spot) + (r - q + delta) * tau
    return jnp.exp(1j * u * drift) * _vg_cf(u, tau, index_params, factor_loading) * _vg_cf(u, tau, residual_params, 1.0)


def make_vgsi_cf(
    spot: jax.Array,
    r: jax.Array,
    q: jax.Array,
    tau: jax.Array,
    index_params: VGSIJaxParams,
    factor_loading: jax.Array,
    residual_params: VGSIJaxParams,
) -> Callable[[jax.Array], jax.Array]:
    """Return a closure ``cf(u)`` evaluating the JAX VGSI log-stock CF."""

    def cf(u: jax.Array) -> jax.Array:
        return vgsi_cf(
            u=u,
            spot=spot,
            r=r,
            q=q,
            tau=tau,
            index_params=index_params,
            factor_loading=factor_loading,
            residual_params=residual_params,
        )

    return cf


def _control_sigma(
    index_params: VGSIJaxParams,
    factor_loading: jax.Array,
    residual_params: VGSIJaxParams,
) -> jax.Array:
    beta_sq = factor_loading * factor_loading
    idx_var = index_params.sigma * index_params.sigma + index_params.theta * index_params.theta * index_params.nu
    res_var = (
        residual_params.sigma * residual_params.sigma
        + residual_params.theta * residual_params.theta * residual_params.nu
    )
    return jnp.sqrt(beta_sq * idx_var + res_var)


def _make_bs_control(
    spot: jax.Array,
    r: jax.Array,
    q: jax.Array,
    tau: jax.Array,
    sigma: jax.Array,
) -> JaxControlVariate:
    """Traceable Black-Scholes control variate (Joshi-Yang, 2011).

    Mirrors :func:`vol_risk.models.numerical.control_variates.make_bs_control`
    without the ``float()`` coercions, so ``tau``, ``r``, ``q``, ``spot``,
    and ``sigma`` may be traced and varied across calibration calls without
    triggering JIT recompilation.
    """
    log_spot = jnp.log(spot)
    drift = log_spot + (r - q - 0.5 * sigma * sigma) * tau
    half_var = 0.5 * sigma * sigma * tau
    fwd = spot * jnp.exp((r - q) * tau)
    disc = jnp.exp(-r * tau)
    sig_sqrt_t = sigma * jnp.sqrt(tau)

    def cf(u: jax.Array) -> jax.Array:
        return jnp.exp(1j * u * drift - half_var * u * u)

    def call_price(strike: jax.Array) -> jax.Array:
        k = jnp.asarray(strike)
        d1 = (jnp.log(fwd / k) + half_var) / sig_sqrt_t
        d2 = d1 - sig_sqrt_t
        return disc * (fwd * norm.cdf(d1) - k * norm.cdf(d2))

    return JaxControlVariate(cf=cf, call_price=call_price)


def vgsi_price_jax(
    spot: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    index_params: VGSIJaxParams,
    factor_loading: ArrayLike,
    residual_params: VGSIJaxParams,
    *,
    r: ArrayLike = 0.0,
    q: ArrayLike = 0.0,
    is_call: bool | ArrayLike = True,
    control: ControlName = "bs",
    grid: JaxFFTCallGrid,
) -> jax.Array:
    r"""JAX VGSI European-option pricer on a precomputed Carr-Madan FFT grid.

    All numeric arguments (``spot``, ``strike``, ``tau``, ``r``, ``q``,
    ``factor_loading``) and the fields of ``index_params`` /
    ``residual_params`` are traced. Inside ``jax.jit`` the caller may vary
    maturity, strike grid, and parameters without recompilation as long as
    array shapes are stable. The FFT grid must be precomputed outside
    ``jit`` with :func:`make_jax_fft_call_grid` covering the strike range.
    """
    spot_a = jnp.asarray(spot)
    tau_a = jnp.asarray(tau)
    r_a = jnp.asarray(r)
    q_a = jnp.asarray(q)
    beta_a = jnp.asarray(factor_loading)
    strike_a = jnp.asarray(strike)

    cf = make_vgsi_cf(
        spot=spot_a,
        r=r_a,
        q=q_a,
        tau=tau_a,
        index_params=index_params,
        factor_loading=beta_a,
        residual_params=residual_params,
    )

    cv: JaxControlVariate | None = None
    if control == "bs":
        sigma_ctrl = _control_sigma(index_params, beta_a, residual_params)
        cv = _make_bs_control(spot_a, r_a, q_a, tau_a, sigma_ctrl)
    elif control != "none":
        msg = f"Unsupported control variate strategy: {control!r}"
        raise ValueError(msg)

    disc = jnp.exp(-r_a * tau_a)
    call_price = fft_call_price_on_grid(
        cf=cf,
        strike=strike_a,
        disc=disc,
        grid=grid,
        control=cv,
    )
    call_mask = jnp.asarray(is_call, dtype=bool)
    put_price = call_price - spot_a * jnp.exp(-q_a * tau_a) + strike_a * jnp.exp(-r_a * tau_a)
    return jnp.where(call_mask, call_price, put_price)


__all__ = [
    "ControlName",
    "VGSIJaxParams",
    "_vg_compensator",
    "_vgsi_compensator",
    "make_vgsi_cf",
    "vgsi_cf",
    "vgsi_price_jax",
    "warn_vgsi_fft_moment_condition",
]
