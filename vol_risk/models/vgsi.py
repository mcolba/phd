r"""Variance Gamma Stock Index (VGSI) pricing under Carr-Madan factor dynamics.

NumPy-only implementation supporting both Carr-Madan FFT and adaptive
quadrature engines. The JAX-traceable variant lives in
:mod:`vol_risk.models.vgsi_jax`.

Implements the characteristic function and Carr-Madan modified-call Fourier
pricing for the two-factor variance-gamma stock model of:

    Carr, P., & Madan, D. B. (2012). Factor Models for Option Pricing.
    Asia-Pacific Financial Markets, 19(4), 319-329.
    https://doi.org/10.1007/s10690-011-9151-7

The risk-neutral log-stock dynamics combine a market-index VG factor scaled by
a loading $\beta$ with an independent idiosyncratic VG component, compensated
so the discounted stock is a martingale (Eq. 5-6 of the reference).

The default damping $\alpha = 0.03$ and the quadrature truncation breakpoints
$(8, 64, 256)$ are taken from the MATLAB reference implementation listed in
the appendix of the preprint DOI:10.1007/s10690-011-9151-7.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from vol_risk.models.numerical.control_variates import make_bs_control
from vol_risk.models.numerical.fourier.base import QuadCallEngine, QuadEngineParams
from vol_risk.models.numerical.fourier.fft_np import FFTCallEngine, FFTEngineParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

_SMALL_NU = 1.0e-12

EngineName = Literal["auto", "quad", "fft_np"]
ControlName = Literal["none", "bs"]
EngineParams: TypeAlias = QuadEngineParams | FFTEngineParams


@dataclass(frozen=True, slots=True)
class VGSIParams:
    r"""Variance Gamma parameters $(\sigma, \nu, \theta)$.

    Attributes:
        sigma: Diffusion volatility of the Brownian component. Non-negative.
        nu: Variance rate of the gamma time change. Non-negative; values
            below ``1e-12`` are treated as the Brownian limit.
        theta: Drift of the Brownian component under the gamma time change.
    """

    sigma: float
    nu: float
    theta: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma < 0.0:
            msg = "VGSIParams.sigma must be finite and non-negative."
            raise ValueError(msg)
        if not np.isfinite(self.nu) or self.nu < 0.0:
            msg = "VGSIParams.nu must be finite and non-negative."
            raise ValueError(msg)
        if not np.isfinite(self.theta):
            msg = "VGSIParams.theta must be finite."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FactorMomentFit:
    r"""OLS factor fit with residual symmetric-VG moment estimates.

    Attributes:
        intercept: Estimated regression intercept (per-period drift).
        factor_loading: Estimated factor loading $\beta$ on index returns.
        residual_sigma: Annualised residual volatility.
        residual_nu: Annualised residual VG variance rate, derived from the
            excess kurtosis of the regression residuals.
        residual_theta: Residual VG skew; zero under the symmetric residual
            assumption of Carr-Madan (2012).
    """

    intercept: float
    factor_loading: float
    residual_sigma: float
    residual_nu: float
    residual_theta: float = 0.0

    @property
    def residual_params(self) -> VGSIParams:
        """Return the residual VG parameters implied by the fit."""
        return VGSIParams(
            sigma=self.residual_sigma,
            nu=self.residual_nu,
            theta=self.residual_theta,
        )


def _validate_market_inputs(
    spot: float,
    strike: ArrayLike,
    r: float,
    q: float,
    tau: float,
    factor_loading: float,
) -> np.ndarray:
    """Validate scalar market inputs and normalise strikes to an ndarray."""
    if not np.isfinite(spot) or spot <= 0.0:
        msg = "spot must be finite and positive."
        raise ValueError(msg)

    strike_arr = np.asarray(strike, dtype=float)
    if np.any(~np.isfinite(strike_arr)) or np.any(strike_arr <= 0.0):
        msg = "strike values must be finite and positive."
        raise ValueError(msg)

    if not np.isfinite(r):
        msg = "r must be finite."
        raise ValueError(msg)
    if not np.isfinite(q):
        msg = "q must be finite."
        raise ValueError(msg)
    if not np.isfinite(tau) or tau < 0.0:
        msg = "tau must be finite and non-negative."
        raise ValueError(msg)
    if not np.isfinite(factor_loading):
        msg = "factor_loading must be finite."
        raise ValueError(msg)

    return strike_arr


def _vg_mgf_base(order: float, params: VGSIParams, scale: float) -> float:
    """Return the VG MGF denominator; ``+inf`` in the Brownian limit."""
    scaled_sigma = scale * params.sigma
    scaled_theta = scale * params.theta
    return 1.0 - scaled_theta * params.nu * order - 0.5 * scaled_sigma * scaled_sigma * params.nu * order * order


def _check_moment(order: float, params: VGSIParams, scale: float, label: str) -> None:
    """Ensure the required exponential moment of a VG component exists."""
    base = _vg_mgf_base(order=order, params=params, scale=scale)
    if np.isfinite(base) and base <= 0.0:
        msg = f"The {label} VG component has no finite exponential moment of order {order:g}."
        raise ValueError(msg)


def _vg_compensator(params: VGSIParams, scale: float = 1.0) -> float:
    r"""VG compensator $\delta$; raises when the unit moment is not finite."""
    scaled_sigma = scale * params.sigma
    scaled_theta = scale * params.theta

    if params.nu <= _SMALL_NU:
        return -(scaled_theta + 0.5 * scaled_sigma * scaled_sigma)

    base = _vg_mgf_base(order=1.0, params=params, scale=scale)
    if base <= 0.0:
        msg = "The unit exponential moment is not finite; martingale compensation fails."
        raise ValueError(msg)

    return float(np.log(base) / params.nu)


def _vgsi_compensator(
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
) -> float:
    r"""Return the Carr-Madan VGSI compensator $\delta$ (Eq. 6, p. 323)."""
    return _vg_compensator(index_params, factor_loading) + _vg_compensator(residual_params, 1.0)


def _vg_cf(u: np.ndarray, tau: float, params: VGSIParams, scale: float = 1.0) -> np.ndarray:
    r"""VG component CF $E[e^{i u\,\mathrm{scale}\,X_\tau}]$.

    The VG CF has no rotating branch cut on the strip of admissibility,
    see Lord & Kahl (2010).
    """
    if tau == 0.0:
        return np.ones_like(u, dtype=complex)

    scaled_sigma = scale * params.sigma
    scaled_theta = scale * params.theta

    if params.nu <= _SMALL_NU:
        exponent = tau * (1j * scaled_theta * u - 0.5 * scaled_sigma * scaled_sigma * u * u)
        return np.exp(exponent)

    base = 1.0 - 1j * scaled_theta * params.nu * u + 0.5 * scaled_sigma * scaled_sigma * params.nu * u * u
    return np.power(base, -tau / params.nu)


def _vgsi_cf(
    u: np.ndarray,
    spot: float,
    r: float,
    q: float,
    tau: float,
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
) -> np.ndarray:
    r"""Return $\phi(u)$ from already-validated inputs."""
    delta = _vgsi_compensator(index_params, factor_loading, residual_params)
    drift = np.log(spot) + (r - q + delta) * tau
    return (
        np.exp(1j * u * drift)
        * _vg_cf(u=u, tau=tau, params=index_params, scale=factor_loading)
        * _vg_cf(u=u, tau=tau, params=residual_params, scale=1.0)
    )


def _vgsi_log_variance(
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
) -> float:
    r"""Per-unit-time variance of $\log S_T$ used for the BS control sigma."""
    beta = float(factor_loading)
    idx_var = index_params.sigma**2 + index_params.theta**2 * index_params.nu
    res_var = residual_params.sigma**2 + residual_params.theta**2 * residual_params.nu
    return beta * beta * idx_var + res_var


def make_vgsi_cf(
    spot: float,
    r: float,
    q: float,
    tau: float,
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a closure evaluating the VGSI log-stock CF on NumPy arrays."""
    return partial(
        _vgsi_cf,
        spot=spot,
        r=r,
        q=q,
        tau=tau,
        index_params=index_params,
        factor_loading=factor_loading,
        residual_params=residual_params,
    )


def _control_sigma(
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
) -> float:
    return float(
        np.sqrt(
            _vgsi_log_variance(
                index_params=index_params,
                factor_loading=factor_loading,
                residual_params=residual_params,
            )
        )
    )


def _vgsi_call_price(
    spot: float,
    strike: ArrayLike,
    tau: float,
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
    *,
    r: float = 0.0,
    q: float = 0.0,
    engine: EngineName = "auto",
    engine_opt: dict | None = None,
    control: ControlName = "bs",
) -> np.ndarray | float:
    """Price European calls under VGSI via a pluggable NumPy Fourier engine.

    Internal call-only entry point used by :func:`vgsi_price`.
    """
    _validate_market_inputs(spot=spot, strike=1.0, r=r, q=q, tau=tau, factor_loading=factor_loading)

    cf = make_vgsi_cf(
        spot=spot,
        r=r,
        q=q,
        tau=tau,
        index_params=index_params,
        factor_loading=factor_loading,
        residual_params=residual_params,
    )

    strike_arr = _validate_market_inputs(
        spot=spot,
        strike=strike,
        r=r,
        q=q,
        tau=tau,
        factor_loading=factor_loading,
    )
    scalar_input = strike_arr.ndim == 0
    strike_1d = np.atleast_1d(strike_arr).astype(float, copy=False)

    if tau == 0.0:
        intrinsic = np.maximum(spot - strike_1d, 0.0)
        out = intrinsic if scalar_input else intrinsic.reshape(strike_arr.shape)
        return float(out[0]) if scalar_input else out

    cv = None
    if control == "bs":
        sigma_ctrl = _control_sigma(
            index_params=index_params,
            factor_loading=factor_loading,
            residual_params=residual_params,
        )
        cv = make_bs_control(spot, r, q, tau, sigma_ctrl, backend="np")
    elif control != "none":
        msg = f"Unsupported control variate strategy: {control!r}"
        raise ValueError(msg)

    if engine == "auto":
        engine = "quad" if strike_1d.size <= 1 else "fft_np"

    disc = np.exp(-r * tau)

    if engine == "quad":
        par = QuadEngineParams(**(engine_opt or {}))
        engine_obj = QuadCallEngine(cf=cf, control=cv, params=par, disc=disc, tau=tau)
    elif engine == "fft_np":
        par = FFTEngineParams(**(engine_opt or {}))
        engine_obj = FFTCallEngine(cf=cf, control=cv, params=par, disc=disc)
    else:
        msg = f"Unsupported engine: {engine!r}"
        raise ValueError(msg)

    damping = engine_obj.params.damping
    _check_moment(order=damping + 1.0, params=index_params, scale=factor_loading, label="index")
    _check_moment(order=damping + 1.0, params=residual_params, scale=1.0, label="residual")

    prices = engine_obj(strike_1d)
    return float(prices[0]) if scalar_input else prices.reshape(strike_arr.shape)


def vgsi_price(
    spot: float,
    strike: ArrayLike,
    tau: float,
    index_params: VGSIParams,
    factor_loading: float,
    residual_params: VGSIParams,
    *,
    r: float = 0.0,
    q: float = 0.0,
    is_call: bool | ArrayLike = True,
    engine: EngineName = "auto",
    engine_opt: EngineParams | None = None,
    control: ControlName = "bs",
) -> np.ndarray | float:
    r"""Price European options under the VGSI model via NumPy engines.

    Args:
        spot: Spot price $S_0$.
        strike: Strike or array of strikes (positive, finite).
        tau: Time to maturity in years.
        index_params: Market-index VG parameters.
        factor_loading: Stock loading $\beta$ on the market factor.
        residual_params: Idiosyncratic residual VG parameters.
        r: Risk-free rate.
        q: Dividend yield.
        is_call: Boolean (or boolean array broadcastable with ``strike``)
            selecting call (``True``) versus put (``False``).
        engine: Numerical engine; ``"auto"`` picks ``"quad"`` for a single
            strike and ``"fft_np"`` otherwise.
        engine_opt: Optional engine configuration overriding the VGSI
            defaults. Must match the selected engine's dataclass.
        control: Control-variate strategy. ``"bs"`` (default) uses a
            variance-matched Black-Scholes leg (Joshi-Yang, 2011); ``"none"``
            integrates the model CF directly.

    Returns:
        Scalar or array of discounted option prices.
    """
    strike_arr = _validate_market_inputs(
        spot=spot,
        strike=strike,
        r=r,
        q=q,
        tau=tau,
        factor_loading=factor_loading,
    )
    scalar_input = strike_arr.ndim == 0
    call_price = _vgsi_call_price(
        spot=spot,
        strike=strike_arr,
        r=r,
        q=q,
        tau=tau,
        index_params=index_params,
        factor_loading=factor_loading,
        residual_params=residual_params,
        engine=engine,
        engine_opt=engine_opt,
        control=control,
    )
    call_arr = np.asarray(call_price, dtype=float)
    call_mask = np.broadcast_to(np.asarray(is_call, dtype=bool), strike_arr.shape)
    put_arr = call_arr - spot * np.exp(-q * tau) + strike_arr * np.exp(-r * tau)
    price = np.where(call_mask, call_arr, put_arr)
    return float(price) if scalar_input else price


def estimate_factor_residual_moments(
    stock_log_returns: ArrayLike,
    index_log_returns: ArrayLike,
    *,
    periods_per_year: float = 252.0,
) -> FactorMomentFit:
    r"""Estimate the factor loading and residual symmetric-VG moments.

    Follows the method-of-moments described in Carr-Madan (2012), Sec. 3.1:
    OLS on log returns gives $\beta$, residual volatility is the sample
    standard deviation, and residual $\nu$ is the annualised excess kurtosis.

    Args:
        stock_log_returns: One-dimensional array of stock log returns.
        index_log_returns: One-dimensional array of index log returns, same
            length as ``stock_log_returns``.
        periods_per_year: Annualisation factor (e.g. ``252`` for daily data).

    Returns:
        A :class:`FactorMomentFit` with the regression intercept, factor
        loading, and annualised residual VG moments (with zero skew).
    """
    stock = np.asarray(stock_log_returns, dtype=float)
    index = np.asarray(index_log_returns, dtype=float)
    if stock.shape != index.shape or stock.ndim != 1:
        msg = "stock_log_returns and index_log_returns must be one-dimensional arrays of equal length."
        raise ValueError(msg)
    if stock.size < 3:
        msg = "at least three return observations are required."
        raise ValueError(msg)
    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        msg = "periods_per_year must be finite and positive."
        raise ValueError(msg)
    if np.any(~np.isfinite(stock)) or np.any(~np.isfinite(index)):
        msg = "return arrays must contain only finite values."
        raise ValueError(msg)

    design = np.column_stack([np.ones_like(index), index])
    intercept, factor_loading = np.linalg.lstsq(design, stock, rcond=None)[0]
    residual = stock - intercept - factor_loading * index
    variance = float(np.mean(residual * residual))
    residual_sigma = float(np.sqrt(variance * periods_per_year))
    if variance <= 0.0:
        residual_nu = 0.0
    else:
        fourth_moment = float(np.mean(residual**4))
        excess_ratio = fourth_moment / (3.0 * variance * variance) - 1.0
        residual_nu = float(max(excess_ratio / periods_per_year, 0.0))

    return FactorMomentFit(
        intercept=float(intercept),
        factor_loading=float(factor_loading),
        residual_sigma=residual_sigma,
        residual_nu=residual_nu,
        residual_theta=0.0,
    )


__all__ = [
    "ControlName",
    "EngineName",
    "EngineParams",
    "FactorMomentFit",
    "VGSIParams",
    "_vg_compensator",
    "_vgsi_compensator",
    "estimate_factor_residual_moments",
    "make_vgsi_cf",
    "vgsi_price",
]
