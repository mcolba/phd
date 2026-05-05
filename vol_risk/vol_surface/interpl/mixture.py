import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special
from scipy.optimize import least_squares

from vol_risk.calibration.option_chain import NoArbBounds
from vol_risk.calibration.transformers import get_atmf_vol
from vol_risk.models.black76 import (
    black76_price,
    black76_undisc_fwd_delta,
    black76_vega,
    implied_black_vol,
)
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.protocols import ModelParams, OptionChainLike
from vol_risk.util import (
    angles_to_simplex,
    angles_to_simplex_jac,
    make_ravel_param_jac,
    simplex_to_angles,
)
from vol_risk.vol_surface.surface import VolSmile, VolSurface

log = logging.getLogger(__name__)

SIGMA_MAX = 4.0
SIGMA_MIN = 0.03
THETA_1_EPSILON = 0.05
THETA_2_EPSILON = 0.15


@dataclass(frozen=True, slots=True)
class LogNormMixParams(ModelParams):
    """Parameters for the log-normal mixture model.

    Attributes:
        w: Mixture component weights.
        fwd_scale: Forward-scale factors.
        sigma: Black-76 volatilities.
    """

    w: np.ndarray
    fwd_scale: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        """Validates parameters."""
        if not (len(self.w) == len(self.fwd_scale) == len(self.sigma)):
            msg = "Parameters 'w', 'fwd_scale', and 'sigma' must have the same length."
            raise ValueError(msg)

        if not np.all(self.w >= 0):
            msg = "All weights 'w' must be non-negative."
            raise ValueError(msg)

        if not np.isclose(np.sum(self.w), 1.0):
            msg = "The sum of weights 'w' must be equal to 1."
            raise ValueError(msg)

        if not np.all(self.fwd_scale > 0):
            msg = "All forward-scale factors must be positive."
            raise ValueError(msg)

        if not np.isclose(np.dot(self.w, self.fwd_scale), 1.0):
            msg = "Martingale constraint violated: sum(w * fwd_scale) must equal 1."
            raise ValueError(msg)

    def mu(self, tau: float) -> np.ndarray:
        """Recover drift parameters for a given time-to-expiry."""
        return np.log(self.fwd_scale) / tau


@dataclass(frozen=True, slots=True)
class LogNormMixCalibParams:
    """Parameters for the log-normal mixture calibration model."""

    bijection_factory: Callable
    weight_function: Callable
    lambda_rough: float
    lambda_w: float
    lambda_mu: float
    lambda_sigma: float
    x0: LogNormMixParams | None


def _param_slices(n: int) -> tuple[slice, slice, slice]:
    """Return slices into stacked parameter vector [w | fwd_scale | sigma]."""
    return slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)


def _mixed_log_norm_opt_price(
    w: ArrayLike,
    fwd_scale: ArrayLike,
    sigma: ArrayLike,
    disc: ArrayLike,
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    is_call: ArrayLike = True,
    pdef: float = 0.0,
) -> np.ndarray:
    """Vectorized log-normal mixture option price."""
    w = np.asarray(w, dtype=float)
    fwd_scale = np.asarray(fwd_scale, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    strike, tau, disc, fwd, is_call = np.broadcast_arrays(
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(disc, dtype=float),
        np.asarray(fwd, dtype=float),
        np.asarray(is_call, dtype=bool),
    )

    return (1 - pdef) * np.sum(
        w[i]
        * black76_price(
            disc=disc,
            fwd=fwd * fwd_scale[i] / (1 - pdef),
            strike=strike,
            tau=tau,
            sigma=sigma[i],
            is_call=is_call,
        )
        for i in range(len(w))
    )


def _mixed_log_norm_opt_jac(
    w: ArrayLike,
    fwd_scale: ArrayLike,
    sigma: ArrayLike,
    disc: ArrayLike,
    fwd: ArrayLike,
    strike: ArrayLike,
    tau: ArrayLike,
    is_call: ArrayLike,
    pdef: float = 0.0,
) -> np.ndarray:
    """Analytical Jacobian of mixture option price w.r.t. stacked [w | fwd_scale | sigma]."""
    w, fwd_scale, sigma = [np.asarray(x, dtype=float).reshape(-1, 1) for x in (w, fwd_scale, sigma)]

    if not w.shape == fwd_scale.shape == sigma.shape:
        msg = "Parameters 'w', 'fwd_scale', and 'sigma' must have the same shape."
        raise ValueError(msg)

    if pdef < 0 or pdef >= 1 - 1e-8:
        msg = "Invalid pdef: must be in [0, 1)."
        raise ValueError(msg)

    strike, tau, disc, fwd, is_call = [
        np.atleast_1d(x).reshape(1, -1)
        for x in np.broadcast_arrays(
            np.asarray(strike, dtype=float),
            np.asarray(tau, dtype=float),
            np.asarray(disc, dtype=float),
            np.asarray(fwd, dtype=float),
            np.asarray(is_call, dtype=bool),
        )
    ]

    F_comp = fwd * fwd_scale / (1.0 - pdef)

    n, m = w.size, strike.size
    s_w, s_fs, s_sig = _param_slices(n)

    price = black76_price(fwd=F_comp, strike=strike, tau=tau, sigma=sigma, disc=disc, is_call=is_call)
    delta = black76_undisc_fwd_delta(fwd=F_comp, strike=strike, tau=tau, sigma=sigma, is_call=is_call)
    vega = black76_vega(fwd=F_comp, strike=strike, tau=tau, sigma=sigma, disc=disc)

    non_zero_mass = 1.0 - pdef
    jac = np.zeros((m, 3 * n), dtype=float)

    jac[:, s_w] = (non_zero_mass * price).T
    jac[:, s_fs] = (w * disc * delta * fwd).T
    jac[:, s_sig] = (non_zero_mass * w * vega).T

    return jac


def mixed_log_norm_price(
    params: LogNormMixParams,
    disc: ArrayLike,
    fwd: ArrayLike,
    tau: ArrayLike,
    strike: ArrayLike,
    is_call: ArrayLike,
) -> np.ndarray:
    """Public API: option prices under a log-normal mixture model.

    Args:
        params: Calibrated mixture parameters.
        disc: Discount factor.
        fwd: Forward price.
        strike: Array of strikes.
        tau: Time to expiry.
        is_call: Call/put flag(s).

    Returns:
        Array of option prices.
    """
    return _mixed_log_norm_opt_price(
        w=params.w,
        fwd_scale=params.fwd_scale,
        sigma=params.sigma,
        disc=disc,
        fwd=fwd,
        strike=strike,
        tau=tau,
        is_call=is_call,
        pdef=0.0,
    )


def make_full_encoder(tau: float, method: str = "simplex") -> tuple:
    """Creates a bijection for log-normal mixture calibration parameters."""
    tau = float(tau)

    def encode(params: LogNormMixParams) -> tuple:
        w, fwd_scale, sigma = params.w, params.fwd_scale, params.sigma
        z = w * fwd_scale

        if not (np.isclose(np.sum(w), 1.0) and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (np.isclose(np.sum(z), 1.0) and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        x0 = simplex_to_angles(w)

        if method == "simplex":
            x1 = simplex_to_angles(z)
        elif method == "manual":
            x1 = fwd_scale[:-1]
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        free = (x0, x1, sigma)
        return (free, ())

    def decode(free: tuple[ArrayLike], _: tuple[ArrayLike] | None) -> LogNormMixParams:
        x0, x1, sigma = free

        w = angles_to_simplex(x0)

        if method == "simplex":
            z = angles_to_simplex(x1)
            fwd_scale = z / w
        elif method == "manual":
            fwd_scale_free = x1
            partial_sum = np.dot(w[:-1], fwd_scale_free)
            if (1 - partial_sum) <= 0:
                msg = "Invalid parameters: remaining forward mass <= 0. Use simplex method instead."
                raise ValueError(msg)
            fwd_scale_n = (1.0 - partial_sum) / w[-1]
            fwd_scale = np.append(fwd_scale_free, fwd_scale_n)
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        return LogNormMixParams(w=w, fwd_scale=fwd_scale, sigma=sigma)

    def jac_decode(free: list[np.ndarray], _: tuple) -> np.ndarray:
        """Jacobian d[w, fwd_scale, sigma] / d[flat_x] with shape (3n, n_flat)."""
        x0, x1, sigma = free
        n = len(x0) + 1
        s_w, s_fs, s_sig = _param_slices(n)
        n_flat = len(x0) + len(x1) + len(sigma)
        jac = np.zeros((3 * n, n_flat), dtype=float)

        sl_x0 = slice(0, len(x0))
        sl_x1 = slice(len(x0), len(x0) + len(x1))
        sl_sig = slice(len(x0) + len(x1), n_flat)

        dw_dx0 = angles_to_simplex_jac(x0)
        jac[s_w, sl_x0] = dw_dx0

        w = angles_to_simplex(x0)

        if method == "simplex":
            z = angles_to_simplex(x1)
            dz_dx1 = angles_to_simplex_jac(x1)
            fwd_scale = z / w

            jac[s_fs, sl_x0] = -(fwd_scale / w)[:, np.newaxis] * dw_dx0
            jac[s_fs, sl_x1] = (1.0 / w)[:, np.newaxis] * dz_dx1

        elif method == "manual":
            fwd_scale_free = x1
            partial_sum = np.dot(w[:-1], fwd_scale_free)
            fwd_scale_n = (1.0 - partial_sum) / w[-1]
            fwd_scale = np.append(fwd_scale_free, fwd_scale_n)

            # Quotient rule on (1 - w[:-1] @ x1) / w[-1]
            dfn_dx0 = (-(dw_dx0[:-1, :].T @ fwd_scale_free) - fwd_scale_n * dw_dx0[-1, :]) / w[-1]
            jac[s_fs.start + n - 1, sl_x0] = dfn_dx0

            for i in range(n - 1):
                jac[s_fs.start + i, sl_x1.start + i] = 1.0

            for j in range(n - 1):
                jac[s_fs.start + n - 1, sl_x1.start + j] = -w[j] / w[-1]

        jac[s_sig, sl_sig] = np.eye(n)

        return jac

    return (encode, decode, jac_decode)


def make_full_encoder_totvar(tau: float, method: str = "simplex") -> tuple:
    """Creates a bijection with additive total variance parametrisation for sigma."""
    tau = float(tau)

    def encode(params: LogNormMixParams) -> tuple:
        w, fwd_scale, sigma = params.w, params.fwd_scale, params.sigma
        z = w * fwd_scale

        if not (np.isclose(np.sum(w), 1.0) and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (np.isclose(np.sum(z), 1.0) and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        dv = np.zeros_like(sigma, dtype=np.float64)
        x0 = simplex_to_angles(w)

        if method == "simplex":
            x1 = simplex_to_angles(z)
        elif method == "manual":
            x1 = fwd_scale[:-1]
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        v0 = sigma**2 * tau

        free = (x0, x1, dv)
        fixed = (v0,)
        return (free, fixed)

    def decode(free: tuple[ArrayLike], fixed: tuple[ArrayLike]) -> LogNormMixParams:
        x0, x1, dv = free
        v0 = fixed[0]

        sigma = np.sqrt((v0 + dv) / tau)

        w = angles_to_simplex(x0)

        if method == "simplex":
            z = angles_to_simplex(x1)
            fwd_scale = z / w
        elif method == "manual":
            fwd_scale_free = x1
            partial_sum = np.dot(w[:-1], fwd_scale_free)
            if (1 - partial_sum) <= 0:
                msg = "Invalid parameters: remaining forward mass <= 0. Use simplex method instead."
                raise ValueError(msg)
            fwd_scale_n = (1.0 - partial_sum) / w[-1]
            fwd_scale = np.append(fwd_scale_free, fwd_scale_n)
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        return LogNormMixParams(w=w, fwd_scale=fwd_scale, sigma=sigma)

    def jac_decode(free: list[np.ndarray], fixed: tuple) -> np.ndarray:
        """Jacobian d[w, fwd_scale, sigma] / d[flat_x] with shape (3n, n_flat)."""
        x0, x1, dv = free
        v0 = fixed[0]
        n = len(x0) + 1
        s_w, s_fs, s_sig = _param_slices(n)
        n_flat = len(x0) + len(x1) + len(dv)
        jac = np.zeros((3 * n, n_flat), dtype=float)

        sl_x0 = slice(0, len(x0))
        sl_x1 = slice(len(x0), len(x0) + len(x1))
        sl_dv = slice(len(x0) + len(x1), n_flat)

        dw_dx0 = angles_to_simplex_jac(x0)
        jac[s_w, sl_x0] = dw_dx0

        w = angles_to_simplex(x0)

        if method == "simplex":
            z = angles_to_simplex(x1)
            dz_dx1 = angles_to_simplex_jac(x1)
            fwd_scale = z / w

            jac[s_fs, sl_x0] = -(fwd_scale / w)[:, np.newaxis] * dw_dx0
            jac[s_fs, sl_x1] = (1.0 / w)[:, np.newaxis] * dz_dx1

        elif method == "manual":
            fwd_scale_free = x1
            partial_sum = np.dot(w[:-1], fwd_scale_free)
            fwd_scale_n = (1.0 - partial_sum) / w[-1]
            fwd_scale = np.append(fwd_scale_free, fwd_scale_n)

            # Quotient rule on (1 - w[:-1] @ x1) / w[-1]
            dfn_dx0 = (-(dw_dx0[:-1, :].T @ fwd_scale_free) - fwd_scale_n * dw_dx0[-1, :]) / w[-1]
            jac[s_fs.start + n - 1, sl_x0] = dfn_dx0

            for i in range(n - 1):
                jac[s_fs.start + i, sl_x1.start + i] = 1.0

            for j in range(n - 1):
                jac[s_fs.start + n - 1, sl_x1.start + j] = -w[j] / w[-1]

        sigma = np.sqrt((v0 + dv) / tau)
        jac[s_sig, sl_dv] = np.diag(1.0 / (2.0 * tau * sigma))

        return jac

    return (encode, decode, jac_decode)


def make_reduced_encoder(tau: float) -> tuple:
    """Creates a bijection with w and fwd_scale fixed; only sigma is free."""

    def encode(params: LogNormMixParams) -> tuple:
        w, fwd_scale, sigma = params.w, params.fwd_scale, params.sigma
        z = w * fwd_scale

        if not (np.isclose(np.sum(w), 1.0) and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (np.isclose(np.sum(z), 1.0) and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        free = (sigma,)
        fixed = (w, fwd_scale)
        return (free, fixed)

    def decode(free: tuple[ArrayLike], fixed: tuple[ArrayLike]) -> LogNormMixParams:
        (sigma,) = free
        w, fwd_scale = fixed
        sigma = np.squeeze(sigma)
        return LogNormMixParams(w=w, fwd_scale=fwd_scale, sigma=sigma)

    def jac_decode(free: list[np.ndarray], fixed: tuple) -> np.ndarray:
        """Jacobian d[w, fwd_scale, sigma] / d[sigma_flat], shape (3n, n)."""
        (sigma,) = free
        n = len(sigma)
        _, _, s_sig = _param_slices(n)
        jac = np.zeros((3 * n, n), dtype=float)
        jac[s_sig, :] = np.eye(n)
        return jac

    return (encode, decode, jac_decode)


def _require_call_only(chain: OptionChainLike) -> None:
    if not np.all(chain.option_type == "C"):
        msg = "Function expects a call-only chain. Use make_otm_to_call first."
        raise ValueError(msg)


BIJECTION_METHODS = {
    "reduced": make_reduced_encoder,
    "base": lambda x: make_full_encoder(x, method="manual"),
    "simplex": lambda x: make_full_encoder(x, method="simplex"),
    "totvar": lambda x: make_full_encoder_totvar(x, method="manual"),
    "totvar_simplex": lambda x: make_full_encoder_totvar(x, method="simplex"),
}

BIJECTION_1ST_SLICE_FALLBACK = {
    "totvar": "base",
    "totvar_simplex": "simplex",
}

BOUNDS_METHODS = {
    "reduced": lambda n, sigma_min: (np.repeat(sigma_min, n), np.repeat(np.inf, n)),
    "full_mu_unbounded": lambda n, sigma_min: (
        np.concatenate(
            [
                np.repeat(0 + THETA_1_EPSILON, n - 1),
                np.repeat(-np.inf, n - 1),
                np.repeat(sigma_min, n),
            ]
        ),
        np.concatenate(
            [
                np.repeat(np.pi / 2 - THETA_1_EPSILON, n - 1),
                np.repeat(np.inf, n - 1),
                np.repeat(SIGMA_MAX, n),
            ]
        ),
    ),
    "full_mu_bounded": lambda n, sigma_min: (
        np.concatenate(
            [
                np.repeat(0 + THETA_1_EPSILON, n - 1),
                np.repeat(0 + THETA_2_EPSILON, n - 1),
                np.repeat(sigma_min, n),
            ]
        ),
        np.concatenate(
            [
                np.repeat(np.pi / 2 - THETA_1_EPSILON, n - 1),
                np.concatenate([[np.pi / 2 - THETA_2_EPSILON], np.repeat(np.inf, n - 2)]),
                np.repeat(SIGMA_MAX, n),
            ]
        ),
    ),
}


def _normalize_fwd_scale(params: LogNormMixParams) -> LogNormMixParams:
    """Normalize fwd_scale so that sum(w * fwd_scale) == 1 (martingale constraint)."""
    ws_tot = np.dot(params.w, params.fwd_scale)
    fwd_scale_new = params.fwd_scale / ws_tot
    return LogNormMixParams(w=params.w, fwd_scale=fwd_scale_new, sigma=params.sigma)


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Smooth approximation to max(x, 0) with scale parameter beta."""
    return beta * special.softplus(x / beta)


def _softplus_deriv(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Derivative of softplus: sigmoid(x / beta)."""
    return special.expit(x / beta)


def excess_roughness(params: LogNormMixParams, tau: float, sigma_atm: float = 0.2) -> float:
    """Compute the excess roughness of a normal mixture density compared to a Gaussian density."""
    # TODO @Marco: implement analytivcal.
    z_grid = np.linspace(-2, 2, 500)
    dz = z_grid[1] - z_grid[0]
    mu = params.mu(tau)
    d2f_dx2 = gaussian_mixture_density_second_derivative(z_grid, params.w, mu, params.sigma)
    roughness = np.sum(d2f_dx2**2 * dz)
    baseline = 3 / (8 * np.sqrt(np.pi) * sigma_atm**5)
    return roughness - baseline


def piecewise_linspace(knots_val: ArrayLike, n: int) -> np.ndarray:
    kx = np.linspace(-1, 1, len(knots_val))
    ky = np.asarray(knots_val)
    x = np.linspace(-1, 1, n)
    return np.interp(x, kx, ky)


def _smirk_start_guess(n: int, sigma_atm: float) -> LogNormMixParams:
    """Generate initial guess for smirk-like smiles."""
    if n < 2:
        msg = "Number of components must be at least 2."
        raise ValueError(msg)

    # Assign 5% weight to the first component and increasing weights thereafter
    w_left = 0.05
    w_scale = np.linspace(0.6, 1, n - 1)
    w_right = (1 - w_left) * w_scale / w_scale.sum()
    w0 = np.concatenate(([w_left], w_right))

    # Assign increasing fwd_scale
    exp_mu_min = 0.8
    exp_mu_left = np.linspace(exp_mu_min, 1, n - 1)
    partial_sum = np.dot(w0[:-1], exp_mu_left)
    exp_mu_right = (1 - partial_sum) / w0[-1]
    fwd_scale0 = np.concatenate([exp_mu_left, [exp_mu_right]])

    # Assign decreasing sigma
    sigma0 = np.clip(piecewise_linspace([sigma_atm * 2, sigma_atm, sigma_atm * 0.5], n), SIGMA_MIN, SIGMA_MAX)

    return LogNormMixParams(w=w0, fwd_scale=fwd_scale0, sigma=sigma0)


def _uninformative_start_guess(n: int, sigma_atm: float) -> LogNormMixParams:
    """Generate initial guess for flat smiles."""
    if n < 2:
        msg = "Number of components must be at least 2."
        raise ValueError(msg)

    w0 = np.repeat(1 / n, n)
    exp_mu_min = 0.8
    exp_mu_max = 2 - exp_mu_min
    fwd_scale0 = np.linspace(exp_mu_min, exp_mu_max, n)
    fwd_scale0 = fwd_scale0 / np.dot(w0, fwd_scale0)
    sigma0 = np.clip(np.repeat(sigma_atm, n), SIGMA_MIN, SIGMA_MAX)
    return LogNormMixParams(w=w0, fwd_scale=fwd_scale0, sigma=sigma0)


INITIAL_GUESS_METHODS = {
    "uninformative": _uninformative_start_guess,
    "smirk": _smirk_start_guess,
}


def calib_mixture_smile(
    n: int,
    k: np.ndarray,
    tau: float,
    fwd: float,
    df: float,
    mkt_prices: np.ndarray,
    loss_weights: ArrayLike = 1,
    p0: LogNormMixParams | None = None,
    lambda_smoothing: float = 0.0,
    prev_params: LogNormMixParams | None = None,
    transform_method: str = "simplex",
    lambda_w: float = 0.0,
    lambda_mu: float = 0.0,
    lambda_sigma: float = 0.0,
    lambda_ca_bounds: float = 0.0,
    sigma_atm: float = 0.2,
    no_arb_bounds: pd.DataFrame | None = None,
) -> tuple[LogNormMixParams, dict]:
    """Calibrate a log-normal mixture model to option prices with analytical Jacobian.

    Args:
        n: Number of mixture components.
        k: Strike array.
        tau: Time to expiry.
        fwd: Forward price.
        df: Discount factor.
        mkt_prices: Market (mid) option prices.
        loss_weights: Per-observation loss weights.
        p0: Initial parameter guess.
        lambda_smoothing: Roughness penalty weight.
        prev_params: Previous-slice params for regularisation.
        transform_method: Encoder name (one of BIJECTION_METHODS keys).
        lambda_w: Weight regularisation strength.
        lambda_mu: Drift regularisation strength (penalises mu change).
        lambda_sigma: Vol regularisation strength.
        lambda_ca_bounds: Calendar-arbitrage softplus penalty weight.
        sigma_atm: ATMF vol used for roughness baseline.
        no_arb_bounds: DataFrame with calendar-arb upper/lower bounds.

    Returns:
        Tuple of (fitted LogNormMixParams, statistics dict).
    """
    if p0 is None:
        p0 = _uninformative_start_guess(n, sigma_atm=sigma_atm, tau=float(tau))

    if transform_method not in BIJECTION_METHODS:
        msg = f"Unsupported transform method: {transform_method}"
        raise ValueError(msg)

    min_vol = 0.0 if "totvar" in transform_method else SIGMA_MIN

    encoder = BIJECTION_METHODS[transform_method](float(tau))
    x0, unravel, jac_fn = make_ravel_param_jac(p0, encoder, check_unravel=False)

    if transform_method == "reduced":
        bounds_type = "reduced"
    elif prev_params is None:
        bounds_type = "full_mu_bounded"
    else:
        bounds_type = "full_mu_unbounded"

    bounds_factory = BOUNDS_METHODS.get(bounds_type)
    bounds = bounds_factory(n, min_vol)

    if len(bounds[0]) != len(x0):
        msg = f"Bounds length does not match number of parameters {len(x0)}."
        raise ValueError(msg)

    # Clip initial guess to bounds
    x0 = np.clip(x0, bounds[0], bounds[1])

    # Pre-compute constants
    k_arr = np.asarray(k, dtype=float)
    is_call_price = np.ones(len(k_arr), dtype=bool)
    weights_base = np.broadcast_to(np.asarray(loss_weights, dtype=float), mkt_prices.shape)
    n_obs = len(mkt_prices)
    n_flat = len(x0)
    s_w, s_fs, s_sig = _param_slices(n)

    # Arb-bounds pre-computation
    arb_strikes = arb_ub = arb_lb = arb_weights = False
    if no_arb_bounds is not None:
        arb_strikes = no_arb_bounds["strike"].to_numpy()
        arb_ub = no_arb_bounds["price_norm_ub"].to_numpy()
        arb_weights = no_arb_bounds["weight"].to_numpy()
        if "price_norm_lb" in no_arb_bounds.columns:
            arb_lb = no_arb_bounds["price_norm_lb"].to_numpy()

    # Previous mu for regularisation
    prev_mu = prev_params.mu(tau) if prev_params is not None else None

    def _loss_function(x: np.ndarray) -> np.ndarray:
        param = unravel(x)
        model_price = _mixed_log_norm_opt_price(
            w=param.w,
            fwd_scale=param.fwd_scale,
            sigma=param.sigma,
            disc=df,
            fwd=fwd,
            strike=k_arr,
            tau=tau,
            is_call=is_call_price,
        )

        residuals = model_price - mkt_prices
        weights = weights_base.copy()

        if lambda_smoothing > 0.0:
            penalty = np.sqrt(softplus(excess_roughness(param, sigma_atm=sigma_atm, tau=tau), beta=0.1))
            residuals = np.concatenate([residuals, np.array([penalty])])
            weights = np.concatenate([weights, np.array([lambda_smoothing])])

        if lambda_w > 0.0 and prev_params is not None:
            delta_w = param.w - prev_params.w
            residuals = np.concatenate([residuals, delta_w])
            weights = np.concatenate([weights, np.repeat(lambda_w, n)])

        if lambda_mu > 0.0 and prev_params is not None:
            mu_current = param.mu(tau)
            delta_mu = mu_current - prev_mu
            residuals = np.concatenate([residuals, delta_mu])
            weights = np.concatenate([weights, np.repeat(lambda_mu, n)])

        if lambda_sigma > 0.0 and prev_params is not None:
            delta_sigma = param.sigma - prev_params.sigma
            residuals = np.concatenate([residuals, delta_sigma])
            weights = np.concatenate([weights, np.repeat(lambda_sigma, n)])

        if arb_strikes is not None:
            prices_norm = _mixed_log_norm_opt_price(
                w=param.w,
                fwd_scale=param.fwd_scale,
                sigma=param.sigma,
                disc=df,
                fwd=fwd,
                strike=arb_strikes,
                tau=tau,
                is_call=np.ones(len(arb_strikes), dtype=bool),
            ) / (fwd * df)

            arb_upper = softplus(arb_weights * (prices_norm - arb_ub), beta=1e-8)
            residuals = np.concatenate([residuals, arb_upper])
            weights = np.concatenate([weights, np.repeat(np.sqrt(lambda_ca_bounds), len(arb_upper))])

            if arb_lb is not None:
                arb_lower = softplus(arb_weights * (arb_lb - prices_norm), beta=1e-8)
                residuals = np.concatenate([residuals, arb_lower])
                weights = np.concatenate([weights, np.repeat(np.sqrt(lambda_ca_bounds), len(arb_lower))])

        return weights * residuals

    def _jacobian(x: np.ndarray) -> np.ndarray:
        """Analytical Jacobian of the weighted residual vector."""
        param = unravel(x)
        J_enc = jac_fn(x)  # (3n, n_flat)

        # --- Price residual block ---
        J_price_params = _mixed_log_norm_opt_jac(
            w=param.w,
            fwd_scale=param.fwd_scale,
            sigma=param.sigma,
            disc=df,
            fwd=fwd,
            strike=k_arr,
            tau=tau,
            is_call=is_call_price,
        )  # (n_obs, 3n)
        J_price = (weights_base[:, np.newaxis] * J_price_params) @ J_enc  # (n_obs, n_flat)

        jac_rows = [J_price]

        # --- Roughness penalty block (numerical Jacobian) ---
        if lambda_smoothing > 0.0:
            roughness_val = excess_roughness(param, sigma_atm=sigma_atm, tau=tau)
            sp_val = softplus(roughness_val, beta=0.1)
            penalty = np.sqrt(sp_val)

            # Numerical Jacobian via centered differences on x
            eps = 1e-7
            jac_rough = np.zeros((1, n_flat), dtype=float)
            for j in range(n_flat):
                x_p = x.copy()
                x_m = x.copy()
                x_p[j] += eps
                x_m[j] -= eps
                p_p = unravel(x_p)
                p_m = unravel(x_m)
                r_p = excess_roughness(p_p, sigma_atm=sigma_atm, tau=tau)
                r_m = excess_roughness(p_m, sigma_atm=sigma_atm, tau=tau)
                sp_p = np.sqrt(softplus(r_p, beta=0.1))
                sp_m = np.sqrt(softplus(r_m, beta=0.1))
                jac_rough[0, j] = (sp_p - sp_m) / (2 * eps)
            jac_rows.append(lambda_smoothing * jac_rough)

        # --- Weight regularisation block ---
        if lambda_w > 0.0 and prev_params is not None:
            # d(w - w_prev)/d_params = [I_n | 0 | 0]
            J_w_params = np.zeros((n, 3 * n), dtype=float)
            J_w_params[:, s_w] = np.eye(n)
            jac_rows.append(lambda_w * (J_w_params @ J_enc))

        # --- Mu regularisation block ---
        if lambda_mu > 0.0 and prev_params is not None:
            # mu = log(fwd_scale) / tau => dmu_i/dfwd_scale_i = 1/(tau * fwd_scale_i)
            J_mu_params = np.zeros((n, 3 * n), dtype=float)
            J_mu_params[:, s_fs] = np.diag(1.0 / (tau * param.fwd_scale))
            jac_rows.append(lambda_mu * (J_mu_params @ J_enc))

        # --- Sigma regularisation block ---
        if lambda_sigma > 0.0 and prev_params is not None:
            J_sig_params = np.zeros((n, 3 * n), dtype=float)
            J_sig_params[:, s_sig] = np.eye(n)
            jac_rows.append(lambda_sigma * (J_sig_params @ J_enc))

        # --- No-arb bounds blocks ---
        if arb_strikes is not None:
            n_arb = len(arb_strikes)
            is_call_arb = np.ones(n_arb, dtype=bool)

            J_arb_params = _mixed_log_norm_opt_jac(
                w=param.w,
                fwd_scale=param.fwd_scale,
                sigma=param.sigma,
                disc=df,
                fwd=fwd,
                strike=arb_strikes,
                tau=tau,
                is_call=is_call_arb,
            ) / (fwd * df)  # (n_arb, 3n)

            prices_norm = _mixed_log_norm_opt_price(
                w=param.w,
                fwd_scale=param.fwd_scale,
                sigma=param.sigma,
                disc=df,
                fwd=fwd,
                strike=arb_strikes,
                tau=tau,
                is_call=is_call_arb,
            ) / (fwd * df)

            # Upper bound: softplus(arb_weight * (price_norm - ub))
            g_ub = arb_weights * (prices_norm - arb_ub)
            sig_ub = _softplus_deriv(g_ub, beta=1e-8)  # sigmoid
            # d(softplus(g))/dx = sigmoid(g/beta) * dg/dx
            # dg/dx = arb_weight * d(price_norm)/dx
            J_arb_ub = (np.sqrt(lambda_ca_bounds) * sig_ub * arb_weights)[:, np.newaxis] * J_arb_params
            jac_rows.append(J_arb_ub @ J_enc)

            if arb_lb is not None:
                # Lower bound: softplus(arb_weight * (lb - price_norm))
                g_lb = arb_weights * (arb_lb - prices_norm)
                sig_lb = _softplus_deriv(g_lb, beta=1e-8)
                # dg/dx = -arb_weight * d(price_norm)/dx
                J_arb_lb = (np.sqrt(lambda_ca_bounds) * sig_lb * (-arb_weights))[:, np.newaxis] * J_arb_params
                jac_rows.append(J_arb_lb @ J_enc)

        return np.vstack(jac_rows)

    res = least_squares(
        fun=_loss_function,
        x0=x0,
        jac=_jacobian,
        method="trf",
        bounds=bounds,
        x_scale="jac",
    )

    if not res.success:
        msg = f"Mixture calibration did not converge for tau={float(tau):.2f}): {res.message}"
        log.warning(msg)

    stats = {
        "error": res.fun[:n_obs],
        "mse": float(np.mean(res.fun[:n_obs] ** 2)),
        "success": res.success,
        "message": res.message,
        "cost": res.cost,
    }

    return unravel(res.x), stats


def _make_smile_fun(params: LogNormMixParams, le: LinearEquityMarket, tau: float) -> VolSmile:
    """Construct a VolSmile object from calibrated log-normal mixture parameters."""
    tau = float(tau)
    disc = le.df(tau)
    fwd = le.fwd(tau)

    sigma_max = np.max(params.sigma)
    k_ub = fwd * np.exp(6 * sigma_max * np.sqrt(tau) + 0.5 * sigma_max**2 * tau)
    k_lb = fwd * np.exp(-6 * sigma_max * np.sqrt(tau) + 0.5 * sigma_max**2 * tau)

    # TODO @Marco: add extrapolator.
    def extrapl_fun():
        pass

    def fun(k: np.ndarray | float) -> np.ndarray | float:
        k_is_scalar = np.isscalar(k)
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))

        # Use OTM contracts for increased stability.
        is_call = k_arr >= fwd
        prices = _mixed_log_norm_opt_price(
            w=params.w,
            fwd_scale=params.fwd_scale,
            sigma=params.sigma,
            disc=disc,
            fwd=fwd,
            strike=k_arr,
            tau=tau,
            is_call=is_call,
        )

        iv = implied_black_vol(
            price=prices,
            fwd=fwd,
            strike=k_arr,
            tau=tau,
            disc=disc,
            is_call=is_call,
        )

        return float(iv[0]) if k_is_scalar else iv

    return VolSmile(interpl=fun)


def _vega_weights(opt: OptionChainLike, line_mkt: LinearEquityMarket) -> np.ndarray:
    """Compute inverse-vega weights for loss weighting."""
    fwd = line_mkt.fwd(opt.tau)
    disc = line_mkt.df(opt.tau)

    k, tau, mid = opt.k, opt.tau, opt.mid
    is_call = opt.option_type == "C"

    iv = implied_black_vol(
        price=mid,
        fwd=fwd,
        strike=k,
        tau=tau,
        disc=disc,
        is_call=is_call,
    ).clip(0.02, 1.5)

    vega = black76_vega(fwd=fwd, strike=k, tau=tau, sigma=iv, disc=disc)
    return 1 / np.maximum(vega, 1e-6)


def calib_mixture_ivs(
    opt: OptionChainLike,
    mkt: LinearEquityMarket,
    n_components: int,
    lw_type: str | None = None,
    x0: LogNormMixParams | None = None,
    transform_method: str = "base",
    t0_start_guess: str = "uninformative",
    lambda_smoothing: float = 0.0,
    lambda_tm1_params: tuple[float, float, float] = (0.0, 0.0, 0.0),
    calendar_arb_bounds: NoArbBounds | None = None,
    lambda_ca_bounds: float = 0.0,
) -> tuple[VolSurface, dict, dict]:
    """Calibrate a log-normal mixture model to each expiry slice.

    Args:
        opt: Option chain iterable yielding (expiry_key, slice) pairs.
        mkt: Linear equity market model (forwards, discount factors).
        n_components: Number of mixture components.
        lw_type: Loss-weight type ('uniform', 'vega', 'vega_and_spread').
        x0: Starting parameters for the first slice (optional).
        transform_method: Encoder name.
        t0_start_guess: Initial-guess method for the first slice.
        lambda_smoothing: Roughness penalty weight.
        lambda_tm1_params: (lambda_w, lambda_mu, lambda_sigma) regularisation.
        calendar_arb_bounds: Calendar-arbitrage bounds object.
        lambda_ca_bounds: Calendar-arb penalty weight.

    Returns:
        Tuple of (VolSurface, params dict, stats dict).
    """
    _require_call_only(opt)

    if transform_method not in BIJECTION_METHODS:
        msg = f"Unsupported transform method: {transform_method}"
        raise ValueError(msg)

    taus = []
    smiles = []
    params = {}

    stats = {"_contracts": []}

    prev_params = x0

    prev_tau = None
    for t, opt_slice in opt:
        sigma_atm = get_atmf_vol(opt_slice, mkt)
        div_dp = _vega_weights(opt_slice, mkt)

        k_sl = opt_slice.k
        mid_sl = opt_slice.mid
        tau = opt_slice.tau[0]

        # Obtain scalar discount factor and forward for this maturity.
        tau_vec = np.array([tau], dtype=float)
        disc = float(mkt.df(tau_vec)[0])
        fwd = float(mkt.fwd(tau_vec)[0])

        if len(np.unique(opt_slice.k)) != len(opt_slice.k):
            msg = f"Duplicate strikes found in option slice for tau={tau}."
            raise ValueError(msg)

        if lw_type is None or lw_type == "uniform":
            loss_weights = np.ones_like(k_sl, dtype=float)
        elif lw_type == "vega":
            loss_weights = div_dp
        elif lw_type == "vega_and_spread":
            vega_weights = div_dp
            iv_spread_inv = 1 / ((opt_slice.ask - opt_slice.bid) * div_dp)
            spread_weights = (0.9) * (iv_spread_inv - np.min(iv_spread_inv)) / (
                np.max(iv_spread_inv) - np.min(iv_spread_inv)
            ) + 0.1
            loss_weights = vega_weights * np.clip(spread_weights, 0.1, 1.0)
        else:
            msg = f"Unsupported weights type: {lw_type}"
            raise ValueError(msg)

        if prev_params is None:
            make_initial_guess = INITIAL_GUESS_METHODS[t0_start_guess]
            p0 = make_initial_guess(n_components, sigma_atm=sigma_atm, tau=tau)
            lambda_w = lambda_mu = lambda_sigma = 0.0

            transform_method_ = BIJECTION_1ST_SLICE_FALLBACK.get(transform_method, transform_method)
        else:
            transform_method_ = transform_method
            p0 = _normalize_fwd_scale(prev_params)
            lambda_w, lambda_mu, lambda_sigma = lambda_tm1_params
            if "totvar" in transform_method_ and prev_tau is not None:
                # Adjust sigma to keep total variance constant
                scaled_sigma = prev_params.sigma * np.sqrt(prev_tau / tau)
                p0 = LogNormMixParams(w=p0.w, fwd_scale=p0.fwd_scale, sigma=scaled_sigma)

        bounds_df = None
        if calendar_arb_bounds is not None:
            bounds_df = calendar_arb_bounds[t].call_ub
            if prev_tau is not None:
                k_tm1 = bounds_df["strike"] / fwd * mkt.fwd(prev_tau)
                norm_denom = mkt.df(prev_tau) * mkt.fwd(prev_tau)
                bounds_df["price_norm_lb"] = (
                    black76_price(
                        fwd=mkt.fwd(prev_tau),
                        strike=k_tm1,
                        tau=prev_tau,
                        sigma=smiles[-1].vol(k_tm1),
                        disc=mkt.df(prev_tau),
                        is_call=True,
                    )
                    / norm_denom
                )

        fitted, stats_t = calib_mixture_smile(
            n=n_components,
            k=k_sl,
            tau=tau,
            fwd=fwd,
            df=disc,
            mkt_prices=mid_sl,
            loss_weights=loss_weights,
            p0=p0,
            prev_params=prev_params,
            lambda_w=lambda_w,
            lambda_mu=lambda_mu,
            lambda_sigma=lambda_sigma,
            transform_method=transform_method_,
            lambda_ca_bounds=lambda_ca_bounds,
            lambda_smoothing=lambda_smoothing,
            sigma_atm=sigma_atm,
            no_arb_bounds=bounds_df,
        )

        # Calculate summary statistics
        model_price = _mixed_log_norm_opt_price(
            w=fitted.w,
            fwd_scale=fitted.fwd_scale,
            sigma=fitted.sigma,
            disc=disc,
            fwd=fwd,
            strike=np.asarray(k_sl, dtype=float),
            tau=tau,
            is_call=True,
        )

        stats["_contracts"].append(
            {
                "tau": np.repeat(tau, len(k_sl)),
                "strike": np.asarray(k_sl, dtype=float),
                "mid": np.asarray(mid_sl, dtype=float),
                "bid": np.asarray(opt_slice.bid, dtype=float),
                "ask": np.asarray(opt_slice.ask, dtype=float),
                "model_price": np.asarray(model_price, dtype=float),
                "div_dp": np.asarray(div_dp, dtype=float),
            }
        )

        taus.append(float(tau))
        smiles.append(_make_smile_fun(fitted, mkt, tau))
        stats[t] = stats_t
        params[t] = {
            "tau": tau,
            "params": fitted,
        }

        # Update results
        prev_params = fitted
        prev_tau = tau

    # Surface level statistics and sanity checks
    # TODO @Marco: move to separate function.
    stats_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "mid": c["mid"],
                    "bid": c["bid"],
                    "ask": c["ask"],
                    "model_price": c["model_price"],
                    "div_dp": c["div_dp"],
                },
                index=pd.MultiIndex.from_arrays([c["tau"], c["strike"]], names=["tau", "strike"]),
            )
            for c in stats["_contracts"]
        ]
    )

    bid_ask_width = stats_df["ask"] - stats_df["bid"]
    iv_error_approx = (stats_df["mid"] - stats_df["model_price"]) * stats_df["div_dp"]
    outside_lower = (stats_df["bid"] - stats_df["model_price"]).clip(lower=0.0)
    outside_upper = (stats_df["model_price"] - stats_df["ask"]).clip(lower=0.0)
    outside_bid_ask_price_error = outside_lower + outside_upper

    iv_error_outside_bid_ask = (outside_bid_ask_price_error * stats_df["div_dp"]).abs()
    in_bid_ask = (stats_df["bid"] <= stats_df["model_price"]) & (stats_df["model_price"] <= stats_df["ask"])
    in_2x_bid_ask = (stats_df["mid"] - 2 * bid_ask_width <= stats_df["model_price"]) & (
        stats_df["model_price"] <= stats_df["mid"] + 2 * bid_ask_width
    )
    max_error_key = iv_error_outside_bid_ask.idxmax()
    tau_at_max_error = float(max_error_key[0])
    strike_at_max_error = float(max_error_key[1])

    surface_stats = {
        "iv_mae_approx": float(np.mean(np.abs(iv_error_approx))),
        "num_out_of_bid_ask": int((~in_bid_ask).sum()),
        "num_out_of_2x_bid_ask": int((~in_2x_bid_ask).sum()),
        "iv_mae_outside_bid_ask": float(np.mean(iv_error_outside_bid_ask)),
        "iv_maxae_outside_bid_ask": float(iv_error_outside_bid_ask.loc[max_error_key]),
    }

    stats |= surface_stats

    if stats["iv_mae_approx"] > 5e-3:
        msg = f"High surface MAE: {stats['iv_mae_approx']:.4f}."
        log.warning(msg)

    if stats["iv_maxae_outside_bid_ask"] > 3e-2:
        msg = (
            f"High bid-ask IV breach: {stats['iv_maxae_outside_bid_ask']:.4f} "
            f"at (tau={tau_at_max_error:.2f}, "
            f"strike={strike_at_max_error:.0f})."
        )
        log.warning(msg)

    if stats["num_out_of_2x_bid_ask"] > 0:
        msg = f"{stats['num_out_of_2x_bid_ask']} model price(s) is outside 2x bid-ask spread(s)."
        log.info(msg)

    return VolSurface(np.array(taus, dtype=float), smiles, mkt), params, stats


def gaussian_pdf(x: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Compute the Gaussian PDF for a mixture component."""
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_mixture_density(x: ArrayLike, mix_weights: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Compute risk-neutral density for a Gaussian mixture at moneyness points.

    Args:
        x: Points where to evaluate the density (array).
        mix_weights: Mixture weights (array).
        mu: Mixture means (array).
        sigma: Mixture volatilities (array).

    Returns:
        Array of densities at each x.
    """
    x_ = np.asarray(x, dtype=float)[:, np.newaxis]
    w_ = np.asarray(mix_weights, dtype=float)[np.newaxis, :]
    mu_ = np.asarray(mu, dtype=float)[np.newaxis, :]
    sigma_ = np.asarray(sigma, dtype=float)[np.newaxis, :]
    pdf = (1.0 / (sigma_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ - mu_) / sigma_) ** 2)
    return (w_ * pdf).sum(axis=1)


def gaussian_mixture_density_second_derivative(
    x: ArrayLike, mix_weights: ArrayLike, mu: ArrayLike, sigma: ArrayLike
) -> np.ndarray:
    """Compute second derivative of Gaussian mixture density analytically."""
    x_ = np.asarray(x, dtype=float)[:, np.newaxis]
    w_ = np.asarray(mix_weights, dtype=float)[np.newaxis, :]
    mu_ = np.asarray(mu, dtype=float)[np.newaxis, :]
    s_ = np.asarray(sigma, dtype=float)[np.newaxis, :]
    pdf = (1.0 / (s_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ - mu_) / s_) ** 2)
    return (w_ * pdf * ((x_ - mu_) ** 2 - s_**2) / s_**4).sum(axis=1)
