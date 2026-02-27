import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import special
from scipy.optimize import least_squares

from vol_risk.calibration.data.transformers import get_atmf_vol
from vol_risk.models.black76 import black76_price, black76_vega, implied_vol_jackel
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.protocols import EuropeanOption, ModelParams, OptionChainLike
from vol_risk.util import angles_to_simplex, make_ravel_param, simplex_to_angles
from vol_risk.vol_surface.surface import VolSmile, VolSurface

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LogNormMixParams(ModelParams):
    """Parameters for the log-normal mixture model.

    Attributes:
        w: The weights of the mixture components.
        mu: The means of the mixture components.
        sigma: The volatilities of the mixture components.
    """

    w: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self):
        """Validates parameters."""
        if not (len(self.w) == len(self.mu) == len(self.sigma)):
            msg = "Parameters 'w', 'mu', and 'sigma' must have the same length."
            raise ValueError(msg)

        if not np.all(self.w >= 0):
            msg = "All weights 'w' must be non-negative."
            raise ValueError(msg)

        if not np.isclose(np.sum(self.w), 1.0):
            msg = "The sum of weights 'w' must be equal to 1."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LogNormMixCalibParams:
    """Parameters for the log-normal mixture calibration model."""

    bijection_factory: Callable
    lambda_rough: float
    lambda_w: float
    lambda_mu: float
    lambda_sigma: float


def _mixed_log_norm_call(
    w: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    DF: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    pdef: float = 0,
) -> np.ndarray:
    """Low-level function returning call option price under a log-normal mixture model."""
    w = np.asarray(w)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if not (w.shape == mu.shape == sigma.shape):
        msg = "w, mu, sigma must have identical 1-D shapes"
        raise ValueError(msg)
    if not np.isclose(w.sum(), 1.0):
        msg = "mixture weights must sum to 1"
        raise ValueError(msg)

    return (1 - pdef) * np.sum(
        w[i] * black76_price(df=DF, f=F * np.exp(mu[i] * tau) / (1 - pdef), k=K, t=tau, sigma=sigma[i], is_call=True)
        for i in range(len(w))
    )


def mixed_log_norm_call(
    x: LogNormMixParams,
    mkt: LinearEquityMarket,
    opt: EuropeanOption,
) -> np.array:
    """Returns the call option price under a log-normal mixture model."""
    k, tau = opt.strike, opt.tau
    fwd = mkt.fwd(tau)
    disc = mkt.df(tau)

    return _mixed_log_norm_call(
        w=x.w,
        mu=x.mu,
        sigma=x.sigma,
        DF=disc,
        F=fwd,
        K=k,
        tau=tau,
    )


def make_full_encoder(tau: float, method: str = "simplex") -> tuple:
    """Creates a bijection for log-normal mixture calibration parameters."""

    def encode(params: LogNormMixParams) -> tuple:
        w, mu, sigma = params.w, params.mu, params.sigma
        z = w * np.exp(mu * tau)

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
            x1 = mu[: len(z) - 1]
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        free = (x0, x1, sigma)
        return (free, ())

    def decode(free: tuple[ArrayLike], _: tuple[ArrayLike] | None) -> LogNormMixParams:
        x0, x1, sigma = free

        w = angles_to_simplex(x0)
        z = angles_to_simplex(x1)

        if method == "simplex":
            mu = np.log(z / w) / tau
        elif method == "manual":
            partial_sum = np.dot(w[:-1], np.exp(x1 * tau))
            if (1 - partial_sum) <= 0:
                msg = "Invalid parameters: remaining forward mass <= 0. Use simplex method instead."
                raise ValueError(msg)
            mu_n = np.log((1 - partial_sum) / w[-1]) / tau
            mu = np.concatenate([x1, np.array(mu_n)])
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        return LogNormMixParams(w=w, mu=mu, sigma=sigma)

    return (encode, decode)


def make_full_encoder_totvar(tau: float, method: str = "simplex") -> tuple:
    """Creates a bijection for log-normal mixture calibration parameters with additive total variance."""

    def encode(params: LogNormMixParams) -> tuple:
        w, mu, sigma = params.w, params.mu, params.sigma
        z = w * np.exp(mu * tau)

        if not (np.isclose(np.sum(w), 1.0) and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (np.isclose(np.sum(z), 1.0) and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        dv = np.zeros_like(sigma, np.float64)
        x0 = simplex_to_angles(w)

        if method == "simplex":
            x1 = simplex_to_angles(z)
        elif method == "manual":
            x1 = mu[: len(z) - 1]
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
        z = angles_to_simplex(x1)

        if method == "simplex":
            mu = np.log(z / w) / tau
        elif method == "manual":
            partial_sum = np.dot(w[:-1], np.exp(x1 * tau))
            if (1 - partial_sum) <= 0:
                msg = "Invalid parameters: remaining forward mass <= 0. Use simplex method instead."
                raise ValueError(msg)
            mu_n = np.log((1 - partial_sum) / w[-1]) / tau
            mu = np.concatenate([x1, np.array(mu_n)])
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        return LogNormMixParams(w=w, mu=mu, sigma=sigma)

    return (encode, decode)


def make_reduced_encoder(tau: float) -> tuple:
    """Creates a bijection for log-normal mixture calibration with mu and w parameters fixed."""

    def encode(params: LogNormMixParams) -> tuple:
        w, mu, sigma = params.w, params.mu, params.sigma
        z = w * np.exp(mu * tau)

        if not (np.isclose(np.sum(w), 1.0) and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (np.isclose(np.sum(z), 1.0) and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        free = sigma
        fixed = (w, mu)
        return (free, fixed)

    def decode(free: tuple[ArrayLike], fixed: tuple[ArrayLike]) -> LogNormMixParams:
        w, mu = fixed
        sigma = np.squeeze(free)
        return LogNormMixParams(w=w, mu=mu, sigma=sigma)

    return (encode, decode)


BIJECTION_METHODS = {
    "reduced": make_reduced_encoder,
    "base": lambda x: make_full_encoder(x, method="manual"),
    "simplex": lambda x: make_full_encoder(x, method="simplex"),
    "totvar": lambda x: make_full_encoder_totvar(x, method="manual"),
    "totvar_simplex": lambda x: make_full_encoder_totvar(x, method="simplex"),
}

BOUNDS_METHODS = {
    "reduced": lambda n, sigma_min: (np.repeat(sigma_min, n), np.repeat(np.inf, n)),
    "full": lambda n, sigma_min: (
        np.concatenate([np.repeat(-np.inf, n - 1), np.repeat(-np.inf, n - 1), np.repeat(sigma_min, n)]),
        np.concatenate([np.repeat(np.inf, n - 1), np.repeat(np.inf, n - 1), np.repeat(np.inf, n)]),
    ),
}


def _force_mu_to_unit_sum(params: LogNormMixParams, tau: float) -> LogNormMixParams:
    """Adjusts the mu parameters so that the mixture has unit expectation."""
    s = np.sum(params.w * np.exp(params.mu * tau))
    mu_new = params.mu - np.log(s) / tau
    return LogNormMixParams(w=params.w, mu=mu_new, sigma=params.sigma)


# def _mixed_log_norm_calib(n, k, t, f, df, mkt_prices, loss_scale=1):
#     """Calibrate a log-normal mixture model to option prices."""
#     # Initial guess
#     w0 = np.repeat(1 / n, n)
#     mu0 = np.zeros(n)
#     mu0[0] = -0.1
#     mu0[-1] = np.log((1 - sum(w0[:-1] * np.exp(mu0[:-1] * t))) / w0[-1]) / t
#     sigma0 = np.repeat(0.2, n)
#     p0 = LogNormMixParams(w0, mu0, sigma0)
#     x0, unravel = make_ravel_param(p0, make_reduced_encoder(tau=t), check_unravel=True)

#     # bounds
#     bounds = (np.repeat(0.03, n), np.repeat(np.inf, n))

#     def _loss_function(x, tau, disc, fwd, k, mkt_opt_p) -> np.ndarray:
#         param = unravel(x)
#         model_price = _mixed_log_norm_call(
#             w=param.w,
#             mu=param.mu,
#             sigma=param.sigma,
#             DF=disc,
#             F=fwd,
#             K=k,
#             tau=tau,
#         )
#         return model_price - mkt_opt_p

#     res = least_squares(
#         fun=lambda x: loss_scale * (_loss_function(x, t, df, f, k, mkt_prices)),
#         x0=x0,
#         jac="2-point",
#         method="trf",
#         bounds=bounds,
#     )

#     return unravel(res.x)


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Smooth approximation to max(x, 0) with scale parameter beta."""
    return beta * special.softplus(x / beta)


def excess_roughness(params: LogNormMixParams, sigma_atm: float = 0.2) -> float:
    """Compute the excess roughness of a normal mixture density compared to a Gaussian density."""
    z_grid = np.linspace(-2, 2, 500)
    dz = z_grid[1] - z_grid[0]
    d2f_dx2 = gaussian_mixture_density_second_derivative(z_grid, params.w, params.mu, params.sigma)
    roughness = sum(d2f_dx2**2 * dz)
    baseline = 3 / (8 * np.sqrt(np.pi) * sigma_atm**5)
    return roughness - baseline


def piecewise_linspace(knots_val: ArrayLike, n: int) -> np.ndarray:
    kx = np.linspace(-1, 1, len(knots_val))
    ky = np.asarray(knots_val)
    x = np.linspace(-1, 1, n)
    return np.interp(x, kx, ky)


def _smirk_start_guess(n: int, sigma_atm: float, tau: float) -> LogNormMixParams:
    """Generate initial guess for smirk-like smiles."""
    if n < 2:
        msg = "Number of components must be at least 2."
        raise ValueError(msg)

    w0 = np.repeat(1 / n, n)
    mu_min = np.exp(-0.30 * tau)
    mu_max = 2 - mu_min
    mu0 = np.log(np.linspace(mu_min, mu_max, n)) / tau
    sigma0 = piecewise_linspace([sigma_atm * 2, sigma_atm, sigma_atm * 1.5], n)
    return LogNormMixParams(w0, mu0, sigma0)


def _uninformative_start_guess(n: int, sigma_atm: float, tau: float) -> LogNormMixParams:
    """Generate initial guess for flat smiles."""
    if n < 2:
        msg = "Number of components must be at least 2."
        raise ValueError(msg)

    w0 = np.repeat(1 / n, n)
    mu_min = np.exp(-0.30 * tau)
    mu_max = 2 - mu_min
    mu0 = np.log(np.linspace(mu_min, mu_max, n)) / tau
    sigma0 = np.repeat(sigma_atm, n)
    return LogNormMixParams(w0, mu0, sigma0)


def calib_mixture_smile(
    n: int,
    k: np.ndarray,
    tau: float,
    fwd: float,
    df: float,
    mkt_prices: np.ndarray,
    loss_weights: ArrayLike = 1,
    p0: LogNormMixParams | None = None,
    lambda_rough: float = 0.0,
    prev_params: LogNormMixParams | None = None,
    transform_method: str = "base",
    lambda_w: float = 0.0,
    lambda_mu: float = 0.0,
    lambda_sigma: float = 0.0,
    pdef: float = 0.0,
) -> np.ndarray:
    """Calibrate a log-normal mixture model to option prices."""
    if p0 is None:
        p0 = _uninformative_start_guess(n, sigma_atm=0.2, tau=float(tau))

    if transform_method not in BIJECTION_METHODS:
        msg = f"Unsupported transform method: {transform_method}"
        raise ValueError(msg)

    min_vol = 0.0 if "totvar" in transform_method else 0.05

    encoder = BIJECTION_METHODS[transform_method](tau)
    x0, unravel = make_ravel_param(p0, encoder, check_unravel=False)

    bounds_type = "reduced" if transform_method == "reduced" else "full"
    bounds_factory = BOUNDS_METHODS.get(bounds_type)
    bounds = bounds_factory(n, min_vol)

    if len(bounds[0]) != len(x0):
        msg = f"Bounds length does not match number of parameters {len(x0)}."
        raise ValueError(msg)

    def _loss_function(x: ArrayLike) -> np.ndarray:
        param = unravel(x)
        model_price = _mixed_log_norm_call(
            w=param.w,
            mu=param.mu,
            sigma=param.sigma,
            DF=df,
            F=fwd,
            K=k,
            tau=tau,
            pdef=pdef,
        )

        residuals = model_price - mkt_prices

        weights = np.broadcast_to(loss_weights, mkt_prices.shape)

        if lambda_rough > 0.0:
            penalty = np.sqrt(softplus(excess_roughness(param), beta=0.1))
            residuals = np.concatenate([residuals, np.array([penalty])])
            weights = np.concatenate([weights, np.array([lambda_rough])])

        if lambda_w > 0.0 and prev_params is not None:
            delta_w = param.w - prev_params.w
            residuals = np.concatenate([residuals, delta_w])
            weights = np.concatenate([weights, np.repeat(lambda_w, delta_w.size)])

        if lambda_mu > 0.0 and prev_params is not None:
            delta_mu = param.mu - prev_params.mu
            residuals = np.concatenate([residuals, delta_mu])
            weights = np.concatenate([weights, np.repeat(lambda_mu, delta_mu.size)])

        if lambda_sigma > 0.0 and prev_params is not None:
            delta_sigma = param.sigma - prev_params.sigma
            residuals = np.concatenate([residuals, delta_sigma])
            weights = np.concatenate([weights, np.repeat(lambda_sigma, delta_sigma.size)])

        return np.sqrt(weights) * residuals

    res = least_squares(
        fun=lambda x: (_loss_function(x)),
        x0=x0,
        jac="3-point",
        method="trf",
        bounds=bounds,
    )

    if not res.success:
        msg = f"Log-normal mixture calibration did not converge: {res.message}"
        log.warning(msg)

    return unravel(res.x)


def _make_smile_fun(params: LogNormMixParams, market: LinearEquityMarket, tau: float, pdef: float = 0.0) -> VolSmile:
    tau_val = float(tau)
    tau_vec = np.array([tau_val], dtype=float)
    df = float(market.df(tau_vec)[0])
    fwd = float(market.fwd(tau_vec)[0])

    def fun(k: np.ndarray | float) -> np.ndarray | float:
        k_is_scalar = np.isscalar(k)
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))

        prices = _mixed_log_norm_call(
            w=params.w,
            mu=params.mu,
            sigma=params.sigma,
            DF=df,
            F=fwd,
            K=k_arr,
            tau=tau_val,
            pdef=pdef,
        )

        iv = np.empty_like(k_arr, dtype=float)
        for i, (ki, pi) in enumerate(zip(k_arr, prices, strict=True)):
            iv[i] = implied_vol_jackel(
                price=float(pi),
                f=fwd,
                k=float(ki),
                t=tau_val,
                df=df,
                is_call=True,
            )

        return float(iv[0]) if k_is_scalar else iv

    return VolSmile(interpl=fun)


def calib_mixture_ivs(
    opt: OptionChainLike,
    mkt: LinearEquityMarket,
    n_components: int,
    lw_type: str | None = None,
    pdef: float = 0.0,
    x0: LogNormMixParams | None = None,
    transform_method: str = "base",
) -> tuple[VolSurface, dict]:
    """Calibrate a log-normal mixture model to each expiry slice."""
    if transform_method not in BIJECTION_METHODS:
        msg = f"Unsupported transform method: {transform_method}"
        raise ValueError(msg)

    taus: list[float] = []
    smiles: list[VolSmile] = []
    stats: dict = {}

    prev_params = x0

    for t, opt_slice in opt:
        sigma_atm = get_atmf_vol(opt_slice, mkt)

        k_sl = opt_slice.k
        mid_sl = opt_slice.mid
        tau_val = opt_slice.tau[0]

        # Obtain scalar discount factor and forward for this maturity.
        tau_vec = np.array([tau_val], dtype=float)
        df = float(mkt.df(tau_vec)[0])
        fwd = float(mkt.fwd(tau_vec)[0])

        if len(np.unique(opt_slice.k)) != len(opt_slice.k):
            raise ValueError("Duplicate strikes present in option slice")

        if lw_type is None or lw_type == "uniform":
            loss_weights = np.ones_like(k_sl, dtype=float)
        elif lw_type == "vega":
            iv = np.array(
                [
                    implied_vol_jackel(price=market_price, f=fwd, k=k, t=tau_val, df=df, is_call=opt_type == "C")
                    for market_price, k, opt_type in zip(opt_slice.mid, opt_slice.k, opt_slice.option_type, strict=True)
                ]
            ).clip(0.01, 1)
            loss_weights = 1 / np.maximum(black76_vega(df=df, f=fwd, k=k_sl, t=tau_val, sigma=iv), 1e-4)
        else:
            msg = f"Unsupported weights type: {lw_type}"
            raise ValueError(msg)

        if prev_params is None:
            p0 = _smirk_start_guess(n_components, sigma_atm=sigma_atm, tau=tau_val)
            # p0 = _uninformative_start_guess(n_components, sigma_atm=sigma_atm, tau=tau_val)
            lambda_w = 0.0
            lambda_mu = 0.0
        else:
            p0 = _force_mu_to_unit_sum(prev_params, tau_val)
            lambda_w = 0.1
            lambda_mu = 0.1

        fitted = calib_mixture_smile(
            n=n_components,
            k=k_sl,
            tau=tau_val,
            fwd=fwd,
            df=df,
            mkt_prices=mid_sl,
            loss_weights=loss_weights,
            p0=p0,
            pdef=pdef,
            prev_params=prev_params,
            lambda_w=lambda_w,
            lambda_mu=lambda_mu,
            transform_method=transform_method,
        )

        prev_params = fitted
        taus.append(float(tau_val))
        smiles.append(_make_smile_fun(fitted, mkt, tau_val, pdef=pdef))
        stats[t] = {
            "tau": tau_val,
            "params": fitted,
        }

    return VolSurface(np.array(taus, dtype=float), smiles), stats


def calib_global_mixture(
    n: int,
    option_chain: pd.DataFrame,
    loss_weights: float | np.ndarray = 1,
) -> LogNormMixParams:
    """Calibrate a single log-normal mixture to all expiry slices simultaneously.

    Args:
        n: Number of mixture components.
        option_chain: DataFrame with columns K, tau, df, fwd, price.
        loss_weights: Scalar or per-observation weights applied to residuals.
    """
    p0 = LogNormMixParams(np.repeat(1 / n, n), np.zeros(n), np.repeat(0.2, n))
    x0, unravel = make_ravel_param(p0, make_full_encoder(tau=0.5), check_unravel=True)

    #TODO: add a compensator for the forward.

    bounds = (
        np.concatenate([np.repeat(-np.inf, n - 1), np.repeat(-np.inf, n - 1), np.repeat(0.03, n)]),
        np.concatenate([np.repeat(np.inf, n - 1), np.repeat(np.inf, n - 1), np.repeat(np.inf, n)]),
    )

    def _loss_function(
        x: np.ndarray,
        tau: ArrayLike,
        disc: ArrayLike,
        fwd: ArrayLike,
        k: ArrayLike,
        mkt_opt_p: ArrayLike,
    ) -> np.ndarray:
        param = unravel(x)
        model_price = _mixed_log_norm_call(
            w=param.w,
            mu=param.mu,
            sigma=param.sigma,
            DF=disc,
            F=fwd,
            K=k,
            tau=tau,
        )
        return model_price - mkt_opt_p

    res = least_squares(
        fun=lambda x: loss_weights
        * _loss_function(x, option_chain.tau, option_chain.df, option_chain.fwd, option_chain.K, option_chain.price),
        x0=x0,
        jac="2-point",
        method="trf",
        bounds=bounds,
    )

    if not res.success:
        log.warning("Global log-normal mixture calibration did not converge: %s", res.message)

    return unravel(res.x)


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
    x_ = np.asarray(x, dtype=float)[:, np.newaxis]              # (N, 1)
    w_ = np.asarray(mix_weights, dtype=float)[np.newaxis, :]    # (1, K)
    mu_ = np.asarray(mu, dtype=float)[np.newaxis, :]            # (1, K)
    sigma_ = np.asarray(sigma, dtype=float)[np.newaxis, :]          # (1, K)
    pdf = (1.0 / (sigma_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ - mu_) / sigma_) ** 2)  # (N, K)
    return (w_ * pdf).sum(axis=1)


def gaussian_mixture_density_second_derivative(
    x: ArrayLike, mix_weights: ArrayLike, mu: ArrayLike, sigma: ArrayLike
) -> np.ndarray:
    """Compute second derivative of Gaussian mixture density analytically."""
    x_ = np.asarray(x, dtype=float)[:, np.newaxis]          # (N, 1)
    w_ = np.asarray(mix_weights, dtype=float)[np.newaxis, :]  # (1, K)
    mu_ = np.asarray(mu, dtype=float)[np.newaxis, :]          # (1, K)
    s_ = np.asarray(sigma, dtype=float)[np.newaxis, :]        # (1, K)
    pdf = (1.0 / (s_ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_ - mu_) / s_) ** 2)  # (N, K)
    return (w_ * pdf * ((x_ - mu_) ** 2 - s_**2) / s_**4).sum(axis=1)
