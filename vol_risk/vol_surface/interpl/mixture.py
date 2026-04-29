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
from vol_risk.models.black76 import black76_price, black76_vega, implied_vol, implied_vol_jackel
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.protocols import EuropeanOption, ModelParams, OptionChainLike
from vol_risk.util import angles_to_simplex, make_ravel_param, simplex_to_angles
from vol_risk.vol_surface.surface import VolSmile, VolSurface

log = logging.getLogger(__name__)

SIGMA_MAX = 4.0
SIGMA_MIN = 0.03
THETA_1_EPSILON = 0.1
THETA_2_EPSILON = 0.15


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
    weight_function: Callable
    lambda_rough: float
    lambda_w: float
    lambda_mu: float
    lambda_sigma: float
    x0: LogNormMixParams | None


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

    # TODO(Marco): vectorise
    return (1 - pdef) * np.sum(
        w[i] * black76_price(df=DF, f=F * np.exp(mu[i] * tau) / (1 - pdef), k=K, t=tau, sigma=sigma[i], is_call=True)
        for i in range(len(w))
    )


def mixed_log_norm(
    w: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    DF: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    tau: ArrayLike,
    is_call: ArrayLike,
    pdef: float = 0,
) -> np.ndarray:
    """Low-level function returning put option price under a log-normal mixture model."""
    w = np.asarray(w)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if not (w.shape == mu.shape == sigma.shape):
        msg = "w, mu, and sigma must have identical 1-D shapes"
        raise ValueError(msg)
    if not np.isclose(w.sum(), 1.0):
        msg = "mixture weights must sum to 1"
        raise ValueError(msg)

    # TODO(Marco): vectorise
    return (1 - pdef) * np.sum(
        w[i] * black76_price(df=DF, f=F * np.exp(mu[i] * tau) / (1 - pdef), k=K, t=tau, sigma=sigma[i], is_call=is_call)
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
    tau = float(tau)

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

        if method == "simplex":
            z = angles_to_simplex(x1)
            mu = np.log(z / w) / tau
        elif method == "manual":
            partial_sum = np.dot(w[:-1], np.exp(x1 * tau))
            if (1 - partial_sum) <= 0:
                msg = "Invalid parameters: remaining forward mass <= 0. Use simplex method instead."
                raise ValueError(msg)
            mu_n = np.log((1 - partial_sum) / w[-1]) / tau
            mu = np.append(x1, mu_n)
        else:
            msg = f"Unsupported bijection method: {method!r}. Use 'simplex' or 'manual'."
            raise ValueError(msg)

        return LogNormMixParams(w=w, mu=mu, sigma=sigma)

    return (encode, decode)


def make_full_encoder_totvar(tau: float, method: str = "simplex") -> tuple:
    """Creates a bijection for log-normal mixture calibration parameters with additive total variance."""
    tau = float(tau)

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
            mu = np.append(x1, mu_n)
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


def _require_call_only(chain: OptionChainLike) -> None:
    # TODO: move out.
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

BIJECTION_FALLBACK = {
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
                np.repeat(-np.inf, n - 1),
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

    # Assign smaller weight to the leftmost component
    w_left = 1 / (n * 3)
    w_right = np.repeat((1 - w_left) / (n - 1), n - 1)
    w0 = np.concatenate(([w_left], w_right))

    # Assignincreasing mu values
    exp_mu_min = 0.85
    exp_mu_left = np.linspace(exp_mu_min, 1, n - 1)
    partial_sum = np.dot(w0[:-1], exp_mu_left)
    exp_mu_right = (1 - partial_sum) / w0[-1]
    mu0 = np.log(np.concatenate([exp_mu_left, [exp_mu_right]])) / tau

    # Assign decreasing sigma values
    sigma0 = np.clip(piecewise_linspace([sigma_atm * 3, sigma_atm, sigma_atm * 0.5], n), SIGMA_MIN, SIGMA_MAX)

    return LogNormMixParams(w0, mu0, sigma0)


def _uninformative_start_guess(n: int, sigma_atm: float, tau: float) -> LogNormMixParams:
    """Generate initial guess for flat smiles."""
    if n < 2:
        msg = "Number of components must be at least 2."
        raise ValueError(msg)

    w0 = np.repeat(1 / n, n)
    exp_mu_min = 0.85
    exp_mu_max = 2 - exp_mu_min
    mu0 = np.log(np.linspace(exp_mu_min, exp_mu_max, n)) / tau
    sigma0 = np.clip(np.repeat(sigma_atm, n), SIGMA_MIN, SIGMA_MAX)
    return LogNormMixParams(w0, mu0, sigma0)


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
    lambda_calendar_arb: float = 0.0,
    pdef: float = 0.0,
    sigma_atm: float = 0.2,
    no_arb_bounds: pd.DataFrame | None = None,
) -> np.ndarray:
    """Calibrate a log-normal mixture model to option prices."""
    if p0 is None:
        p0 = _uninformative_start_guess(n, sigma_atm=sigma_atm, tau=float(tau))

    if transform_method not in BIJECTION_METHODS:
        msg = f"Unsupported transform method: {transform_method}"
        raise ValueError(msg)

    min_vol = 0.0 if "totvar" in transform_method else SIGMA_MIN

    encoder = BIJECTION_METHODS[transform_method](float(tau))
    x0, unravel = make_ravel_param(p0, encoder, check_unravel=False)

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

        if lambda_smoothing > 0.0:
            penalty = np.sqrt(softplus(excess_roughness(param, sigma_atm=sigma_atm), beta=0.1))
            residuals = np.concatenate([residuals, np.array([penalty])])
            weights = np.concatenate([weights, np.array([lambda_smoothing])])

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

        if no_arb_bounds is not None:
            prices_norm = _mixed_log_norm_call(
                w=param.w,
                mu=param.mu,
                sigma=param.sigma,
                DF=df,
                F=fwd,
                K=no_arb_bounds["strike"].values,
                tau=tau,
                pdef=pdef,
            ) / (fwd * df)
            prices_ub = no_arb_bounds["price_norm_ub"].to_numpy()
            arbitrage = softplus(prices_norm - prices_ub, beta=1e-6)

            if "price_norm_lb" in no_arb_bounds.columns:
                prices_lb = no_arb_bounds["price_norm_lb"].to_numpy()
                arbitrage_lb = softplus(prices_lb - prices_norm, beta=1e-6)
                arbitrage = np.concatenate([arbitrage, arbitrage_lb])

            residuals = np.concatenate([residuals, arbitrage])
            weights = np.concatenate([weights, np.repeat(lambda_calendar_arb, arbitrage.size)])

        return weights * residuals

    res = least_squares(
        fun=lambda x: _loss_function(x),
        x0=x0,
        jac="3-point",
        method="trf",
        bounds=bounds,
        x_scale="jac",
    )

    if not res.success:
        msg = f"Mixture calibration did not converge for tau={float(tau):.2f}): {res.message}"
        log.warning(msg)

    stats = {
        "error": res.fun[: len(mkt_prices)],
        "mse": np.mean(res.fun[: len(mkt_prices)] ** 2),
        "success": res.success,
        "message": res.message,
        "cost": res.cost,
    }

    return unravel(res.x), stats


def _make_smile_fun(params: LogNormMixParams, le: LinearEquityMarket, tau: float, pdef: float = 0.0) -> VolSmile:
    """Construct a VolSmile object from calibrated log-normal mixture parameters."""
    tau = float(tau)
    df = le.df(tau)
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
        prices = mixed_log_norm(
            w=params.w,
            mu=params.mu,
            sigma=params.sigma,
            DF=df,
            F=fwd,
            K=k_arr,
            tau=tau,
            pdef=pdef,
            is_call=is_call,
        )

        iv = np.empty_like(k_arr, dtype=float)
        for i, (ki, pi, ci) in enumerate(zip(k_arr, prices, is_call, strict=True)):
            price = float(pi)
            is_call = bool(ci)

            iv[i] = implied_vol_jackel(
                price=price,
                f=fwd,
                k=float(ki),
                t=tau,
                df=df,
                is_call=ci,
            )

        return float(iv[0]) if k_is_scalar else iv

    return VolSmile(interpl=fun)


def _vega_weights(opt: OptionChainLike, line_mkt: LinearEquityMarket) -> np.ndarray:
    fwd = line_mkt.fwd(opt.tau)
    disc = line_mkt.df(opt.tau)

    k, tau, mid = opt.k, opt.tau, opt.mid
    is_call = opt.option_type == "C"

    iv = np.array(
        [
            implied_vol_jackel(price=mid, f=fwd, k=k, t=tau, df=disc, is_call=is_call)
            for mid, k, tau, is_call, disc, fwd in zip(mid, k, tau, is_call, disc, fwd, strict=True)
        ],
        dtype=float,
    ).clip(0.01, 1.5)

    vega = black76_vega(df=disc, f=fwd, k=k, t=tau, sigma=iv)
    return 1 / np.maximum(vega, 1e-4)


def calib_mixture_ivs(
    opt: OptionChainLike,
    mkt: LinearEquityMarket,
    n_components: int,
    lw_type: str | None = None,
    pdef: float = 0.0,
    x0: LogNormMixParams | None = None,
    transform_method: str = "base",
    t0_start_guess: str = "uninformative",
    lambda_smoothing: float = 0.0,
    lambda_tm1_params: tuple[float, float, float] = (0.0, 0.0, 0.0),
    calendar_arb_bounds: NoArbBounds | None = None,
) -> tuple[VolSurface, LogNormMixParams]:
    """Calibrate a log-normal mixture model to each expiry slice."""
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
        df = float(mkt.df(tau_vec)[0])
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
            # p0 = _smirk_start_guess(n_components, sigma_atm=sigma_atm, tau=tau)
            # p0 = _uninformative_start_guess(n_components, sigma_atm=sigma_atm, tau=tau)
            p0 = INITIAL_GUESS_METHODS[t0_start_guess](n_components, sigma_atm=sigma_atm, tau=tau)
            lambda_w = lambda_mu = lambda_sigma = 0.0
            transform_method_ = BIJECTION_FALLBACK.get(transform_method, transform_method)
            if transform_method_ != transform_method:
                msg = f""" Transform method '{transform_method}' is not supported for the first slice.
                Falling back to '{transform_method_}'."""
                log.info(msg)
        else:
            transform_method_ = transform_method
            p0 = _force_mu_to_unit_sum(prev_params, tau)
            lambda_w, lambda_mu, lambda_sigma = lambda_tm1_params
            if "totvar" in transform_method_ and prev_tau is not None:
                # Adjust sigma to keep total variance constant
                scaled_sigma = prev_params.sigma * np.sqrt(prev_tau / tau)
                p0 = LogNormMixParams(w=p0.w, mu=p0.mu, sigma=scaled_sigma)

        bounds_df = None
        if calendar_arb_bounds is not None:
            bounds_df = calendar_arb_bounds[t].call_ub
            if prev_tau is not None:
                k_tm1 = bounds_df["strike"] / fwd * mkt.fwd(prev_tau)
                norm_denom = mkt.df(prev_tau) * mkt.fwd(prev_tau)
                bounds_df["price_norm_lb"] = (
                    black76_price(
                        df=mkt.df(prev_tau),
                        f=mkt.fwd(prev_tau),
                        k=k_tm1,
                        t=prev_tau,
                        sigma=smiles[-1].vol(k_tm1),
                        is_call=True,
                    )
                    / norm_denom
                )

        fitted, stats_t = calib_mixture_smile(
            n=n_components,
            k=k_sl,
            tau=tau,
            fwd=fwd,
            df=df,
            mkt_prices=mid_sl,
            loss_weights=loss_weights,
            p0=p0,
            pdef=pdef,
            prev_params=prev_params,
            lambda_w=lambda_w,
            lambda_mu=lambda_mu,
            lambda_sigma=lambda_sigma,
            transform_method=transform_method_,
            lambda_calendar_arb=mkt.spot,
            lambda_smoothing=lambda_smoothing,
            sigma_atm=sigma_atm,
            no_arb_bounds=bounds_df,
        )

        # Calculate summary statistics
        model_price = _mixed_log_norm_call(
            w=fitted.w,
            mu=fitted.mu,
            sigma=fitted.sigma,
            DF=df,
            F=fwd,
            K=k_sl,
            tau=tau,
            pdef=pdef,
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
        smiles.append(_make_smile_fun(fitted, mkt, tau, pdef=pdef))
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
    df = pd.concat(
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

    bid_ask_width = df["ask"] - df["bid"]
    iv_error_approx = (df["mid"] - df["model_price"]) * df["div_dp"]
    outside_lower = (df["bid"] - df["model_price"]).clip(lower=0.0)
    outside_upper = (df["model_price"] - df["ask"]).clip(lower=0.0)
    outside_bid_ask_price_error = outside_lower + outside_upper

    iv_error_outside_bid_ask = (outside_bid_ask_price_error * df["div_dp"]).abs()
    in_bid_ask = (df["bid"] <= df["model_price"]) & (df["model_price"] <= df["ask"])
    in_2x_bid_ask = (df["mid"] - 2 * bid_ask_width <= df["model_price"]) & (
        df["model_price"] <= df["mid"] + 2 * bid_ask_width
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
