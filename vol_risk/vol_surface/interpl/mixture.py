"""TODO:
- add constraints in forward-moneyness direction C(k_i, T1) < C(k_i, T2) for i in [k1, ..., kn].
- add regularisation:
  - T=0 -> prior, T>0 -> changes in parameters between maturities.
  - Smooth density (a al Le Floc’h).
- add post fit calendar arbitrage check and arbitrage repair algo.
- Calib stats: store/output error, listed mxt boundaries, and prices outside bid ask.
"""

from dataclasses import dataclass
from math import tau
from typing import Protocol

import numpy as np
from scipy import special
from scipy.optimize import least_squares

from vol_risk.models.black76 import black76_price, black76_vega, implied_vol_jackel
from vol_risk.protocols import EuropeanOption, ModelParams
from vol_risk.util import angles_to_simplex, make_ravel_param, simplex_to_angles


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
        """Validates parameters after initialization."""
        if not (len(self.w) == len(self.mu) == len(self.sigma)):
            msg = "Parameters 'w', 'mu', and 'sigma' must have the same length."
            raise ValueError(msg)

        if not np.isclose(np.sum(self.w), 1.0):
            msg = "The sum of weights 'w' must be equal to 1."
            raise ValueError(msg)


def _mixed_log_norm_call(
    w: np.array,
    mu: np.array,
    sigma: np.array,
    DF: np.array,
    F: np.array,
    K: np.array,
    tau: np.array,
) -> np.ndarray:
    w = np.asarray(w)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    if not (w.shape == mu.shape == sigma.shape):
        msg = "w, mu, sigma must have identical 1-D shapes"
        raise ValueError(msg)
    if not np.isclose(w.sum(), 1.0):
        msg = "mixture weights must sum to 1"
        raise ValueError(msg)

    return sum(
        w[i] * black76_price(df=DF, f=F * np.exp(mu[i] * tau), k=K, t=tau, sigma=sigma[i], is_call=True)
        for i in range(len(w))
    )


class LinearMarket(Protocol):
    """Protocol for a market providing discount factors and forward prices."""

    def disc(self, t: np.ndarray[float]) -> np.ndarray[float]: ...
    def fwd(self, t: np.ndarray[float]) -> np.ndarray[float]: ...


def mixed_log_norm_call(x: LogNormMixParams, mkt: LinearMarket, opt: EuropeanOption) -> np.array:
    """Wrapper for log normal mixture interpolator."""
    k, tau = opt.strike, opt.tau
    fwd = mkt.fwd(tau)
    disc = mkt.disc(tau)

    return _mixed_log_norm_call(
        w=x.w,
        mu=x.mu,
        sigma=x.sigma,
        DF=disc,
        F=fwd,
        K=k,
        tau=tau,
    )


def make_logn_mix_calib_full_encoder(tau) -> tuple:
    """Creates a bijection for log-normal mixture calibration parameters."""

    def encode(p: LogNormMixParams) -> tuple:
        """Encodes LogNormMixParams into parameters wnforcing the simplex constraints."""
        w, mu, sigma = p.w, p.mu, p.sigma
        z = w * np.exp(mu * tau)

        if not (sum(w) == 1 and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (sum(z) == 1 and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        x0 = simplex_to_angles(w)
        x1 = simplex_to_angles(z)

        free = (x0, x1, sigma)
        return (free, ())

    def decode(free, fixed: tuple[np.ndarray] | None) -> LogNormMixParams:
        """Reconstructs a LogNormMixParams from transformed parameter space."""
        x0, x1, sigma = free
        w = angles_to_simplex(x0)
        z = angles_to_simplex(x1)
        mu = np.log(z / w) / tau
        return LogNormMixParams(w=w, mu=mu, sigma=sigma)

    return (encode, decode)


def make_logn_mix_reduced_encoder(tau):
    """Creates a bijection for log-normal mixture calibration parameters."""

    def encode(p: LogNormMixParams) -> tuple:
        """Encodes LogNormMixParams into parameters wnforcing the simplex constraints."""
        w, mu, sigma = p.w, p.mu, p.sigma
        z = w * np.exp(mu * tau)

        if not (sum(w) == 1 and np.all(w >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        if not (sum(z) == 1 and np.all(z >= 0)):
            msg = "Not a bijection. Limit the domain to unit sphere coordinates."
            raise ValueError(msg)

        free = sigma
        fixed = (w, mu)
        return (free, fixed)

    def decode(free, fixed) -> tuple:
        """Reconstructs a LogNormMixParams from transformed parameter space."""
        w, mu = fixed
        sigma = np.squeeze(free)
        return LogNormMixParams(w=w, mu=mu, sigma=sigma)

    return (encode, decode)


def _mixed_log_norm_calib(n, k, t, f, df, p, w=1):
    """Calibrate a log-normal mixture model to option prices."""
    # Initial guess
    w0 = np.repeat(1 / n, n)
    mu0 = np.zeros(n)
    mu0[0] = -0.1
    mu0[-1] = np.log((1 - sum(w0[:-1] * np.exp(mu0[:-1] * t))) / w0[-1]) / t
    sigma0 = np.repeat(0.2, n)
    p0 = LogNormMixParams(w0, mu0, sigma0)
    x0, unravel = make_ravel_param(p0, make_logn_mix_reduced_encoder(tau=t), check_unravel=True)

    # bounds
    bounds = (np.repeat(0.03, n), np.repeat(np.inf, n))

    def _loss_function(x, tau, disc, fwd, k, mkt_opt_p) -> np.ndarray:
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
        fun=lambda x: w * (_loss_function(x, t, df, f, k, p)),
        x0=x0,
        jac="2-point",
        method="trf",
        bounds=bounds,
    )

    return unravel(res.x)


def _mixed_log_norm_slice_calib(n, k, t, f, df, p, w=1, reg_lambda=0.0):
    """Calibrate a log-normal mixture model to option prices."""
    # Initial guess
    p0 = LogNormMixParams(np.repeat(1 / n, n), np.repeat(0, n), np.repeat(0.2, n))
    x0, unravel = make_ravel_param(p0, make_logn_mix_calib_full_encoder(tau=t), check_unravel=True)

    # bounds
    bounds = (
        np.concatenate(
            [
                np.repeat(-np.inf, n - 1),
                np.repeat(-np.inf, n - 1),
                np.repeat(0.03, n),
            ]
        ),
        np.concatenate(
            [
                np.repeat(np.inf, n - 1),
                np.repeat(np.inf, n - 1),
                np.repeat(np.inf, n),
            ]
        ),
    )

    def _loss_function(x, tau, disc, fwd, k, mkt_opt_p, w, reg_lambda=0.0):
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

        residuals = model_price - mkt_opt_p

        if reg_lambda > 0.0:
            z_grid = np.linspace(-2, 2, 200)
            dz = z_grid[1] - z_grid[0]
            d2f_dx2 = gaussian_mixture_density_second_derivative(z_grid, param.w, param.mu, param.sigma)
            roughness = sum(d2f_dx2**2 * dz)

            atm_sigma = 0.16 * 0.9  # TODO(Marco): feed from previous slice
            baseline = 3 / (8 * np.sqrt(np.pi) * atm_sigma**5)
            excess_roughness = roughness - baseline

            def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
                return beta * special.softplus(x / beta)

            penalty = np.sqrt(softplus(excess_roughness, beta=0.1))
            residuals = np.concatenate([residuals, np.array([penalty])])
            w = np.concatenate([w, np.array([reg_lambda])])

        return w * residuals

    res = least_squares(
        fun=lambda x: (_loss_function(x, t, df, f, k, p, w, reg_lambda)),
        x0=x0,
        jac="2-point",
        method="trf",
        bounds=bounds,
    )

    return unravel(res.x)


def _mixed_log_norm_global_calib(n, df, w=1):
    """Calibrate a log-normal mixture model to option prices."""
    # Initial guess
    p0 = LogNormMixParams(np.repeat(1 / n, n), np.repeat(0, n), np.repeat(0.2, n))
    x0, unravel = make_ravel_param(p0, make_logn_mix_calib_full_encoder(tau=0.5), check_unravel=True)

    # TODO: add a compensator. The forward is not constant across maturities.

    # bounds
    bounds = (
        np.concatenate(
            [
                np.repeat(-np.inf, n - 1),
                np.repeat(-np.inf, n - 1),
                np.repeat(0.03, n),
            ]
        ),
        np.concatenate(
            [
                np.repeat(np.inf, n - 1),
                np.repeat(np.inf, n - 1),
                np.repeat(np.inf, n),
            ]
        ),
    )

    def _loss_function(x, tau, disc, fwd, k, mkt_opt_p):
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
        fun=lambda x: w * (_loss_function(x, df.tau, df.df, df.fwd, df.K, df.price)),
        x0=x0,
        jac="2-point",
        method="trf",
        bounds=bounds,
    )

    return unravel(res.x)


def _mixed_log_norm_sequential_calib(n, k, t, f, df, p, w=1):
    """Calibrate a log-normal mixture model to option prices."""
    # Initial guess
    p0 = LogNormMixParams(np.repeat(1 / n, n), np.repeat(0, n), np.repeat(0.2, n))
    x0, unravel = make_ravel_param(p0, make_logn_mix_calib_full_encoder(tau=t), check_unravel=True)

    # bounds
    bounds = (
        np.concatenate(
            [
                np.repeat(-np.inf, n - 1),
                np.repeat(-np.inf, n - 1),
                np.repeat(0.03, n),
            ]
        ),
        np.concatenate(
            [
                np.repeat(np.inf, n - 1),
                np.repeat(np.inf, n - 1),
                np.repeat(np.inf, n),
            ]
        ),
    )

    def _loss_function(x, tau, disc, fwd, k, mkt_opt_p):
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
        fun=lambda x: w * (_loss_function(x, t, df, f, k, p)),
        x0=x0,
        jac="2-point",
        method="trf",
        bounds=bounds,
    )

    return unravel(res.x)


def gaussian_pdf(x, mu, sigma):
    """Compute the Gaussian PDF for a mixture component."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_mixture_density(x, w, mu, sigma):
    """Compute risk-neutral density for a Gaussian mixture at moneyness points.

    Args:
        x: Points where to evaluate the density (array).
        w: Mixture weights (array).
        mu: Mixture means (array).
        sigma: Mixture volatilities (array).

    Returns:
        Array of densities at each x.
    """
    x = np.asarray(x)
    density = np.zeros_like(x, dtype=float)
    for w_i, mu_i, sigma_i in zip(w, mu, sigma, strict=False):
        density += w_i * gaussian_pdf(x, mu_i, sigma_i)
    return density


def gaussian_mixture_density_second_derivative(x, w, mu, sigma):
    """Compute second derivative of Gaussian mixture density analytically.

    Args:
        x: Points where to evaluate the second derivative (array).
        w: Mixture weights (array).
        mu: Mixture means (array).
        sigma: Mixture volatilities (array).

    Returns:
        Array of second derivatives at each x.
    """
    x = np.asarray(x)
    second_deriv = np.zeros_like(x, dtype=float)
    for w_i, mu_i, sigma_i in zip(w, mu, sigma, strict=False):
        second_deriv += w_i * gaussian_pdf(x, mu_i, sigma_i) * ((x - mu_i) ** 2 - sigma_i**2) / (sigma_i**4)
    return second_deriv


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.style.use("seaborn-v0_8")

    # Example strikes and maturity
    K = np.linspace(80, 120, 20)
    K_fine = np.linspace(70, 130, 100)
    F = 100
    tau = np.array([0.5])
    r = 0.0
    q = 0.0
    DF = np.exp(-r * tau)
    FWD = F * np.exp((r - q) * tau)

    # Define three different smiles using hard-coded arrays
    smiles = {
        "smirk": np.array(
            [
                0.20,
                0.198,
                0.196,
                0.194,
                0.192,
                0.19,
                0.188,
                0.186,
                0.184,
                0.182,
                0.18,
                0.178,
                0.176,
                0.174,
                0.174,
                0.174,
                0.174,
                0.174,
                0.174,
                0.174,
            ]
        ),
        "U-shape": np.array(
            [
                0.20,
                0.195,
                0.19,
                0.185,
                0.18,
                0.175,
                0.17,
                0.165,
                0.16,
                0.155,
                0.16,
                0.165,
                0.17,
                0.175,
                0.18,
                0.185,
                0.19,
                0.195,
                0.20,
                0.205,
            ]
        ),
        "W-shape": np.array(
            [
                0.19,
                0.175,
                0.17,
                0.165,
                0.16,
                0.165,
                0.17,
                0.175,
                0.18,
                0.185,
                0.18,
                0.175,
                0.17,
                0.165,
                0.16,
                0.165,
                0.17,
                0.175,
                0.18,
                0.19,
            ]
        ),
    }

    # EXAMPLE 1: slice calibration using full encoder
    fitted_mixtures_full = {}
    for name, sigma in smiles.items():
        # Generate synthetic option prices using Black76 (original grid)
        prices = np.array(
            [black76_price(df=DF, f=FWD, k=k, t=tau, sigma=s, is_call=True)[0] for k, s in zip(K, sigma, strict=False)]
        )

        vega = np.array([black76_vega(df=DF, f=FWD, k=k, t=tau, sigma=s)[0] for k, s in zip(K, sigma, strict=False)])

        # Fit mixture model (original grid) using full encoder
        n_components = 3
        fitted = _mixed_log_norm_slice_calib(
            n=n_components,
            k=K,
            t=tau,
            f=FWD,
            df=DF,
            p=prices,
            w=1 / vega,
            reg_lambda=0.0001,
        )
        fitted_mixtures_full[name] = fitted

        # Compute fitted prices and implied vols on fine grid
        fitted_prices_fine = _mixed_log_norm_call(
            w=fitted.w,
            mu=fitted.mu,
            sigma=fitted.sigma,
            DF=DF,
            F=FWD,
            K=K_fine,
            tau=tau,
        )
        fitted_iv_fine = np.array(
            [
                implied_vol_jackel(df=float(DF), f=float(FWD), k=k, t=float(tau), price=fp, theta=1)
                for k, fp in zip(K_fine, fitted_prices_fine, strict=False)
            ]
        )

        plt.plot(K, sigma, "o", label=f"{name} market IV (orig)")
        plt.plot(K_fine, fitted_iv_fine, "--", label=f"{name}")

        print(f"\n{name} smile fit (full encoder):")
        print("Weights:", np.round(fitted.w, 4))
        print("Means (mu):", np.round(fitted.mu, 4))
        print("Vols (sigma):", np.round(fitted.sigma, 4))

    plt.xlabel("Strike (K)")
    plt.ylabel("Implied Volatility")
    plt.title("Full encoder: Market vs Fitted IV")
    plt.legend()
    plt.show()

    # Plot risk-neutral densities for each mixture (full encoder)
    plt.figure()
    fine_m_grid = np.linspace(-1, 1, 500)
    for name, fitted in fitted_mixtures_full.items():
        density = gaussian_mixture_density(fine_m_grid, fitted.w, fitted.mu, fitted.sigma)
        plt.plot(fine_m_grid, density, label=f"{name}")
    plt.xlabel("Moneyness")
    plt.ylabel("Risk-neutral density")
    plt.title("Full encoder: Risk-neutral density")
    plt.legend()
    plt.show()

    # # EXAMPLE 2:  fixed weigts and means
    # fitted_mixtures = {}
    # for name, sigma in smiles.items():
    #     # Generate synthetic option prices using Black76 (original grid)
    #     prices = np.array(
    #         [black76_price(df=DF, f=FWD, k=k, t=tau, sigma=s, is_call=True)[0] for k, s in zip(K, sigma, strict=False)]
    #     )

    #     # Fit mixture model (original grid)
    #     n_components = 5
    #     fitted = _mixed_log_norm_calib(
    #         n=n_components,
    #         k=K,
    #         t=tau,
    #         f=FWD,
    #         df=DF,
    #         p=prices,
    #         w=1,
    #     )
    #     fitted_mixtures[name] = fitted

    #     # Compute fitted prices and implied vols on fine grid
    #     fitted_prices_fine = _mixed_log_norm_call(
    #         w=fitted.w,
    #         mu=fitted.mu,
    #         sigma=fitted.sigma,
    #         DF=DF,
    #         F=FWD,
    #         K=K_fine,
    #         tau=tau,
    #     )
    #     fitted_iv_fine = np.array(
    #         [
    #             implied_vol_jackel(df=float(DF), f=float(FWD), k=k, t=float(tau), price=fp, theta=1)
    #             for k, fp in zip(K_fine, fitted_prices_fine, strict=False)
    #         ]
    #     )

    #     plt.plot(K, sigma, "o", label=f"{name} market IV (orig)")
    #     plt.plot(K_fine, fitted_iv_fine, "--", label=f"{name}")

    #     print(f"\n{name} smile fit:")
    #     print("Weights:", np.round(fitted.w, 4))
    #     print("Means (mu):", np.round(fitted.mu, 4))
    #     print("Vols (sigma):", np.round(fitted.sigma, 4))

    # plt.xlabel("Strike (K)")
    # plt.ylabel("Implied Volatility")
    # plt.title("Time varying vols: Market vs Fitted IV")
    # plt.legend()
    # plt.show()

    # # Plot risk-neutral densities for each mixture
    # plt.figure()
    # fine_m_grid = np.linspace(-1, 1, 500)
    # for name, fitted in fitted_mixtures.items():
    #     density = gaussian_mixture_density(fine_m_grid, fitted.w, fitted.mu, fitted.sigma)
    #     plt.plot(fine_m_grid, density, label=f"{name}")
    # plt.xlabel("Moneyness")
    # plt.ylabel("Risk-neutral density")
    # plt.title("Time varying vols: Risk-neutral density")
    # plt.legend()
    # plt.show()

    # # EXAMPLE 3:  global calibration
    # n_components = 5
    # tau = [0.1, 0.5, 1.0]

    # import pandas as pd

    # def make_df(k, sigma, tau):
    #     """Create a DataFrame for given smiles and tau."""
    #     return pd.DataFrame(
    #         {
    #             "K": k,
    #             "sigma": sigma,
    #             "tau": tau,
    #             "spot": 100,
    #             "fwd": 100,
    #             "df": 1,
    #             "price": black76_price(df=1, f=100, k=k, t=tau, sigma=sigma, is_call=True),
    #         }
    #     )

    # option_chain = pd.concat(
    #     [
    #         make_df(K, smiles["W-shape"], 0.1),
    #         make_df(K, smiles["U-shape"], 0.5),
    #         make_df(K, smiles["smirk"], 1.0),
    #     ],
    #     ignore_index=True,
    # )

    # global_fitted = _mixed_log_norm_global_calib(
    #     n=n_components,
    #     df=option_chain,
    # )

    # print("\nGlobal smile fit:")
    # print("Weights:", np.round(global_fitted.w, 4))
    # print("Means (mu):", np.round(global_fitted.mu, 4))
    # print("Vols (sigma):", np.round(global_fitted.sigma, 4))

    # # Plot global calibration IV fits
    # plt.figure()
    # for tau_val in sorted(option_chain["tau"].unique()):
    #     mask = option_chain["tau"] == tau_val
    #     K_orig = option_chain.loc[mask, "K"].to_numpy()
    #     sigma_orig = option_chain.loc[mask, "sigma"].to_numpy()
    #     DF_val = option_chain.loc[mask, "df"].iloc[0]
    #     FWD_val = option_chain.loc[mask, "fwd"].iloc[0]
    #     K_fine = np.linspace(K_orig.min() * 0.9, K_orig.max() * 1.1, 100)
    #     fitted_prices_fine = _mixed_log_norm_call(
    #         w=global_fitted.w,
    #         mu=global_fitted.mu,
    #         sigma=global_fitted.sigma,
    #         DF=DF_val,
    #         F=FWD_val,
    #         K=K_fine,
    #         tau=tau_val,
    #     )
    #     fitted_iv_fine = np.array(
    #         [
    #             implied_vol_jackel(df=DF_val, f=float(FWD_val), k=k, t=tau_val, price=fp, theta=1)
    #             for k, fp in zip(K_fine, fitted_prices_fine, strict=True)
    #         ]
    #     )
    #     plt.plot(K_orig, sigma_orig, "o", label=f"Market IV (tau={tau_val})")
    #     plt.plot(K_fine, fitted_iv_fine, "--", label=f"Fitted IV (tau={tau_val})")
    # plt.xlabel("Strike (K)")
    # plt.ylabel("Implied Volatility")
    # plt.title("Global Calibration: Market vs Fitted IV")
    # plt.legend()
    # plt.show()

    # # Plot global calibration density
    # plt.figure()
    # fine_m_grid = np.linspace(-1, 1, 500)
    # FWD_val0 = option_chain["fwd"].iloc[0]
    # density = gaussian_mixture_density(fine_m_grid, global_fitted.w, global_fitted.mu, global_fitted.sigma)
    # plt.plot(fine_m_grid, density, label="Global Mixture Density")
    # plt.xlabel("Moneyness")
    # plt.ylabel("Risk-neutral density")
    # plt.title("Global Calibration: Risk-neutral density")
    # plt.legend()
    # plt.show()
