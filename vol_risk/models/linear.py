import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from vol_risk.protocols import OptionChain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearEquityMarket:
    """Linear model for equity pricing with forward price calculation."""

    spot: float
    disc_curve: Callable
    cont_carry_curve: Callable

    def fwd(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the forward price."""
        return self.spot * np.exp(-self.cont_carry_curve(tau) * tau) / self.disc_curve(tau)

    def df(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the discount factor."""
        return self.disc_curve(tau)

    def zero_rate(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the zero rate."""
        return -np.log(self.disc_curve(tau)) / tau

    def zero_dvd_yield(self, tau: ArrayLike) -> ArrayLike:
        """Calculate the zero dividend yield."""
        return self.cont_carry_curve(tau)


def make_raw_interpolator(
    tau: np.ndarray,
    r: np.ndarray,
    add_zero_anchor: bool = True,
    flat_extrap: bool = True,
) -> Callable[[ArrayLike], ArrayLike]:
    """Create a discount curve using flat forward interpolation.

    Parameters:
    - tau: array of maturities (must be increasing)
    - r: array of zero rates
    - add_zero_anchor: add a zero point at tau = 0 if True

    Returns:
    - discount(t): function returning discount factor at time t (scalar or array)

    Source: https://downloads.dxfeed.com/specifications/dxLibOptions/HaganWest.pdf
    """
    tau = np.squeeze(np.asarray(tau))
    r = np.squeeze(np.asarray(r))

    if tau.ndim != 1 or r.ndim != 1:
        msg = "tau and r must be 1-dimensional arrays."
        raise ValueError(msg)
    if tau.size != r.size:
        msg = "tau and r must have the same length."
        raise ValueError(msg)
    if np.any(np.diff(tau) <= 0):
        msg = "tau must be strictly increasing."
        raise ValueError(msg)
    if tau.size < 2:
        msg = "Need at least two points for interpolation and extrapolation."
        raise ValueError(msg)

    # Interpolate linearly in r * t = -log D(t)
    rt = tau * r

    if add_zero_anchor and tau[0] > 0:
        tau = np.insert(arr=tau, obj=0, values=0.0)
        rt = np.insert(arr=rt, obj=0, values=0.0)

    interp_rt = interp1d(
        x=tau,
        y=rt,
        kind="linear",
        fill_value=(rt[0], rt[-1]),
        assume_sorted=True,
        bounds_error=False,
    )

    def _zc(x: float | np.ndarray) -> float | np.ndarray:
        if flat_extrap:
            x = np.clip(x, tau[0], tau[-1])

        x = np.asarray(x, dtype=float)
        y = interp_rt(x) / x
        return float(y) if np.ndim(x) == 0 else y

    return _zc


def make_raw_disc_curve(
    tau: np.ndarray,
    r: np.ndarray,
    add_zero_anchor: bool = True,
) -> Callable[[ArrayLike], ArrayLike]:
    interp = make_raw_interpolator(tau=tau, r=r, add_zero_anchor=add_zero_anchor)

    def _disc(x: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.exp(-interp(x) * x)
        return float(y) if np.ndim(x) == 0 else y

    return _disc


def make_simple_linear_market(s: float = 100.0, r: float = 0, q: float = 0) -> LinearEquityMarket:
    """Creates a dummy linear market data object."""
    return LinearEquityMarket(
        spot=s,
        disc_curve=lambda tau: np.exp(-r * tau),
        cont_carry_curve=lambda _: q,
    )


def put_call_df(opt: OptionChain) -> pd.DataFrame:
    """Create a DataFrame with call-put differences and bid-ask bounds."""
    pt = opt.df.pivot_table(index=["strike", "spot"], columns="option_type", values=["mid", "bid", "ask"]).pipe(
        lambda x: x.loc[x.notna().all(axis=1), :]
    )

    g_mid = pt.loc[:, ("mid", "C")] - pt.loc[:, ("mid", "P")]
    g_min = pt.loc[:, ("bid", "C")] - pt.loc[:, ("ask", "P")]
    g_max = pt.loc[:, ("ask", "C")] - pt.loc[:, ("bid", "P")]

    return (
        pd.DataFrame(index=pt.index)
        .assign(
            g_mid=g_mid,
            g_min=g_min,
            g_max=g_max,
        )
        .sort_values(by="strike")
        .reset_index()
    )


def calib_linear_equity_market(opt: OptionChain, axes=False) -> tuple[LinearEquityMarket, dict]:
    """Calibrate a linear equity market model to an option chain.

    The put-call parity is used to extract implied interest rates (r) and income yields (q) via linear regression:
        C_t - P_t = S * exp(-q_t * t) - K * exp(-r_t * t) + epsilon,
    where alpha_t = S * exp(-q_t * t) and beta_t = exp(-r_t * t) are the regression coefficients.

    Reference: Binsbergen et al. 2022. "Risk-Free Interest Rates". Journal of Financial Economics 143 (1): 1-29. https://doi.org/10.1016/j.jfineco.2021.06.012.

    """
    n = np.unique(opt.expiry).size

    tau = np.empty(n, dtype=float)
    alpha = np.empty(n, dtype=float)
    beta = np.empty(n, dtype=float)

    valid_idx = np.ones(n, dtype=bool)
    stats = {}

    for i, (t, sl) in enumerate(opt):
        pc_df = put_call_df(sl)

        if pc_df.shape[0] < 10:
            msg = f"Maturity {t} has less than 10 observables. It will be skipped."
            logger.warning(msg)
            valid_idx[i] = False

        # Calculate P - C and perform linear regression against strike (K)
        K = pc_df["strike"].to_numpy().reshape(-1, 1)
        y = pc_df["g_mid"].to_numpy()

        # Fit linear regression
        lr = LinearRegression(fit_intercept=True).fit(X=-K, y=y)
        alpha_t = lr.intercept_
        beta_t = lr.coef_[0]

        if beta_t <= 0 or alpha_t <= 0:
            msg = f"Calibrated alpha/beta must be positive. Maturity {t} will be skipped."
            logger.warning(msg)
            valid_idx[i] = False

        # check if fitted line is within bid-ask bounds
        fitted = lr.predict(-K)
        in_bid_ask_t = np.all((fitted >= pc_df["g_min"]) & (fitted <= pc_df["g_max"]))

        if not in_bid_ask_t:
            msg = f"Fitted line for maturity {t} is not within the put-call bid-ask bounds."
            logger.warning(msg)

        if axes is not None:
            moneyness = K.ravel() / opt.spot
            axes[i].plot(moneyness, fitted, color="orange")
            axes[i].fill_between(moneyness, pc_df["g_min"], pc_df["g_max"], color="lightgray", alpha=0.5)
            axes[i].scatter(moneyness, y, color="blue", s=20, label="C-P")
            axes[i].text(0.1, 0.9, f"T={t.date()}, τ= {sl.tau[0]:.2f}", transform=axes[i].transAxes)

        # Append results
        tau[i] = sl.tau[0]
        alpha[i] = alpha_t
        beta[i] = beta_t
        stats[t] = {
            "coeff": (alpha_t, beta_t),
            "n_obs": int(pc_df.shape[0]),
            "in_bid_ask": bool(in_bid_ask_t),
            "tau": float(tau[i]),
            "excluded": not valid_idx[i],
        }

    spot = opt.spot
    r = -np.log(beta[valid_idx]) / tau[valid_idx]
    q = -np.log(alpha[valid_idx] / spot) / tau[valid_idx]

    model = LinearEquityMarket(
        spot=float(spot),
        disc_curve=make_raw_disc_curve(tau=tau[valid_idx], r=r),
        cont_carry_curve=make_raw_interpolator(tau=tau[valid_idx], r=q),
    )

    return model, stats
