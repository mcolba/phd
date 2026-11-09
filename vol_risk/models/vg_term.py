r"""Additive Variance Gamma term structure with power-law scaling (Azzone & Baviera, 2022).

Implements the VG case ($\alpha = 0$) of the additive normal tempered stable process with an
exponentially mean-reverting forward-variance curve.

The paper's scaling exponents $\beta$ and $\delta$ are named ``k_exp`` and
``eta_exp`` here, because ``beta`` already denotes the VGSI factor loading; the
forward-variance mean-reversion timescale $\ell$ is ``variance_timescale``. The integrated
variance clock is written $\vartheta$ to distinguish it from the VG drift $\theta$.

    Azzone, M., & Baviera, R. (2022). Additive normal tempered stable processes for equity
    derivatives and power-law scaling. Quantitative Finance, 22(3), 501-518.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from vol_risk.models.vg_base import VGParams

    FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AdditiveVGScaling:
    r"""Power-law exponents and variance mean-reversion timescale shared by all factors.

    Attributes:
        k_exp: Paper's $\beta$: $\hat k(\tau) = \bar k\,\vartheta(\tau)^{\beta}$, $0 < \beta \le 1$.
        eta_exp: Paper's $\delta$: $\eta(\tau) = \bar\eta\,\vartheta(\tau)^{\delta}$, $-\beta < \delta \le 0$.
        variance_timescale: Forward-variance mean-reversion timescale $\ell$ in years.
    """

    k_exp: float = 1.0
    eta_exp: float = -0.5
    variance_timescale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.k_exp) or not 0.0 < self.k_exp <= 1.0:
            msg = "k_exp must be finite and lie in (0, 1] (Azzone-Baviera theorem 2.3 with alpha = 0)."
            raise ValueError(msg)
        if not np.isfinite(self.eta_exp) or not -self.k_exp < self.eta_exp <= 0.0:
            msg = "eta_exp must be finite and satisfy -k_exp < eta_exp <= 0."
            raise ValueError(msg)
        if not np.isfinite(self.variance_timescale) or self.variance_timescale <= 0.0:
            msg = "variance_timescale must be finite and positive."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AdditiveVGParams:
    r"""Factor state $(v_0, \bar k, \bar\eta, v_\infty)$ of the additive VG term structure.

    Attributes:
        v0: Instantaneous forward variance.
        k_bar: Kurtosis level $\bar k$.
        eta_bar: Skew level $\bar\eta$.
        v_inf: Long-run forward variance.
    """

    v0: float
    v_inf: float
    k_bar: float
    eta_bar: float

    def __post_init__(self) -> None:
        for name in ("v0", "k_bar", "eta_bar", "v_inf"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                msg = f"AdditiveVGParams.{name} must be finite and positive."
                raise ValueError(msg)

    def as_array(self) -> FloatArray:
        """Return ``(v0, k_bar, eta_bar, v_inf)`` as a float array."""
        return np.asarray([self.v0, self.k_bar, self.eta_bar, self.v_inf], dtype=float)


def integrated_variance(tau: ArrayLike, v0: ArrayLike, v_inf: ArrayLike, variance_timescale: float) -> FloatArray:
    r"""Return $\vartheta(\tau) = v_\infty\tau + (v_0 - v_\infty)\ell(1 - e^{-\tau/\ell})$ (appendix eq. 3)."""
    tau_arr, v0_arr, v_inf_arr = (np.asarray(value, dtype=float) for value in (tau, v0, v_inf))
    ell = variance_timescale
    return v_inf_arr * tau_arr + (v0_arr - v_inf_arr) * ell * (-np.expm1(-tau_arr / ell))


def vg_params_at_tau(
    v0: ArrayLike,
    k_bar: ArrayLike,
    eta_bar: ArrayLike,
    v_inf: ArrayLike,
    tau: ArrayLike,
    scaling: AdditiveVGScaling,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Return the standard VG parameters $(\sigma_\tau, \nu_\tau, \theta_\tau)$ (appendix eq. 4).

    With $\sigma_\tau^2 = \vartheta(\tau)/\tau$, $\hat k = \bar k\vartheta^{\beta}$ and
    $\eta = \bar\eta\vartheta^{\delta}$: $\nu_\tau = \hat k/\sigma_\tau^2$ and
    $\theta_\tau = -(\tfrac12 + \eta)\sigma_\tau^2$. All inputs broadcast together.
    """
    v0_arr, k_bar_arr, eta_bar_arr, v_inf_arr, tau_arr = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (v0, k_bar, eta_bar, v_inf, tau)]
    )
    clock = integrated_variance(tau_arr, v0_arr, v_inf_arr, scaling.variance_timescale)
    sigma_sq = clock / tau_arr
    sigma = np.sqrt(sigma_sq)
    nu = k_bar_arr * clock**scaling.k_exp / sigma_sq
    eta = eta_bar_arr * clock**scaling.eta_exp
    theta = -(0.5 + eta) * sigma_sq
    return sigma, nu, theta


def params_from_reference(
    reference: VGParams,
    tau_ref: float,
    v_inf: float,
    scaling: AdditiveVGScaling,
) -> AdditiveVGParams:
    r"""Invert eq. 4 at ``tau_ref`` to recover $(v_0, \bar k, \bar\eta)$ for a given $v_\infty$.

    The reference slice must have strictly positive ``sigma`` and ``nu`` and
    ``theta < -sigma**2 / 2`` so that the implied $\bar\eta$ is positive.
    """
    if not np.isfinite(tau_ref) or tau_ref <= 0.0:
        msg = "tau_ref must be finite and positive."
        raise ValueError(msg)
    if not np.isfinite(v_inf) or v_inf <= 0.0:
        msg = "v_inf must be finite and positive."
        raise ValueError(msg)
    if reference.sigma <= 0.0 or reference.nu <= 0.0:
        msg = "The reference VG slice must have strictly positive sigma and nu."
        raise ValueError(msg)
    sigma_sq = reference.sigma**2
    # Average forward variance over [0, tau_ref] relative to v0 - v_inf.
    ell = scaling.variance_timescale
    weight = (ell / tau_ref) * (-np.expm1(-tau_ref / ell))
    v0 = v_inf + (sigma_sq - v_inf) / weight
    if v0 <= 0.0:
        msg = "The reference slice implies non-positive v0; choose v_inf consistent with sigma at tau_ref."
        raise ValueError(msg)
    clock = tau_ref * sigma_sq
    eta_ref = -reference.theta / sigma_sq - 0.5
    if eta_ref <= 0.0:
        msg = "The reference slice implies non-positive eta_bar; theta must satisfy theta < -sigma**2 / 2."
        raise ValueError(msg)
    return AdditiveVGParams(
        v0=float(v0),
        k_bar=float(reference.nu * sigma_sq / clock**scaling.k_exp),
        eta_bar=float(eta_ref / clock**scaling.eta_exp),
        v_inf=float(v_inf),
    )
