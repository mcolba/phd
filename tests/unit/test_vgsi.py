from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from vol_risk.models.black76 import black76_price
from vol_risk.models.numerical.fourier.fft_jax import (
    JaxFFTCallEngine,
    JaxFFTCallEngineParams,
    make_jax_fft_call_grid,
)
from vol_risk.models.vgsi import (
    VGSIParams,
    _vg_cf,
    _vgsi_call_price,
    _vgsi_cf,
    _vgsi_compensator,
    estimate_factor_residual_moments,
    make_vgsi_cf,
    vgsi_price,
)
from vol_risk.models.vgsi_jax import (
    VGSIJaxParams,
    vgsi_price_jax,
)
from vol_risk.models.vgsi_jax import (
    make_vgsi_cf as make_vgsi_cf_jax,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "vg_prices_quantlib.csv"

ZERO_PARAMS = VGSIParams(sigma=0.0, nu=0.0, theta=0.0)


def _jax_params_from(p: VGSIParams) -> VGSIJaxParams:
    return VGSIJaxParams(sigma=jnp.asarray(p.sigma), nu=jnp.asarray(p.nu), theta=jnp.asarray(p.theta))


@pytest.fixture(scope="module")
def ql_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def sony_market() -> dict[str, float | VGSIParams]:
    """Sony scenario from Carr & Madan (2012), Sec. 3.2."""
    return {
        "spot": 13010.0,
        "tau": 0.1205,
        "r": 0.0062,
        "q": 0.0,
        "factor_loading": 0.1931,
        "index_params": VGSIParams(sigma=0.2658, nu=0.0505, theta=-0.3855),
        "residual_params": VGSIParams(sigma=0.3031, nu=0.001, theta=0.0),
    }


def _row_params(row: pd.Series, *, factor_loading: float) -> VGSIParams:
    """Return VG params scaled by ``factor_loading`` for index-only pricing."""
    return VGSIParams(
        sigma=float(row.sigma) / factor_loading,
        nu=float(row.nu),
        theta=float(row.theta) / factor_loading,
    )


def _price_reduced_vgsi(
    ql_data: pd.DataFrame,
    *,
    mode: Literal["residual", "index"],
    factor_loading: float = 1.0,
    damping: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Price each ``ql_data`` row with one VG component active."""
    quad_opt = {
        "damping": damping,
        "breakpoints": (8.0, 64.0, 512.0),
        "adaptive_trunc": True,
    }
    prices: list[float] = []
    for row in ql_data.itertuples(index=False):
        tau = float(row.tau)
        active = VGSIParams(sigma=float(row.sigma), nu=float(row.nu), theta=float(row.theta))
        if mode == "residual":
            index_params, residual_params, loading = ZERO_PARAMS, active, 1.0
        else:
            index_params = _row_params(row, factor_loading=factor_loading)
            residual_params, loading = ZERO_PARAMS, factor_loading
        prices.append(
            float(
                vgsi_price(
                    spot=float(row.spot),
                    strike=float(row.strike),
                    tau=tau,
                    index_params=index_params,
                    factor_loading=loading,
                    residual_params=residual_params,
                    r=float(row.r),
                    q=float(row.q),
                    engine="quad",
                    engine_opt=quad_opt,
                    control="none",
                )
            )
        )
    return np.asarray(prices), ql_data["quantlib_call_price"].to_numpy()


def _cumulants_from_log_cf(log_cf_values: np.ndarray, h: float) -> tuple[float, float, float, float]:
    r"""Return $(\kappa_1, \kappa_2, \kappa_3, \kappa_4)$ from $\log\varphi$ samples."""
    coeffs_d1 = np.array([-1.0, 9.0, -45.0, 45.0, -9.0, 1.0]) / (60.0 * h)
    coeffs_d2 = np.array([2.0, -27.0, 270.0, 270.0, -27.0, 2.0]) / (180.0 * h**2)
    coeffs_d3 = np.array([-1.0, 8.0, -13.0, 13.0, -8.0, 1.0]) / (8.0 * h**3)
    coeffs_d4 = np.array([-1.0, 12.0, -39.0, -39.0, 12.0, -1.0]) / (6.0 * h**4)
    kappa1 = float((coeffs_d1 @ log_cf_values).imag)
    kappa2 = -float((coeffs_d2 @ log_cf_values).real)
    kappa3 = float((coeffs_d3 @ log_cf_values).imag)
    kappa4 = float((coeffs_d4 @ log_cf_values).real)
    return kappa1, kappa2, kappa3, kappa4


def _index_vg_cumulants(params: VGSIParams, tau: float, h: float = 0.1) -> tuple[float, float, float, float]:
    r"""Return cumulants of the VG random variable $X_\\tau$ extracted from its cf."""
    us = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]) * h
    phi = _vg_cf(u=us.astype(complex), tau=tau, params=params, scale=1.0)
    return _cumulants_from_log_cf(np.log(phi), h=h)


def test_paper_nikkei_index_cumulants_match_figure_3(sony_market: dict) -> None:
    r"""Reproduce the Nikkei VG stats quoted in Fig. 3 of Carr-Madan (2012).

    The caption reports annualised volatility, skewness and kurtosis
    $(0.2795,\\,-0.2022,\\,3.1792)$ for the risk-neutral index dynamics on
    July 27 1998 at $\\tau=0.126$.
    """
    tau = 0.126
    mean_t, var_t, c3_t, c4_t = _index_vg_cumulants(sony_market["index_params"], tau=tau)

    annual_vol = float(np.sqrt(var_t / tau))
    annual_skew = c3_t / var_t**1.5 * np.sqrt(tau)
    annual_kurt = 3.0 + c4_t / var_t**2 * tau

    assert mean_t == pytest.approx(sony_market["index_params"].theta * tau, abs=1.0e-6)
    assert annual_vol == pytest.approx(0.2795, abs=5.0e-4)
    assert annual_skew == pytest.approx(-0.2022, abs=5.0e-4)
    assert annual_kurt == pytest.approx(3.1792, abs=5.0e-4)


def test_estimate_factor_residual_moments() -> None:
    rng = np.random.default_rng(7)
    index_returns = rng.normal(loc=0.0002, scale=0.01, size=2000)
    residuals = rng.standard_t(df=8, size=2000) * 0.005
    stock_returns = 0.0004 + 1.35 * index_returns + residuals

    fit = estimate_factor_residual_moments(
        stock_log_returns=stock_returns,
        index_log_returns=index_returns,
        periods_per_year=252.0,
    )

    assert fit.factor_loading == pytest.approx(1.35, abs=0.04)
    assert fit.residual_sigma > 0.0
    assert fit.residual_nu >= 0.0
    assert fit.residual_params.theta == 0.0


@pytest.mark.parametrize(
    ("mode", "factor_loading"),
    [("residual", 1.0), ("index", 1.7)],
)
def test_reduced_vgsi_matches_quantlib_vg(
    ql_data: pd.DataFrame,
    mode: Literal["residual", "index"],
    factor_loading: float,
) -> None:
    prices, expected = _price_reduced_vgsi(ql_data, mode=mode, factor_loading=factor_loading, damping=0.03)

    abs_error = np.abs(prices - expected)
    rel_error = abs_error / expected
    not_small = expected >= 0.1

    assert float(abs_error.max()) <= 1.0e-4
    assert float(rel_error[not_small].max()) <= 1.0e-4


def test_reduced_vgsi_index_only_and_residual_only_agree(ql_data: pd.DataFrame) -> None:
    residual_prices, _ = _price_reduced_vgsi(ql_data, mode="residual")
    index_prices, _ = _price_reduced_vgsi(ql_data, mode="index", factor_loading=1.7)

    assert_allclose(index_prices, residual_prices, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize(
    ("engine", "rtol", "atol"),
    [
        ("quad", 1.0e-6, 1.0e-6),
        ("fft_np", 5.0e-4, 5.0e-4),
        ("fft_jax", 5.0e-4, 5.0e-4),
    ],
)
@pytest.mark.parametrize("control", ["none", "bs"])
def test_engine_dispatch_agrees_on_multi_strike(
    sony_market: dict,
    engine: Literal["quad", "fft_np", "fft_jax"],
    control: Literal["none", "bs"],
    rtol: float,
    atol: float,
) -> None:
    """All engines should agree on a strike vector within their numerical tolerance."""
    strikes = np.linspace(0.7, 1.4, 12) * sony_market["spot"]
    reference = np.asarray(
        vgsi_price(strike=strikes, is_call=True, engine="quad", control="bs", **sony_market),
        dtype=float,
    )
    if engine == "fft_jax":
        grid = make_jax_fft_call_grid(strike=strikes, params=JaxFFTCallEngineParams())
        prices = np.asarray(
            vgsi_price_jax(
                spot=sony_market["spot"],
                strike=jnp.asarray(strikes),
                tau=sony_market["tau"],
                index_params=_jax_params_from(sony_market["index_params"]),
                factor_loading=sony_market["factor_loading"],
                residual_params=_jax_params_from(sony_market["residual_params"]),
                r=sony_market["r"],
                q=sony_market["q"],
                is_call=True,
                control=control,
                grid=grid,
            ),
            dtype=float,
        )
    else:
        prices = np.asarray(
            vgsi_price(strike=strikes, is_call=True, engine=engine, control=control, **sony_market),
            dtype=float,
        )
    assert_allclose(prices, reference, rtol=rtol, atol=atol)


_PROPERTY_INDEX_PARAMS: list[VGSIParams] = [
    VGSIParams(0.20, 0.00, -0.05),  # Brownian-limit branch
    VGSIParams(0.10, 0.05, -0.20),  # low vol, low nu, negative skew
    VGSIParams(0.22, 0.30, -0.25),  # typical equity-like negative skew
    VGSIParams(0.45, 0.08, -0.40),  # high vol, low nu, strong negative skew
    VGSIParams(0.12, 1.20, 0.00),  # high nu, symmetric heavy tails
    VGSIParams(0.18, 0.50, 0.20),  # positive skew, moderate kurtosis
    VGSIParams(0.45, 1.00, -0.60),  # high vol, high nu, strong negative skew
    VGSIParams(0.30, 1.50, 0.35),  # near positive moment boundary at scale=1.5
]
_PROPERTY_FACTOR_LOADINGS: list[float] = [1.5]
_PROPERTY_RESIDUAL_PARAMS: list[VGSIParams] = [
    VGSIParams(0.10, 0.01, 0.0),
    VGSIParams(0.40, 0.20, 0.0),
]
_PROPERTY_REGIMES: list[tuple[VGSIParams, float, VGSIParams]] = list(
    product(
        _PROPERTY_INDEX_PARAMS,
        _PROPERTY_FACTOR_LOADINGS,
        _PROPERTY_RESIDUAL_PARAMS,
    )
)
_PROPERTY_TAUS: list[float] = [0.08, 1]
_PROPERTY_RATES: list[float] = [-0.05, 0.05]
_PROPERTY_DIVIDEND_YIELDS: list[float] = [-0.05, 0.05]
_PROPERTY_MONEYNESS: list[float] = [0.7, 1.0, 1.3]


def _all_property_mkt_cases() -> list[tuple[VGSIParams, float, VGSIParams, float, float, float, float]]:
    return [
        (idx, beta, res, tau, r, q, m)
        for (idx, beta, res), tau, r, q, m in product(
            _PROPERTY_REGIMES, _PROPERTY_TAUS, _PROPERTY_RATES, _PROPERTY_DIVIDEND_YIELDS, _PROPERTY_MONEYNESS
        )
    ]


def _regime_property_mkt_cases() -> list[tuple[VGSIParams, float, VGSIParams, float, float, float]]:
    return [
        (idx, beta, res, tau, r, q)
        for (idx, beta, res), tau, r, q in product(
            _PROPERTY_REGIMES,
            _PROPERTY_TAUS,
            _PROPERTY_RATES,
            _PROPERTY_DIVIDEND_YIELDS,
        )
    ]


@pytest.mark.parametrize(("idx", "beta", "res", "tau", "r", "q"), _regime_property_mkt_cases())
def test_property_martingale_condition(
    idx: VGSIParams, beta: float, res: VGSIParams, tau: float, r: float, q: float
) -> None:
    r"""Discounted stock is a martingale: $\\varphi(-i)=F$."""
    spot = 100.0
    fwd = spot * np.exp((r - q) * tau)

    phi_minus_i = _vgsi_cf(
        u=-1j,
        spot=spot,
        r=r,
        q=q,
        tau=tau,
        index_params=idx,
        factor_loading=beta,
        residual_params=res,
    )
    assert phi_minus_i == pytest.approx(fwd, rel=1.0e-10, abs=1.0e-9)


@pytest.mark.slow
@pytest.mark.parametrize(("idx", "beta", "res", "tau", "r", "q", "moneyness"), _all_property_mkt_cases())
def test_property_admissible_contour_yields_positive_prices(
    idx: VGSIParams,
    beta: float,
    res: VGSIParams,
    tau: float,
    r: float,
    q: float,
    moneyness: float,
) -> None:
    """Compensator and call price stay finite and positive at the default damping."""
    spot = 100.0
    fwd = spot * np.exp((r - q) * tau)
    strike = moneyness * fwd

    delta = _vgsi_compensator(index_params=idx, factor_loading=beta, residual_params=res)
    assert np.isfinite(delta)

    price = float(
        vgsi_price(
            spot=spot,
            strike=strike,
            tau=tau,
            index_params=idx,
            factor_loading=beta,
            residual_params=res,
            r=r,
            q=q,
            is_call=True,
        )
    )
    lower = max(spot * np.exp(-q * tau) - strike * np.exp(-r * tau), 0.0)
    upper = spot * np.exp(-q * tau)

    assert np.isfinite(price)
    assert price >= lower - 1.0e-7
    assert price <= upper + 1.0e-7


@pytest.mark.slow
@pytest.mark.parametrize(("idx", "beta", "res", "tau", "r", "q", "moneyness"), _all_property_mkt_cases())
def test_property_homogeneity_degree_one(
    idx: VGSIParams,
    beta: float,
    res: VGSIParams,
    tau: float,
    r: float,
    q: float,
    moneyness: float,
) -> None:
    """Price is degree-one homogeneous in (spot, strike)."""
    spot = 100.0
    strike = moneyness * spot * np.exp((r - q) * tau)
    scale = 3.7

    base = float(
        vgsi_price(
            spot=spot,
            strike=strike,
            tau=tau,
            index_params=idx,
            factor_loading=beta,
            residual_params=res,
            r=r,
            q=q,
            is_call=True,
        )
    )
    scaled = float(
        vgsi_price(
            spot=scale * spot,
            strike=scale * strike,
            tau=tau,
            index_params=idx,
            factor_loading=beta,
            residual_params=res,
            r=r,
            q=q,
            is_call=True,
        )
    )
    assert scaled == pytest.approx(scale * base, rel=1.0e-7, abs=1.0e-7)


@pytest.mark.parametrize(("idx", "beta", "res"), _PROPERTY_REGIMES)
@pytest.mark.parametrize("moneyness", _PROPERTY_MONEYNESS)
def test_property_short_time_fourier_matches_intrinsic(
    idx: VGSIParams,
    beta: float,
    res: VGSIParams,
    moneyness: float,
) -> None:
    r"""As $\\tau\\to 0^+$ the call price collapses to its forward intrinsic value."""
    spot = 100.0
    r, q = 0.03, 0.01
    tau = 1.0e-8
    strike = moneyness * spot
    intrinsic = max(spot * np.exp(-q * tau) - strike * np.exp(-r * tau), 0.0)
    price = float(
        _vgsi_call_price(
            spot=spot,
            strike=strike,
            tau=tau,
            index_params=idx,
            factor_loading=beta,
            residual_params=res,
            r=r,
            q=q,
            control="bs",
        )
    )
    assert price == pytest.approx(intrinsic, abs=1.0e-2)


@pytest.mark.parametrize("is_call", [True, False])
def test_exact_zero_time_returns_intrinsic(is_call: bool) -> None:
    spot = 100.0
    strikes = np.array([80.0, 100.0, 120.0])

    prices = vgsi_price(
        spot=spot,
        strike=strikes,
        tau=0.0,
        index_params=VGSIParams(0.2, 0.1, -0.1),
        factor_loading=1.5,
        residual_params=VGSIParams(0.1, 0.2, 0.0),
        is_call=is_call,
    )

    expected_call = np.maximum(spot - strikes, 0.0)
    expected_put = np.maximum(strikes - spot, 0.0)
    assert_allclose(prices, expected_call if is_call else expected_put)


@pytest.mark.parametrize(
    ("sigma_idx", "sigma_res", "beta"),
    [(0.15, 0.10, 0.7), (0.25, 0.20, 1.0), (0.35, 0.05, 1.4)],
)
@pytest.mark.parametrize("tau", [0.1, 0.5, 1.5])
@pytest.mark.parametrize("moneyness", [0.9, 1.0, 1.15])
def test_property_black_scholes_limit(
    sigma_idx: float,
    sigma_res: float,
    beta: float,
    tau: float,
    moneyness: float,
) -> None:
    """With both VG components in their Brownian limit, VGSI reduces to Black-Scholes."""
    idx = VGSIParams(sigma=sigma_idx, nu=0.0, theta=0.0)
    res = VGSIParams(sigma=sigma_res, nu=0.0, theta=0.0)
    spot = 100.0
    r, q = 0.04, 0.01
    fwd = spot * np.exp((r - q) * tau)
    strike = moneyness * fwd
    sigma_eff = float(np.sqrt((beta * sigma_idx) ** 2 + sigma_res**2))

    vgsi_call = float(
        vgsi_price(
            spot=spot,
            strike=strike,
            tau=tau,
            index_params=idx,
            factor_loading=beta,
            residual_params=res,
            r=r,
            q=q,
            is_call=True,
            engine_opt={"adaptive_trunc": True},
        )
    )
    bsm_call = float(
        np.asarray(
            black76_price(fwd=fwd, strike=strike, tau=tau, sigma=sigma_eff, disc=np.exp(-r * tau), is_call=True)
        ).reshape(())
    )
    assert vgsi_call == pytest.approx(bsm_call, abs=1.0e-8)


def test_static_no_arbitrage_in_strike(sony_market: dict) -> None:
    """Call prices are monotone decreasing, convex in $K$, and respect price bounds."""
    strikes = np.linspace(0.5, 1.6, 23) * sony_market["spot"]
    calls = np.asarray(vgsi_price(strike=strikes, is_call=True, **sony_market))

    diff = np.diff(calls)
    assert np.all(diff < 0.0), "call prices must be strictly decreasing in strike"

    second_diff = np.diff(diff)
    assert np.all(second_diff > -1.0e-9), "call prices must be convex in strike"

    disc = float(np.exp(-sony_market["r"] * sony_market["tau"]))
    div_disc = float(np.exp(-sony_market["q"] * sony_market["tau"]))
    upper = sony_market["spot"] * div_disc
    lower = np.maximum(sony_market["spot"] * div_disc - strikes * disc, 0.0)
    assert np.all(calls <= upper + 1.0e-8)
    assert np.all(calls >= lower - 1.0e-8)


@pytest.mark.parametrize("idx_params", _PROPERTY_INDEX_PARAMS[:3])
@pytest.mark.parametrize(("tau1", "tau2"), [(0.1, 0.2), (0.2, 0.5), (0.5, 1.0)])
def test_vg_semigroup_log_cf_additive_in_time(
    idx_params: VGSIParams,
    tau1: float,
    tau2: float,
) -> None:
    r"""A Lévy process satisfies additivity in time for its log characteristic function."""
    us = np.array([0.3, 0.7, 1.5, -0.5, -1.2], dtype=complex)
    phi_sum = _vg_cf(u=us, tau=tau1 + tau2, params=idx_params, scale=1.0)
    phi_1 = _vg_cf(u=us, tau=tau1, params=idx_params, scale=1.0)
    phi_2 = _vg_cf(u=us, tau=tau2, params=idx_params, scale=1.0)
    assert_allclose(np.log(phi_sum), np.log(phi_1) + np.log(phi_2), rtol=1.0e-12, atol=1.0e-12)


def test_vgsi_compensator_rejects_index_explosion() -> None:
    """A factor loading that breaks the unit MGF must fail fast."""
    blowup_index = VGSIParams(sigma=0.50, nu=1.00, theta=1.50)
    with pytest.raises(ValueError, match="unit exponential moment"):
        _vgsi_compensator(
            index_params=blowup_index,
            factor_loading=2.0,
            residual_params=ZERO_PARAMS,
        )


def test_vgsi_rejects_invalid_damping_moment() -> None:
    with pytest.raises(ValueError, match="finite exponential moment"):
        vgsi_price(
            spot=100.0,
            strike=100.0,
            tau=1.0,
            index_params=VGSIParams(sigma=0.5, nu=1.0, theta=0.5),
            factor_loading=1.0,
            residual_params=ZERO_PARAMS,
            engine="quad",
            engine_opt={"damping": 1.0},
        )


def test_jax_jit_pricer_matches_reference_and_jacobian(sony_market: dict) -> None:
    """VGSIJaxParams pytree round-trips through jit and jacfwd via JaxFFTCallEngine."""
    spot = sony_market["spot"]
    tau = sony_market["tau"]
    r = sony_market["r"]
    q = sony_market["q"]
    strikes = jnp.asarray(np.linspace(0.8, 1.2, 9) * spot)
    disc = float(np.exp(-r * tau))
    fft_params = JaxFFTCallEngineParams()
    grid = make_jax_fft_call_grid(strike=np.asarray(strikes), params=fft_params)

    def price(idx: VGSIJaxParams, beta: jax.Array, res: VGSIJaxParams) -> jax.Array:
        cf = make_vgsi_cf_jax(
            spot=spot,
            r=r,
            q=q,
            tau=tau,
            index_params=idx,
            factor_loading=beta,
            residual_params=res,
        )
        return JaxFFTCallEngine(cf=cf, disc=disc, params=fft_params, grid=grid)(strikes)

    price_jit = jax.jit(price)
    jac_jit = jax.jit(jax.jacfwd(price, argnums=1))

    idx = _jax_params_from(sony_market["index_params"])
    res = _jax_params_from(sony_market["residual_params"])
    beta = jnp.asarray(sony_market["factor_loading"])

    reference = np.asarray(
        vgsi_price(strike=np.asarray(strikes), is_call=True, engine="fft_np", control="none", **sony_market),
        dtype=float,
    )
    jitted = np.asarray(price_jit(idx, beta, res), dtype=float)
    assert_allclose(jitted, reference, rtol=5.0e-5, atol=5.0e-5)

    jac_ad = np.asarray(jac_jit(idx, beta, res), dtype=float)
    h = 1.0e-4
    jac_fd = (np.asarray(price(idx, beta + h, res)) - np.asarray(price(idx, beta - h, res))) / (2.0 * h)
    assert_allclose(jac_ad, jac_fd, rtol=1.0e-3, atol=1.0e-5)


def test_jax_residual_pricer_can_be_jitted_and_differentiated(sony_market: dict) -> None:
    """The JAX price helper should be usable inside a residual function."""
    spot = sony_market["spot"]
    tau = sony_market["tau"]
    r = sony_market["r"]
    q = sony_market["q"]
    strikes = jnp.asarray(np.linspace(0.8, 1.2, 7) * spot)
    fft_params = JaxFFTCallEngineParams()
    grid = make_jax_fft_call_grid(strike=np.asarray(strikes), params=fft_params)
    market_prices = jnp.asarray(
        vgsi_price(
            spot=spot,
            strike=np.asarray(strikes),
            tau=tau,
            index_params=sony_market["index_params"],
            factor_loading=sony_market["factor_loading"],
            residual_params=sony_market["residual_params"],
            r=r,
            q=q,
            is_call=True,
            engine="fft_np",
            control="none",
        ),
        dtype=float,
    )

    idx_jax = _jax_params_from(sony_market["index_params"])
    res_jax = _jax_params_from(sony_market["residual_params"])

    def residual(beta: jax.Array) -> jax.Array:
        model_prices = vgsi_price_jax(
            spot=spot,
            strike=strikes,
            tau=tau,
            index_params=idx_jax,
            factor_loading=beta,
            residual_params=res_jax,
            r=r,
            q=q,
            is_call=True,
            control="none",
            grid=grid,
        )
        return model_prices - market_prices

    residual_jit = jax.jit(residual)
    jac_jit = jax.jit(jax.jacfwd(residual))

    beta = jnp.asarray(sony_market["factor_loading"])
    residual_values = np.asarray(residual_jit(beta), dtype=float)
    jacobian = np.asarray(jac_jit(beta), dtype=float)

    assert residual_values.shape == strikes.shape
    assert jacobian.shape == strikes.shape

    h = 1.0e-4
    fd = (np.asarray(residual(beta + h), dtype=float) - np.asarray(residual(beta - h), dtype=float)) / (2.0 * h)
    assert_allclose(jacobian, fd, rtol=1.0e-3, atol=1.0e-5)
