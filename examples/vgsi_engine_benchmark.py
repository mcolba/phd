"""Benchmark VGSI option pricing across the available Fourier engines."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

ENABLE_JAX_X64 = True
jax.config.update("jax_enable_x64", ENABLE_JAX_X64)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vol_risk.models.numerical.fourier.fft_jax import (  # noqa: E402
    JaxFFTCallEngineParams,
    JaxFFTCallGrid,
    make_jax_fft_call_grid,
)
from vol_risk.models.vgsi import VGSIParams, vgsi_price  # noqa: E402
from vol_risk.models.vgsi_jax import VGSIJaxParams, vgsi_price_jax  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

N_SIM = 100
N_K = 10
N_T = 10

SPOT = 100.0
R = 0.05
Q = 0.02
FACTOR_LOADING = 0.1931

INDEX_SIGMA = 0.2658
INDEX_NU = 0.0505
INDEX_THETA = -0.3855

RESIDUAL_SIGMA = 0.3031
RESIDUAL_NU = 0.001
RESIDUAL_THETA = 0.0

LOG_FWD_MONEYNESS = np.linspace(np.log(0.7), np.log(1.4), N_K)
MATURITIES = np.linspace(1.0 / 12.0, 1.0, N_T)

QUAD_ENGINE_OPT = {
    "damping": 0.03,
    "upper_bound": 524.0,
    "breakpoints": (8.0, 64.0),
    "epsabs": 1.0e-6,
    "epsrel": 1.0e-6,
    "limit": 100,
}
FFT_ENGINE_OPT = {}
ENGINES = (
    ("base.py quadrature", "quad", QUAD_ENGINE_OPT),
    ("fft_np", "fft_np", FFT_ENGINE_OPT),
)


def flatten_fft_grid(grid: JaxFFTCallGrid) -> tuple[tuple[jax.Array, ...], tuple[float, float, int, float, float]]:
    """Return JAX pytree leaves and static metadata for an FFT grid."""
    children = (
        grid.frequency,
        grid.shifted_frequency,
        grid.log_strike,
        grid.strike,
        grid.weights,
        grid.phase,
        grid.denominator,
    )
    aux = (
        grid.damping,
        grid.log_strike_step,
        grid.grid_size,
        grid.log_strike_half_width,
        grid.frequency_step,
    )
    return children, aux


def unflatten_fft_grid(aux: tuple[float, float, int, float, float], children: tuple[jax.Array, ...]) -> JaxFFTCallGrid:
    """Return an FFT grid from JAX pytree leaves and static metadata."""
    damping, log_strike_step, grid_size, log_strike_half_width, frequency_step = aux
    frequency, shifted_frequency, log_strike, strike, weights, phase, denominator = children
    return JaxFFTCallGrid(
        damping=damping,
        log_strike_step=log_strike_step,
        grid_size=grid_size,
        log_strike_half_width=log_strike_half_width,
        frequency_step=frequency_step,
        frequency=frequency,
        shifted_frequency=shifted_frequency,
        log_strike=log_strike,
        strike=strike,
        weights=weights,
        phase=phase,
        denominator=denominator,
    )


jax.tree_util.register_pytree_node(JaxFFTCallGrid, flatten_fft_grid, unflatten_fft_grid)


def make_params(bump: float) -> tuple[VGSIParams, float, VGSIParams]:
    """Return perturbed VGSI parameters for the public pricing API."""
    index_params = VGSIParams(
        sigma=INDEX_SIGMA * (1.0 + 0.0015 * bump),
        nu=INDEX_NU * (1.0 + 0.0010 * bump),
        theta=INDEX_THETA * (1.0 - 0.0008 * bump),
    )
    residual_params = VGSIParams(
        sigma=RESIDUAL_SIGMA * (1.0 - 0.0010 * bump),
        nu=RESIDUAL_NU * (1.0 + 0.0012 * bump),
        theta=RESIDUAL_THETA,
    )
    factor_loading = FACTOR_LOADING * (1.0 + 0.0005 * bump)
    return index_params, factor_loading, residual_params


def make_jax_params(bump: jax.Array) -> tuple[VGSIJaxParams, jax.Array, VGSIJaxParams]:
    """Return perturbed VGSI parameters for the JAX pricing API."""
    index_params = VGSIJaxParams(
        sigma=INDEX_SIGMA * (1.0 + 0.0015 * bump),
        nu=INDEX_NU * (1.0 + 0.0010 * bump),
        theta=INDEX_THETA * (1.0 - 0.0008 * bump),
    )
    residual_params = VGSIJaxParams(
        sigma=RESIDUAL_SIGMA * (1.0 - 0.0010 * bump),
        nu=RESIDUAL_NU * (1.0 + 0.0012 * bump),
        theta=jnp.asarray(RESIDUAL_THETA),
    )
    factor_loading = FACTOR_LOADING * (1.0 + 0.0005 * bump)
    return index_params, factor_loading, residual_params


def fwd_for_tau(tau: float) -> float:
    """Return the forward price for one maturity."""
    return SPOT * float(np.exp((R - Q) * tau))


def strikes_for_tau(tau: float) -> np.ndarray:
    """Return strikes from the constant log-forward-moneyness grid."""
    return fwd_for_tau(tau) * np.exp(LOG_FWD_MONEYNESS)


def grid_for_strikes(strikes: np.ndarray | jax.Array) -> JaxFFTCallGrid:
    """Return the JAX FFT grid for one fixed-shape strike array."""
    return make_jax_fft_call_grid(
        strike=strikes,
        params=JaxFFTCallEngineParams(**FFT_ENGINE_OPT),
    )


def price_public_engine(engine: str, engine_opt: dict[str, object]) -> float:
    """Return checksum for one public VGSI engine."""
    checksum = 0.0
    for sim_idx in range(N_SIM):
        index_params, factor_loading, residual_params = make_params(float(sim_idx + 1))
        for tau in MATURITIES:
            tau_value = float(tau)
            prices = np.asarray(
                vgsi_price(
                    spot=SPOT,
                    strike=strikes_for_tau(tau_value),
                    tau=tau_value,
                    index_params=index_params,
                    factor_loading=factor_loading,
                    residual_params=residual_params,
                    r=R,
                    q=Q,
                    is_call=True,
                    engine=engine,
                    engine_opt=engine_opt,
                    control="bs",
                ),
                dtype=float,
            )
            checksum += float(np.sum(prices))
    return checksum


def price_base_quad() -> float:
    """Return checksum for the base quadrature engine."""
    return price_public_engine("quad", QUAD_ENGINE_OPT)


def price_fft_np() -> float:
    """Return checksum for the NumPy FFT engine."""
    return price_public_engine("fft_np", FFT_ENGINE_OPT)


def make_jitted_pricer(tau: float) -> Callable[[VGSIJaxParams, jax.Array, VGSIJaxParams], jax.Array]:
    """Return a JIT-compiled VGSI pricer for one maturity."""
    strikes_jax = jnp.asarray(strikes_for_tau(tau))
    fft_grid = grid_for_strikes(strikes_jax)

    @jax.jit
    def pricer(
        index_params: VGSIJaxParams,
        factor_loading: jax.Array,
        residual_params: VGSIJaxParams,
    ) -> jax.Array:
        return vgsi_price_jax(
            spot=SPOT,
            strike=strikes_jax,
            tau=tau,
            index_params=index_params,
            factor_loading=factor_loading,
            residual_params=residual_params,
            r=R,
            q=Q,
            is_call=True,
            control="bs",
            grid=fft_grid,
        )

    return pricer


def make_jitted_pricer_surface() -> Callable[
    [float, jax.Array, JaxFFTCallGrid, VGSIJaxParams, jax.Array, VGSIJaxParams],
    jax.Array,
]:
    """Return a JIT-compiled VGSI pricer with maturity inputs passed in."""

    def pricer(
        tau: float,
        strikes: jax.Array,
        grid: JaxFFTCallGrid,
        index_params: VGSIJaxParams,
        factor_loading: jax.Array,
        residual_params: VGSIJaxParams,
    ) -> jax.Array:
        return vgsi_price_jax(
            spot=SPOT,
            strike=strikes,
            tau=tau,
            index_params=index_params,
            factor_loading=factor_loading,
            residual_params=residual_params,
            r=R,
            q=Q,
            is_call=True,
            control="bs",
            grid=grid,
        )

    return jax.jit(pricer)


jitted_pricers = tuple(make_jitted_pricer(float(tau)) for tau in MATURITIES)
jitted_surface_pricer = make_jitted_pricer_surface()
surface_inputs = tuple(
    (
        float(tau),
        jnp.asarray(strikes_for_tau(float(tau))),
        grid_for_strikes(strikes_for_tau(float(tau))),
    )
    for tau in MATURITIES
)


def price_fft_jax_no_jit() -> float:
    """Return checksum for the eager JAX FFT engine."""
    checksum = 0.0
    for sim_idx in range(N_SIM):
        bump = jnp.asarray(sim_idx + 1, dtype=float)
        index_params, factor_loading, residual_params = make_jax_params(bump)
        for tau, strikes, grid in surface_inputs:
            prices = vgsi_price_jax(
                spot=SPOT,
                strike=strikes,
                tau=tau,
                index_params=index_params,
                factor_loading=factor_loading,
                residual_params=residual_params,
                r=R,
                q=Q,
                is_call=True,
                control="bs",
                grid=grid,
            )
            prices.block_until_ready()
            checksum += float(jnp.sum(prices))
    return checksum


def price_fft_jax_jit() -> float:
    """Return checksum for the JIT-compiled JAX FFT engine."""
    checksum = 0.0
    for sim_idx in range(N_SIM):
        bump = jnp.asarray(sim_idx + 1, dtype=float)
        index_params, factor_loading, residual_params = make_jax_params(bump)
        for pricer in jitted_pricers:
            prices = pricer(index_params, factor_loading, residual_params)
            prices.block_until_ready()
            checksum += float(jnp.sum(prices))
    return checksum


def price_fft_jax_jit_surface() -> float:
    """Return checksum for the JIT-compiled JAX FFT surface pricer."""
    checksum = 0.0
    for sim_idx in range(N_SIM):
        bump = jnp.asarray(sim_idx + 1, dtype=float)
        index_params, factor_loading, residual_params = make_jax_params(bump)
        for tau, strikes, grid in surface_inputs:
            prices = jitted_surface_pricer(tau, strikes, grid, index_params, factor_loading, residual_params)
            prices.block_until_ready()
            checksum += float(jnp.sum(prices))
    return checksum


def print_timing(label: str, work: Callable[[], float]) -> None:
    """Print elapsed time and checksum for one benchmark work unit."""
    start = perf_counter()
    checksum = work()
    elapsed = perf_counter() - start
    sys.stdout.write(f"{label:>18}: {elapsed:10.4f} s  checksum={checksum:.6f}\n")


prices_per_sim = N_K * N_T
total_prices = prices_per_sim * N_SIM
sys.stdout.write(f"VGSI pricing benchmark: {N_K} strikes x {N_T} maturities\n")
sys.stdout.write(f"{prices_per_sim:,} prices per simulation; {N_SIM} simulations; {total_prices:,} prices per engine\n")

print_timing("base.py quadrature", price_base_quad)
print_timing("fft_np", price_fft_np)
print_timing("fft_jax no jit", price_fft_jax_no_jit)
print_timing("fft_jax jit first", price_fft_jax_jit)
print_timing("fft_jax jit cached", price_fft_jax_jit)
print_timing("jit surface first", price_fft_jax_jit_surface)
print_timing("jit surface cached", price_fft_jax_jit_surface)
