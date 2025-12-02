import ctypes as ct
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from vol_risk.models.linear import LinearEquityMarket, make_simple_linear_market

logger = logging.getLogger(__name__)

# Import the DLL from _lib directory
DLL_PATH = pathlib.Path(__file__).parent.parent / "_lib" / "heston.dll"
lib = ct.CDLL(str(DLL_PATH))

# Define function signatures
lib.hestonPricer.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # S_adj
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # r
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # K
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # mat
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # par
    ct.POINTER(ct.c_int),  # n
    ct.POINTER(ct.c_int),  # m
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # p
]

lib.hestonDelta.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # S_adj
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # r
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # q
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # K
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # mat
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # par
    ct.POINTER(ct.c_int),  # n
    ct.POINTER(ct.c_int),  # m
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # delta
]

lib.hestonJac.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # S_adj
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # r
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # K
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # mat
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # par
    ct.POINTER(ct.c_int),  # n
    ct.POINTER(ct.c_int),  # m
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # jac
]

lib.hestonCalibrator.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # S_adj
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # r
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # K
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # mat
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # price
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # par
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),  # statistics
    ct.POINTER(ct.c_int),  # n
    ct.POINTER(ct.c_int),  # m
    ct.POINTER(ct.c_bool),  # print
]

# All exports return void
lib.hestonPricer.restype = None
lib.hestonDelta.restype = None
lib.hestonJac.restype = None
lib.hestonCalibrator.restype = None

levmar_stop_reason = {
    1: "Solved: Stopped by small gradient.",
    2: "Solved: Stopped by small parameter increment.",
    3: "Unsolved: stopped by itmax.",
    4: "Unsolved: singular matrix.",
    5: "Unsolved: no further error reduction is possible.",
    6: "Solved: Stopped by small error.",
    7: "Unsolved: stopped by invalid values, user error.",
}


def as_1d_c_array(x: ArrayLike, size: int | None = None) -> np.ndarray:
    """Convert input to a contiguous 1D numpy array of float64."""
    # See https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tools/validation/validation.py

    arr = np.asarray(x)

    # Broadcast to shape (size,)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if size and arr.size == 1:
        arr = np.broadcast_to(arr, (size,))

    # Verify ndim and size
    if arr.ndim > 1:
        msg = "Input arrays must be 1D or broadcastable to 1D"
        raise ValueError(msg)
    if size and arr.size != size:
        msg = f"Input array has size {arr.size}, expected {size}"
        raise ValueError(msg)

    return np.require(arr, dtype=np.float64, requirements=["C", "A", "O", "W"])


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Heston model parameters."""

    kappa: float
    v_inf: float
    vol_of_vol: float
    rho: float
    v0: float

    def __post_init__(self):
        if 2 * self.kappa * self.v_inf <= self.vol_of_vol**2:
            msg = "Feller condition violated."
            logger.warning(msg)


def heston_price_cui(
    params: HestonParams,
    el: LinearEquityMarket,
    K: ArrayLike,
    T: ArrayLike,
) -> ArrayLike:
    """Heston price via Cui et al. implemented in C++ DLL.

    DLL expects adjusted spot S_adj = S * exp(-q*T).
    """
    bK, bT = np.broadcast_arrays(K, T)
    n = bK.size

    # Get EQ Linear market data
    r = el.zero_rate(bT)
    disc_fwd = el.fwd(bT) * el.disc_curve(bT)

    # Convert to contiguous C arrays
    bK, bT, r, disc_fwd = [as_1d_c_array(x, size=n) for x in [bK, bT, r, disc_fwd]]

    params_arr = as_1d_c_array([params.kappa, params.v_inf, params.vol_of_vol, params.rho, params.v0])
    out = np.empty(shape=(n), dtype=np.float64)

    n_c = ct.c_int(n)
    m_c = ct.c_int(5)

    lib.hestonPricer(disc_fwd, r, bK, bT, params_arr, ct.byref(n_c), ct.byref(m_c), out)

    if bK.size == 1:
        return out[0]

    return out


def heston_delta(
    params: HestonParams,
    el: LinearEquityMarket,
    K: ArrayLike,
    T: ArrayLike,
) -> ArrayLike:
    """Call option delta under the Heston model via Cui et al. DLL."""
    bK, bT = np.broadcast_arrays(K, T)
    n = bK.size

    r = el.zero_rate(bT)
    disc_fwd = el.fwd(bT) * el.disc_curve(bT)
    q = el.cont_dvd_curve(bT)

    disc_fwd, r, q, bK, bT = [as_1d_c_array(x, size=n) for x in [disc_fwd, r, q, bK, bT]]

    params_arr = as_1d_c_array([params.kappa, params.v_inf, params.vol_of_vol, params.rho, params.v0])
    out = np.empty(shape=(n), dtype=np.float64)

    n_c = ct.c_int(n)
    m_c = ct.c_int(5)

    lib.hestonDelta(disc_fwd, r, q, bK, bT, params_arr, ct.byref(n_c), ct.byref(m_c), out)

    if bK.size == 1:
        return out[0]

    return out


def heston_jacobian(
    params: HestonParams,
    el: LinearEquityMarket,
    K: ArrayLike,
    T: ArrayLike,
) -> ArrayLike:
    """Jacobian of Heston prices with respect to model parameters."""
    bK, bT = np.broadcast_arrays(K, T)
    n = bK.size

    r = el.zero_rate(bT)
    disc_fwd = el.fwd(bT) * el.disc_curve(bT)

    disc_fwd, r, bK, bT = [as_1d_c_array(x, size=n) for x in [disc_fwd, r, bK, bT]]

    params_arr = as_1d_c_array([params.kappa, params.v_inf, params.vol_of_vol, params.rho, params.v0])
    out = np.empty(shape=(n * 5), dtype=np.float64)

    n_c = ct.c_int(n)
    m_c = ct.c_int(5)

    lib.hestonJac(disc_fwd, r, bK, bT, params_arr, ct.byref(n_c), ct.byref(m_c), out)

    jac = out.reshape(n, 5)

    if bK.size == 1:
        return jac[0]

    return jac


def heston_calibrator(
    x0: HestonParams,
    el: LinearEquityMarket,
    p: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
) -> tuple[HestonParams, dict]:
    """Calibrate Heston params via Cui et al. implemented in C++ DLL.

    Args:
        x0: Initial guess for Heston parameters.
        el: Linear equity market with spot, rates, and dividend yield.
        p: Observed option prices.
        K: Strike prices.
        T: Time to maturity (in years).
        print_result: Whether to print calibration statistics.

    Returns:
        Tuple of (calibrated HestonParams, statistics dict).
    """
    bK, bT, bp = np.broadcast_arrays(K, T, p)
    n = bK.size

    r = el.zero_rate(bT)
    disc_fwd = el.fwd(bT) * el.disc_curve(bT)

    # Convert to contiguous C arrays
    bK, bT, bp, r, disc_fwd = [as_1d_c_array(x, size=n) for x in [bK, bT, bp, r, disc_fwd]]

    # Mutable C buffers
    params_arr = as_1d_c_array([x0.kappa, x0.v_inf, x0.vol_of_vol, x0.rho, x0.v0])
    stats = as_1d_c_array(np.empty(shape=(7), dtype=np.float64))

    n_c = ct.c_int(n)
    m_c = ct.c_int(5)

    lib.hestonCalibrator(disc_fwd, r, bK, bT, bp, params_arr, stats, ct.byref(n_c), ct.byref(m_c), ct.c_bool(False))

    # Extract calibrated parameters
    par_opt = HestonParams(
        kappa=params_arr[0],
        v_inf=params_arr[1],
        vol_of_vol=params_arr[2],
        rho=params_arr[3],
        v0=params_arr[4],
    )

    if int(stats[6]) not in [1, 2, 6]:
        msg = (
            f"Heston calibration did not converge. LEVMAR retur code = {int(stats[6])} "
            f"({levmar_stop_reason.get(int(stats[6]), 'Unknown stop reason.')})"
        )
        logger.warning(msg)

    # Build statistics dict
    stats = {
        "error_l2_initial": stats[0],
        "error_l2_final": stats[1],
        "jt_e_inf_norm": stats[2],
        "dp_inf_norm": stats[3],
        "cpu_time_sec": stats[4],
        "iterations": int(stats[5]),
        "stop_code": int(stats[6]),
        "stop_reason": levmar_stop_reason.get(int(stats[6]), "Unknown stop reason."),
    }

    return par_opt, stats


if __name__ == "__main__":
    import time

    start_time = time.time()

    # base market data
    S = 100.0
    T = 1.0
    r = 0.02
    q = 0.00

    el_market = make_simple_linear_market(s=S, r=r, q=q)

    # True Heston params used to generate prices
    par_true = HestonParams(1, 0.09, 1, -0.3, 0.09)

    # Build a grid of strikes around ATM
    n_obs = 100
    strikes = np.linspace(80.0, 120.0, n_obs)

    # Generate prices using the native pricer
    prices = np.array([heston_price_cui(par_true, el_market, K, T) for K in strikes])

    # Initial guess for calibration
    par_init = HestonParams(1.05, 0.08, 0.95, -0.35, 0.15)

    # Calibrate
    par_opt, stats = heston_calibrator(par_init, el_market, prices, strikes, T)

    end_time = time.time()

    print(f"Calibration took {end_time - start_time:.4f} seconds")
    print("True params:      ", par_true)
    print("Initial guess:    ", par_init)
    print("Calibrated params:", par_opt)
    print("Calibration stats:", stats)
