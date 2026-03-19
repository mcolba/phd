import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arbitragerepair import constraints
from arbitragerepair.repair import matrix, solvers
from scipy.interpolate import RBFInterpolator

from vol_risk.calibration.option_chain import OptionChain, OptionSlice
from vol_risk.models.black76 import black76_price, implied_vol_jackel
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.vol_surface.moneyness import DeltaMoneyness, Moneyness

log = logging.getLogger(__name__)


def _as_float_array(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _as_scalar(x: np.ndarray | float) -> float:
    return float(_as_float_array(x)[0])


def _require_call_only(chain: OptionChain) -> None:
    if not np.all(chain.option_type == "C"):
        msg = "Synthetic quote augmentation expects a call-only chain. Use make_otm_to_call first."
        raise ValueError(msg)


def _clip_call_price(price: float, discount: float, forward: float, strike: float) -> float:
    lower = max(discount * (forward - strike), 0.0)
    upper = discount * forward
    epsilon = 1e-10 * max(1.0, upper)
    return float(np.clip(price, lower + epsilon, upper - epsilon))


def _slice_total_variance(
    option_slice: OptionSlice,
    market: LinearEquityMarket,
    min_total_variance: float,
) -> tuple[np.ndarray, float, float]:
    tau = option_slice.slice_tau
    if tau <= 0.0:
        msg = "Synthetic quote augmentation requires strictly positive maturities."
        raise ValueError(msg)

    discount = _as_scalar(market.df(tau))
    forward = _as_scalar(market.fwd(tau))
    total_variance = []

    for strike, price in zip(option_slice.k, option_slice.mid, strict=True):
        clean_price = _clip_call_price(float(price), discount, forward, float(strike))
        sigma = implied_vol_jackel(
            price=clean_price,
            f=forward,
            k=float(strike),
            t=tau,
            df=discount,
            is_call=True,
        )
        total_variance.append(max(float(sigma) ** 2 * tau, min_total_variance))

    return np.asarray(total_variance, dtype=float), forward, discount


def _make_quote_grid(k_min: float, k_max: float, grid_size: int) -> np.ndarray:
    return np.linspace(k_min, k_max, grid_size, dtype=float)


def _build_rbf_surface(
    chain: OptionChain,
    market: LinearEquityMarket,
    spline_smoothing: float,
    min_total_variance: float,
) -> tuple[RBFInterpolator, np.ndarray, np.ndarray]:
    point_blocks = []
    value_blocks = []

    for _, option_slice in chain:
        total_variance, forward, _ = _slice_total_variance(
            option_slice=option_slice,
            market=market,
            min_total_variance=min_total_variance,
        )
        log_fwd_moneyness = np.log(option_slice.k / forward)
        tau_coords = np.full(log_fwd_moneyness.shape, option_slice.slice_tau, dtype=float)
        point_blocks.append(np.column_stack((tau_coords, log_fwd_moneyness)))
        value_blocks.append(total_variance)

    if not point_blocks:
        msg = "Cannot build a 2D total-variance spline from an empty chain."
        raise ValueError(msg)

    points = np.vstack(point_blocks)
    values = np.concatenate(value_blocks)
    center = points.mean(axis=0)
    scale = points.std(axis=0)
    scale[scale == 0.0] = 1.0

    spline = RBFInterpolator(
        (points - center) / scale,
        values,
        kernel="thin_plate_spline",
        smoothing=spline_smoothing,
    )
    return spline, center, scale


def _evaluate_total_variance_surface(
    spline: RBFInterpolator,
    center: np.ndarray,
    scale: np.ndarray,
    tau: np.ndarray | float,
    log_fwd_moneyness: np.ndarray | float,
) -> np.ndarray:
    tau_arr, log_k_arr = np.broadcast_arrays(
        np.asarray(tau, dtype=float),
        np.asarray(log_fwd_moneyness, dtype=float),
    )
    coords = np.column_stack((tau_arr.reshape(-1), log_k_arr.reshape(-1)))
    scaled_coords = (coords - center) / scale
    return np.asarray(spline(scaled_coords), dtype=float).reshape(tau_arr.shape)


# def _normalise_quotes_for_repair(
#     tau: np.ndarray,
#     strike: np.ndarray,
#     undiscounted_call: np.ndarray,
#     forward: np.ndarray,
#     min_price: float | None,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     df = pd.DataFrame(
#         {
#             "idx": np.arange(tau.size, dtype=int),
#             "tau": tau,
#             "strike": strike,
#             "price": undiscounted_call,
#             "forward": forward,
#         }
#     ).sort_values(["tau", "strike"], kind="stable")

#     mask = ~df["price"].isna()
#     if min_price is not None:
#         mask &= df["price"] >= min_price
#     df = df.loc[mask].copy()

#     idx = df["idx"].to_numpy(dtype=int)
#     tau_norm = df["tau"].to_numpy(dtype=float)
#     strike_norm = (df["strike"] / df["forward"]).to_numpy(dtype=float)
#     price_norm = (df["price"] / df["forward"]).to_numpy(dtype=float)
#     forward_ord = df["forward"].to_numpy(dtype=float)
#     return tau_norm, strike_norm, price_norm, idx, forward_ord


def _solve_weighted_l1_repair(
    mat_a: np.ndarray,
    vec_b: np.ndarray,
    price: np.ndarray,
    weights: np.ndarray,
    solver: str = "glpk",
) -> np.ndarray:
    """Weighted version of the l1() arbitrage repair in arbitragerepair.repair."""
    n_quote = mat_a.shape[1]
    MAX_ATTEMPTS = 1
    sol = []

    if price.shape[0] != n_quote or weights.shape[0] != n_quote:
        msg = "mat_a, price, and weights must agree on the number of quotes."
        raise ValueError(msg)
    if np.any(weights <= 0.0):
        msg = "weights must be strictly positive."
        raise ValueError(msg)

    A = -np.hstack((mat_a, -mat_a))
    b = -(vec_b - mat_a.dot(price))
    coeff = np.hstack((weights, weights))

    A1 = np.vstack((A, -np.diag(np.ones(2 * n_quote))))
    b1 = np.hstack((b, np.zeros(2 * n_quote)))

    G = matrix(A1)
    h = matrix(b1)
    c = matrix(coeff)

    """
    Scale the constraint for numerical stability
    A * (scale * epsilon) >= scale * b
    """
    G *= 2.0
    h *= 2.0

    i_attempt = 1
    scale = 0.1
    status = "initial"
    while status != "optimal":
        scale *= 10
        G *= scale
        h *= scale

        # solve the LP
        sol = solvers.lp(c, G, h, solver=solver)
        status = sol["status"]

        i_attempt += 1
        if i_attempt > MAX_ATTEMPTS:
            break

    if status == "optimal":
        x = np.array(sol["x"])
        epsilon = x[:n_quote] - x[n_quote:]
        epsilon = epsilon.flatten()
        epsilon /= scale
    else:
        epsilon = []
        log.warning("Optimal perturbation is not found.")

    return epsilon


def repair_arbitrage(
    chain: OptionChain,
    market: LinearEquityMarket,
    synthetic_weight: float = 1.0,
    min_price: float | None = None,
    tolerance: float = 0.0,
    solver: str = "glpk",
) -> OptionChain:
    """Repair static arbitrage on a call-only chain with heavier synthetic-quote penalties."""
    _require_call_only(chain)

    df = chain.df.copy()
    if "synthetic" not in df.columns:
        df["synthetic"] = False

    if "repair_adj" not in df.columns:
        df["repair_adj"] = 0.0

    df["repair_weight"] = np.where(df["synthetic"], synthetic_weight, 1.0)

    tau = _as_float_array(chain.tau)
    strike = _as_float_array(chain.k)
    mid = _as_float_array(chain.mid)
    disc = _as_float_array(market.df(tau))
    forward = _as_float_array(market.fwd(tau))
    undisc_mid = mid / disc

    normaliser = constraints.Normalise(min_price=min_price)
    normaliser.fit(T=tau, K=strike, C=undisc_mid, F=forward)
    T1, K1, C1 = normaliser.transform(T=tau, K=strike, C=undisc_mid)
    mat_A, vec_b, _, _ = constraints.detect(T=T1, K=K1, C=C1, tolerance=tolerance, verbose=False)

    epsilon = _solve_weighted_l1_repair(
        mat_a=mat_A,
        vec_b=vec_b,
        price=C1,
        weights=df["repair_weight"].to_numpy(dtype=float),
        solver=solver,
    )

    if len(epsilon) == 0:
        log.warning("No repair applied to the chain.")
        return chain

    # epsilon2 = repair.l1(mat_A, vec_b, C1)
    # assert np.allclose(epsilon, epsilon2, atol=1e-10)

    _, C0 = normaliser.inverse_transform(K=K1, C=C1 + epsilon)
    df.loc[:, "mid"] = C0 * disc
    df.loc[:, "repair_adj"] = (C0 - undisc_mid) * disc

    return chain.__class__(df, chain._calendar)


def append_synthetic_quotes(
    chain: OptionChain,
    market: LinearEquityMarket,
    k_min: float,
    k_max: float,
    grid_size: int = 41,
    spline_smoothing: float = 0.0,
    min_obs_per_slice: int = 3,
    min_total_variance: float = 1e-8,
    synthetic_weight: float = 10.0,
) -> OptionChain:
    """Append synthetic tail quotes from a global 2D total-variance surface."""
    _require_call_only(chain)

    df = chain.df.copy()
    df["synthetic"] = df.get("synthetic", False)
    df["synthetic"] = df["synthetic"].fillna(value=False).astype(bool)
    df["repair_adj"] = df.get("repair_adj", 0.0)
    df["repair_adj"] = df["repair_adj"].fillna(0.0).astype(float)
    df["repair_weight"] = np.where(df["synthetic"], synthetic_weight, 1.0)

    target_grid = _make_quote_grid(k_min, k_max, grid_size)
    spline, center, scale = _build_rbf_surface(
        chain=chain,
        market=market,
        spline_smoothing=spline_smoothing,
        min_total_variance=min_total_variance,
    )
    synthetic_rows = []

    for _, option_slice in chain:
        if len(option_slice) < min_obs_per_slice:
            continue

        _, forward, discount = _slice_total_variance(
            option_slice=option_slice,
            market=market,
            min_total_variance=min_total_variance,
        )
        observed_k = np.log(option_slice.k / forward)
        missing_grid = target_grid[(target_grid < observed_k.min()) | (target_grid > observed_k.max())]
        if missing_grid.size == 0:
            continue

        synth_total_variance = np.maximum(
            _evaluate_total_variance_surface(
                spline=spline,
                center=center,
                scale=scale,
                tau=np.full(missing_grid.shape, option_slice.slice_tau, dtype=float),
                log_fwd_moneyness=missing_grid,
            ).reshape(-1),
            min_total_variance,
        )
        synth_sigma = np.sqrt(synth_total_variance / option_slice.slice_tau)
        synth_strikes = forward * np.exp(missing_grid)
        synth_mid = _as_float_array(
            black76_price(
                df=discount,
                f=forward,
                k=synth_strikes,
                t=option_slice.slice_tau,
                sigma=synth_sigma,
                is_call=np.ones_like(synth_strikes, dtype=bool),
            )
        )

        template = option_slice.df.iloc[0].copy()
        for strike_i, mid_i, k_i, total_var_i in zip(
            synth_strikes,
            synth_mid,
            missing_grid,
            synth_total_variance,
            strict=True,
        ):
            row = template.copy()
            row["strike"] = float(strike_i)
            row["mid"] = float(mid_i)
            row["bid"] = np.nan
            row["ask"] = np.nan
            row["option_type"] = "C"
            row["volume"] = 0
            if "open_interest" in row.index:
                row["open_interest"] = 0
            row["synthetic"] = True
            row["repair_adj"] = 0.0
            row["repair_weight"] = synthetic_weight
            row["synthetic_source"] = "thin_plate_total_variance_2d"
            row["log_fwd_moneyness"] = float(k_i)
            row["total_variance"] = float(total_var_i)
            synthetic_rows.append(row)

    if synthetic_rows:
        df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)

    return chain.__class__(df, chain._calendar)


def liquidity_filter(
    chain: OptionChain,
    oi_min: None | int = None,
    bid_min: None | float = None,
    mid_min: None | float = None,
    rel_bid_ask_max: None | float = None,
    min_ttm: None | int = None,
    min_k_per_slice: int = 3,
) -> OptionChain:
    """Filter an option chain for liquid contracts."""
    df = chain.df.copy()

    # Liquidity filters
    mask = pd.Series(data=True, index=df.index)
    if oi_min is not None:
        mask &= (df["open_interest"].notna()) & (df["open_interest"] >= oi_min)
    if bid_min is not None:
        mask &= (df["bid"].notna()) & (df["bid"] >= bid_min)
    if mid_min is not None:
        mask &= (df["mid"].notna()) & (df["mid"] >= mid_min)
    if rel_bid_ask_max is not None:
        if (df["mid"].isna().any()) or (df["mid"] <= 0).any():
            msg = "Relative bid-ask spread filtering requires mid prices to be positive."
            raise ValueError(msg)
        mask &= ((df["ask"] - df["bid"]) / df["mid"]) <= rel_bid_ask_max
    if min_ttm is not None:
        anchor_days = df["anchor"].to_numpy(dtype="datetime64[D]")
        expiry_days = df["expiry"].to_numpy(dtype="datetime64[D]")
        mask &= np.busday_count(anchor_days, expiry_days) >= min_ttm

    df = df.loc[mask, :]

    # Ensure each slice has at least min_k_per_slice strikes
    slice_mask = pd.Series(data=True, index=df.index)
    if min_k_per_slice > 1:
        slice_mask &= df.groupby("expiry")["strike"].transform("nunique") >= min_k_per_slice

    return chain.__class__(df.loc[slice_mask].copy(), chain._calendar)


def make_otm_to_call(chain: OptionChain, le: LinearEquityMarket) -> OptionChain:
    """Create a call-only view of an option chain."""
    df = chain.df.copy()
    tau = chain.tau

    is_otm_c = (df["option_type"] == "C") & (df["strike"] >= le.fwd(tau))
    is_otm_p = (df["option_type"] == "P") & (df["strike"] <= le.fwd(tau))

    calls = df.loc[is_otm_c, :]

    # OTM puts to ITM calls using put-call parity
    puts = df.loc[is_otm_p, :]
    p_mid = chain.mid[is_otm_p]
    p_tau = chain.tau[is_otm_p]
    p_fwd_contract = le.df(p_tau) * (le.fwd(p_tau) - puts["strike"])

    puts = puts.assign(
        option_type="C",
        bid=np.nan,
        ask=np.nan,
        mid=p_fwd_contract + p_mid,
    )

    return chain.__class__(pd.concat([calls, puts], ignore_index=True), chain._calendar)


def get_atmf_vol(chain: OptionSlice, le: LinearEquityMarket) -> float:
    """Get ATMF volatilities from an option chain."""
    otm_chain = make_otm_to_call(chain, le)
    tau = otm_chain.slice_tau
    fwd = le.fwd(tau)

    idx_sort = np.argsort(otm_chain.k)
    strike = otm_chain.k[idx_sort]
    price = otm_chain.mid[idx_sort]

    k = 3
    idx_closest = np.searchsorted(strike, fwd)
    idx_first = max(idx_closest - k, 0)
    idx_last = min(idx_closest + k + 1, len(strike))
    mask = list(range(idx_first, idx_last))

    deg = 2 if len(mask) > 2 else 1
    z = np.polyfit(strike[mask], price[mask], deg)
    atm_price = np.poly1d(z)(fwd)

    vol = implied_vol_jackel(
        price=atm_price,
        f=fwd,
        k=fwd,
        t=tau,
        df=le.df(tau),
        is_call=True,
    )

    return vol


def apply_cutoffs(
    chain: OptionChain,
    moneyness: Moneyness,
    bounds: tuple[float, float] = (-np.inf, np.inf),
) -> OptionChain:
    """Apply cutoffs to an option chain."""
    if bounds[0] >= bounds[1]:
        msg = f"Lower bound {bounds[0]} must be less than upper bound {bounds[1]}."
        raise ValueError(msg)

    if isinstance(moneyness, DeltaMoneyness):
        ub, lb = bounds
    else:
        lb, ub = bounds

    filtered_frames = []

    for _, sl in chain:
        strikes = sl.k
        tau = sl.slice_tau
        mask = np.ones_like(strikes, dtype=bool)
        sigma = get_atmf_vol(sl, moneyness.le)

        if (lb is not None) and not np.isneginf(lb):
            strike_lb = moneyness.invert(moneyness=lb, tau=tau, sigma=sigma)
            mask &= strikes >= strike_lb
        if (ub is not None) and not np.isposinf(ub):
            strike_ub = moneyness.invert(moneyness=ub, tau=tau, sigma=sigma)
            mask &= strikes <= strike_ub

        filtered_frames.append(sl.df.loc[mask].copy())

    return chain.__class__(pd.concat(filtered_frames, ignore_index=True), chain._calendar)
