import logging
import math
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial, reduce

import numpy as np
import pandas as pd
from arbitragerepair import constraints

# from arbitragerepair.repair import l1
# from cvxopt import matrix, solvers
from scipy import sparse
from scipy.optimize import linprog

from vol_risk.market_data.opt_chain import NoArbBounds, OptionChain, OptionSlice
from vol_risk.models.black76 import (
    black76_price,
    black76_undisc_fwd_delta,
    black76_vega,
    implied_black_vol,
)
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.vol_surface.moneyness import MONEYNESS_REGISTRY, DeltaMoneyness, Moneyness

log = logging.getLogger(__name__)

# Type alias for a single transformation step
ChainTransform = Callable[[OptionChain], OptionChain]

# solvers.options["show_progress"] = False


@dataclass(frozen=True)
class ChainCutoff:
    """Parameters passed to :func:`apply_cutoffs`."""

    moneyness_type: str
    bounds: tuple[float, float]


def compose(*transforms: ChainTransform) -> ChainTransform:
    """Compose multiple OptionChain transforms into a single pipeline."""

    def pipeline(chain: OptionChain) -> OptionChain:
        return reduce(lambda c, t: t(c), transforms, chain)

    return pipeline


def _as_float_array(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _as_scalar(x: np.ndarray | float) -> float:
    return float(_as_float_array(x)[0])


def _require_call_only(chain: OptionChain) -> None:
    if not np.all(chain.option_type == "C"):
        msg = "Function expects a call-only chain. Use make_otm_to_call first."
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

    discount = _as_scalar(market.disc(tau))
    forward = _as_scalar(market.fwd(tau))
    total_variance = []

    for strike, price in zip(option_slice.k, option_slice.mid, strict=True):
        # TODO @Marco: remove clipping (?)
        clean_price = _clip_call_price(float(price), discount, forward, float(strike))
        sigma = implied_black_vol(
            price=clean_price,
            fwd=forward,
            strike=float(strike),
            tau=tau,
            disc=discount,
            is_call=True,
        )
        total_variance.append(max(float(sigma) ** 2 * tau, min_total_variance))

    return np.asarray(total_variance, dtype=float), forward, discount


def _make_quote_grid(k_min: float, k_max: float, grid_size: int) -> np.ndarray:
    return np.linspace(k_min, k_max, grid_size, dtype=float)


def liquidity_filter(
    chain: OptionChain,
    oi_min: None | int = None,
    bid_min: None | float = None,
    mid_min: None | float = None,
    rel_bid_ask_max: None | float = None,
    min_ttm: None | int = None,
    validate_chain: bool = False,
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
        mask &= (expiry_days - anchor_days).astype(int) >= min_ttm

    return chain.__class__(df.loc[mask, :], chain._calendar, validate=validate_chain)


def _solve_weighted_l1_repair(
    mat_A: np.ndarray,  # noqa: N803
    vec_b: np.ndarray,
    price: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted version of the l1() arbitrage repair from arbitragerepair."""
    n_quote = mat_A.shape[1]
    sol = []

    if weights is None:
        weights = np.ones(n_quote)

    if price.shape[0] != n_quote or weights.shape[0] != n_quote:
        msg = "mat_a, price, and weights must agree on the number of quotes."
        raise ValueError(msg)

    # Construct required quantities for the LP
    A = -np.hstack((mat_A, -mat_A))
    b = -(vec_b - mat_A.dot(price))
    coeff = np.hstack((weights, weights))

    A1 = np.vstack((A, -np.diag(np.ones(2 * n_quote))))
    b1 = np.hstack((b, np.zeros(2 * n_quote)))

    # The original l1() implementation in arbitragerepair uses cvxopt, which can be very slow for large aparse problems.
    """
    G = matrix(A1)
    h = matrix(b1)
    c = matrix(coeff)

    # Scale the constraint for numerical stability
    # A * (scale * epsilon) >= scale * b

    G *= 2.0
    h *= 2.0

    scale = 1
    for _ in range(max_attempts):
        sol = solvers.lp(c, G, h, solver=solver)

        if sol["status"] == "optimal":
            break

        c *= 10
        h *= 10
        scale *= 10

    if sol["status"] == "optimal":
        x = np.array(sol["x"])
        epsilon = (x[:n_quote] - x[n_quote:]).flatten()
        epsilon /= scale
    else:
        epsilon = []
        log.warning("Arbitrage repair optimal perturbation is not found.")
    """

    A1_sparse = sparse.csr_matrix(A1)
    sol = linprog(
        c=coeff,
        A_ub=A1_sparse,
        b_ub=b1,
        method="highs-ds",
    )

    if sol.success:
        epsilon = sol.x[:n_quote] - sol.x[n_quote:]
    else:
        epsilon = []
        log.warning("Arbitrage repair optimal perturbation is not found.")

    return epsilon


def repair_arbitrage(
    chain: OptionChain,
    market: LinearEquityMarket,
    synthetic_weight: float = 1.0,
    min_price: float | None = None,
    tolerance: float = 0.0,
    # solver: str = "glpk",
    validate_chain: bool = False,
) -> OptionChain:
    """Repair static arbitrage on a call-only chain with heavier synthetic-quote penalties."""
    _require_call_only(chain)
    idx_sorted = chain.df.reset_index(drop=True).sort_values(by=["expiry", "strike"]).index.to_numpy()
    df_sorted = chain.df.iloc[idx_sorted].copy()

    if "synthetic" not in df_sorted.columns:
        df_sorted["synthetic"] = False

    if "repair_adj" not in df_sorted.columns:
        df_sorted["repair_adj"] = 0.0

    df_sorted["repair_weight"] = np.where(df_sorted["synthetic"], synthetic_weight, 1.0)

    tau = _as_float_array(chain.tau[idx_sorted])
    strike = _as_float_array(chain.k[idx_sorted])
    mid = _as_float_array(chain.mid[idx_sorted])
    disc = _as_float_array(market.disc(tau))
    forward = _as_float_array(market.fwd(tau))
    undisc_mid = mid / disc

    normaliser = constraints.Normalise(min_price=min_price)
    normaliser.fit(T=tau, K=strike, C=undisc_mid, F=forward)
    T1, K1, C1 = normaliser.transform(T=tau, K=strike, C=undisc_mid)
    mat_A, vec_b, _, n_beach = constraints.detect(T=T1, K=K1, C=C1, tolerance=tolerance, verbose=False)

    calendart_arbitrage = sum(n_beach[-3:])
    if calendart_arbitrage > 0:
        arb_repair_mask = normaliser._order_mask  # noqa: SLF001
        repair_weights = df_sorted["repair_weight"].to_numpy(dtype=float)[arb_repair_mask]
        epsilon = _solve_weighted_l1_repair(
            mat_A=mat_A,
            vec_b=vec_b,
            price=C1,
            weights=repair_weights,
        )

        if len(epsilon) == 0:
            log.warning("No repair applied to the chain.")
            return chain

        _, C0 = normaliser.inverse_transform(K=K1, C=C1 + epsilon)
        repair_disc = disc[arb_repair_mask]
        repair_undisc_mid = undisc_mid[arb_repair_mask]
        df_sorted.iloc[arb_repair_mask, df_sorted.columns.get_loc("mid")] = C0 * repair_disc
        df_sorted.iloc[arb_repair_mask, df_sorted.columns.get_loc("repair_adj")] = (
            C0 - repair_undisc_mid
        ) * repair_disc

    return chain.__class__(df_sorted, chain._calendar, validate=validate_chain)


def append_synthetic_quotes() -> OptionChain:
    raise NotImplementedError


def min_strikes_per_slice_filter(
    chain: OptionChain,
    n: int,
    validate_chain: bool = False,
) -> OptionChain:
    """Filter out expiry slices with fewer than n unique strikes."""
    if n <= 1:
        return chain

    mask = chain.df.groupby("expiry")["strike"].transform("nunique") >= n
    if not mask.any():
        msg = f"No expiry slices have at least {n} unique strikes after filtering."
        raise ValueError(msg)

    return chain.__class__(chain.df.loc[mask].copy(), chain._calendar, validate=validate_chain)


def remove_short_span_slices(chain: OptionChain, validate_chain: bool = False) -> OptionChain:
    """Remove expiry slices with unusually short strike spans compared to their neighbors."""
    span = chain.df.groupby("expiry")["strike"].agg(np.ptp).sort_index()

    mean_span = span.mean()
    rel_span = span / mean_span
    rel_span_change = rel_span.diff()

    mask_mat = (rel_span_change < -0.15) & (rel_span < 0.8)
    mask_mat.iloc[-1] = False
    mask = ~chain.df["expiry"].isin(span.index[mask_mat])

    return chain.__class__(chain.df.loc[mask].copy(), chain._calendar, validate=validate_chain)


def make_otm_to_call(chain: OptionChain, le: LinearEquityMarket, validate_chain: bool = False) -> OptionChain:
    """Create a call-only view of an option chain."""
    df = chain.df.copy()
    tau = chain.tau

    is_otm_c = (df["option_type"] == "C") & (df["strike"] >= le.fwd(tau))
    is_otm_p = (df["option_type"] == "P") & (df["strike"] < le.fwd(tau))

    calls = df.loc[is_otm_c, :]

    # OTM puts to ITM calls using put-call parity
    puts = df.loc[is_otm_p, :]
    p_tau = chain.tau[is_otm_p]
    p_fwd_contract = le.disc(p_tau) * (le.fwd(p_tau) - puts["strike"].to_numpy())

    puts = puts.assign(
        option_type="C",
        quote_type="synthetic",
        bid=p_fwd_contract + chain.bid[is_otm_p],
        ask=p_fwd_contract + chain.ask[is_otm_p],
        mid=p_fwd_contract + chain.mid[is_otm_p],
    )

    return chain.__class__(pd.concat([calls, puts], ignore_index=True), chain._calendar, validate=validate_chain)


def _filter_informative_values(
    values: np.ndarray,
    priority: np.ndarray,
    removable: np.ndarray,
    min_distance: float = 0.01,
    max_distance: float | None = None,
) -> np.ndarray:
    """Filter out values that are too close to their neighbors, keeping those with higher priority."""
    n = values.size
    active = np.ones(n, dtype=bool)
    prev_idx = np.arange(n) - 1
    next_idx = np.arange(n) + 1
    next_idx[-1] = -1

    for i in np.argsort(priority):
        if (not active[i]) or (not removable[i]):
            continue

        left_idx = prev_idx[i]
        right_idx = next_idx[i]

        if left_idx == -1 or right_idx == -1:
            continue

        nearest_distance = min(
            np.log(values[i] / values[left_idx]),
            np.log(values[right_idx] / values[i]),
        )
        if nearest_distance < min_distance:
            active[i] = False
            if left_idx != -1:
                next_idx[left_idx] = right_idx
            if right_idx != -1:
                prev_idx[right_idx] = left_idx

    if max_distance is not None:
        for i in np.argsort(-priority):
            if not active[i]:
                continue

            left_idx = prev_idx[i]
            right_idx = next_idx[i]

            if left_idx == -1 or right_idx == -1:
                continue

            furthest_distance = max(
                np.log(values[i] / values[left_idx]),
                np.log(values[right_idx] / values[i]),
            )
            if furthest_distance < max_distance:
                active[i] = False
                if left_idx != -1:
                    next_idx[left_idx] = right_idx
                if right_idx != -1:
                    prev_idx[right_idx] = left_idx

    return active


def soft_liquidity_filter(
    chain: OptionChain,
    lin_mkt: LinearEquityMarket,
    oi_soft_min: int = 50,
    rel_bid_ask_soft_max: float = 0.20,
    min_lk_distance: float = 0.05,
    max_lk_distance: float | None = None,
    validate_chain: bool = False,
) -> OptionChain:
    """Drop low-liquidity near-duplicate quotes."""
    iv_mid, iv_bid, iv_ask = [
        implied_black_vol(
            price=p,
            fwd=lin_mkt.fwd(chain.tau),
            strike=chain.k,
            tau=chain.tau,
            disc=lin_mkt.disc(chain.tau),
            is_call=chain.option_type == "C",
        )
        for p in (chain.mid, chain.bid, chain.ask)
    ]

    # Remove qotes with invalid mid implied volatility
    mask0 = ~(np.isnan(iv_mid) | np.isinf(iv_mid) | (iv_mid < 1e-2) | (iv_mid > 2.0))

    df = chain.df.iloc[mask0].reset_index(drop=True).copy()
    mask = np.ones(df.shape[0], dtype=bool)
    iv_mid, iv_bid, iv_ask, tau = iv_mid[mask0], iv_bid[mask0], iv_ask[mask0], chain.tau[mask0]

    strike = df["strike"].to_numpy()
    open_int = df["open_interest"].fillna(0.0).to_numpy()
    rel_bid_ask = np.nan_to_num((iv_ask - iv_bid) / iv_mid, nan=np.inf)

    def _piecewise_score(x: np.ndarray, bounds: tuple[float, float, float]) -> np.ndarray:
        x_min, x_mid, x_max = bounds
        score = np.where(
            x <= x_mid,
            0.5 * (x - x_min) / (x_mid - x_min),
            0.5 + 0.5 * (x - x_mid) / (x_max - x_mid),
        )
        return np.clip(score, 0.0, 1.0)

    oi_score = _piecewise_score(open_int, (0.0, oi_soft_min, 500))
    spread_score = _piecewise_score(-rel_bid_ask, (-1, -rel_bid_ask_soft_max, 0.01))
    liquidity_score = (oi_score + spread_score) / 2.0

    is_removable = (open_int < oi_soft_min) | (rel_bid_ask > rel_bid_ask_soft_max)

    for _, grp in df.groupby("expiry", sort=False):
        idx = grp.index.to_numpy()

        if idx.size < 3:
            continue

        sqrt_tau = np.sqrt(float(tau[idx[0]]))
        order_k = np.argsort(strike[idx])
        idx_sorted = idx[order_k]

        active = _filter_informative_values(
            values=strike[idx_sorted],
            priority=liquidity_score[idx_sorted],
            removable=is_removable[idx_sorted],
            min_distance=min_lk_distance * sqrt_tau,
            max_distance=max_lk_distance * sqrt_tau if max_lk_distance is not None else None,
        )

        mask[idx_sorted[~active]] = False

    return OptionChain(df[mask].copy(), chain._calendar, validate=validate_chain)


def get_atmf_vol(sl: OptionSlice, le: LinearEquityMarket) -> float:
    """Get ATMF volatilities from an option chain."""
    _require_call_only(sl)
    tau = sl.slice_tau
    fwd = le.fwd(tau)

    idx_sort = np.argsort(sl.k)
    strike = sl.k[idx_sort]
    price = sl.mid[idx_sort]

    k = 3
    idx_closest = np.searchsorted(strike, fwd)
    idx_first = max(idx_closest - k, 0)
    idx_last = min(idx_closest + k + 1, len(strike))
    mask = list(range(idx_first, idx_last))

    deg = 2 if len(mask) > 2 else 1
    z = np.polyfit(strike[mask], price[mask], deg)
    atm_price = np.poly1d(z)(fwd)

    return float(
        implied_black_vol(
            price=atm_price,
            fwd=fwd,
            strike=fwd,
            tau=tau,
            disc=le.disc(tau),
            is_call=True,
        )
    )


def apply_cutoffs(
    chain: OptionChain,
    cutoffs: Iterator[ChainCutoff],
    lin_mkt: LinearEquityMarket,
    op: str = "or",
    validate_chain: bool = False,
) -> OptionChain:
    """Apply cutoffs to an option chain."""

    def _unpack_bounds(bounds: tuple[float, float], moneyness: Moneyness) -> tuple[float, float]:
        if not isinstance(moneyness, DeltaMoneyness):
            return bounds
        return bounds[::-1]

    strikes = chain.k
    tau = chain.tau
    mask = np.zeros_like(strikes, dtype=bool)

    atmf_vol_map = {k: get_atmf_vol(sl, lin_mkt) for k, sl in chain}
    atmf_vol = pd.Series(chain.expiry).map(atmf_vol_map).to_numpy(dtype=float)

    for co in cutoffs:
        inner_mask = np.ones_like(strikes, dtype=bool)

        bounds, mn_convention = co.bounds, co.moneyness_type
        moneyness_engine = MONEYNESS_REGISTRY[mn_convention](le=lin_mkt)

        if bounds[0] >= bounds[1]:
            msg = f"Lower bound {bounds[0]} must be less than upper bound {bounds[1]}."
            raise ValueError(msg)

        lb, ub = _unpack_bounds(bounds, moneyness_engine)

        if (lb is not None) and not np.isneginf(lb):
            strike_lb = moneyness_engine.invert(moneyness=lb, tau=tau, sigma=atmf_vol)
            inner_mask &= strikes >= strike_lb

        if (ub is not None) and not np.isposinf(ub):
            strike_ub = moneyness_engine.invert(moneyness=ub, tau=tau, sigma=atmf_vol)
            inner_mask &= strikes <= strike_ub

        logic_op = operator.iand if op.lower() == "and" else operator.ior
        mask = logic_op(mask, inner_mask)

    return chain.__class__(chain.df.iloc[mask, :], chain._calendar, validate=validate_chain)


def _collect_slice_data(
    chain: OptionChain,
    market: LinearEquityMarket,
    min_total_variance: float,
) -> list[dict]:
    """Build per-slice boundary data (sorted by log-moneyness within each slice)."""
    slices: list[dict] = []
    for expiry, sl in chain:
        tau = sl.slice_tau
        fwd = _as_scalar(market.fwd(tau))
        disc = _as_scalar(market.disc(tau))
        total_var, _, _ = _slice_total_variance(sl, market, min_total_variance)
        log_m = np.log(np.asarray(sl.k, dtype=float) / fwd)
        order = np.argsort(log_m)
        slices.append(
            {
                "expiry": expiry,
                "tau": tau,
                "fwd": fwd,
                "disc": disc,
                "log_m": log_m[order],
                "total_var": total_var[order],
                "price_norm": sl.mid[order] / (disc * fwd),
            }
        )
    return slices


def _sweep_one_wing(
    slice_data: list[dict],
    m_range: tuple[float, float],
) -> list[tuple[float, float, float]]:
    """Single-threshold carry-forward sweep for one wing."""
    if len(m_range) != 2:
        msg = f"m_range must be a tuple of (lower_bound, upper_bound), got {m_range}."
        raise ValueError(msg)
    if np.sign(m_range[0]) != np.sign(m_range[1]) and math.prod(m_range) != 0:
        msg = f"m_range bounds must have the same sign, got {m_range}."
        raise ValueError(msg)

    m0, m1 = min(m_range), max(m_range)

    carry_fwd = None

    def _get_range_idx(m0, m1) -> np.ndarray:  # noqa: ANN001
        if np.sign(m1) == 0:
            if np.isfinite(m1):
                return -1
            return 0
        if np.isfinite(m0):
            return 0
        return -1

    point_idx = _get_range_idx(m0, m1)

    synthetics = []
    for i, cur in enumerate(slice_data):
        if i == 0:
            continue

        prev = slice_data[i - 1]

        cur_mask = (cur["log_m"] > m0) & (cur["log_m"] <= m1)
        prev_mask = (prev["log_m"] > m0) & (prev["log_m"] <= m1)

        if cur_mask.any():
            carry_fwd = None
        elif not prev_mask.any() and carry_fwd is not None:
            synthetics.append((cur["tau"], *carry_fwd))
        elif prev_mask.any():
            idx = np.argsort(prev["log_m"][prev_mask])[point_idx]

            ref_m = float(prev["log_m"][prev_mask][idx])
            ref_price_norm = float(prev["price_norm"][prev_mask][idx])

            synthetics.append((cur["tau"], ref_m, ref_price_norm))
            carry_fwd = (ref_m, ref_price_norm)

    return synthetics


def get_calendar_arb_upper_bounds(
    chain: OptionChain,
    market: LinearEquityMarket,
    lm_bounds: tuple[float, float] = (-0.6, 0.6),
    min_total_variance: float = 1e-8,
) -> NoArbBounds:
    """Build a synthetic OptionChain of boundary quotes preventing calendar arbitrage."""
    _require_call_only(chain)
    m_lb, m_ub = lm_bounds

    slice_data = _collect_slice_data(chain, market, min_total_variance)
    slice_data.sort(key=lambda s: s["tau"], reverse=True)
    expiry_info = {s["expiry"]: s for s in slice_data}

    raw = []

    step_size = 0.1
    lw_points = np.concatenate(
        [
            [-np.inf],
            np.linspace(m_lb, 0.0, num=1 + math.ceil((0.0 - m_lb) / step_size)),
        ]
    )[:, None]
    rw_points = np.concatenate(
        [
            np.linspace(0.0, m_ub, num=1 + math.ceil((m_ub - 0.0) / step_size)),
            [np.inf],
        ]
    )[:, None]

    intervals_arr = np.vstack(
        [
            np.hstack([lw_points[:-1], lw_points[1:]]),
            np.hstack([rw_points[:-1], rw_points[1:]]),
        ]
    )

    for i in list(map(tuple, intervals_arr)):
        raw.extend(_sweep_one_wing(slice_data, i))

    # Build carry-forward rows for the synthetic OptionChain
    tau_to_expiry = {s["tau"]: s["expiry"] for s in slice_data}

    rows = []
    for tau, lm, price_norm in raw:
        info = expiry_info[tau_to_expiry[tau]]
        fwd, disc = info["fwd"], info["disc"]
        strike = fwd * np.exp(lm)
        sigma = np.clip(
            implied_black_vol(
                price=price_norm * disc * fwd,
                fwd=fwd,
                strike=strike,
                tau=tau,
                disc=disc,
                is_call=True,
            ),
            0.02,
            1.5,
        )
        vega = black76_vega(fwd=fwd, strike=strike, tau=tau, sigma=sigma, disc=info["disc"]).item()
        weight = info["disc"] * fwd / max(vega, 1e-6)
        rows.append(
            {
                "expiry": tau_to_expiry[tau],
                "strike": float(strike),
                "lkf": float(lm),
                "tau": float(tau),
                "option_type": "C",
                "price_norm_ub": price_norm,
                "price_norm_lb": np.nan,
                "weight": float(weight),
            }
        )

    # NaN-placeholder rows for gap intervals (no market data, no carry-forward).
    for sd in slice_data:
        tau_sd = sd["tau"]
        for interval in intervals_arr:
            m0, m1 = float(min(interval)), float(max(interval))
            if not np.isfinite(m0) or not np.isfinite(m1):
                continue
            if ((sd["log_m"] > m0) & (sd["log_m"] <= m1)).any():
                continue
            if any(r[0] == tau_sd and m0 < r[1] <= m1 for r in raw):
                continue
            mid_lm = (m0 + m1) / 2.0
            info = expiry_info[sd["expiry"]]
            rows.append(
                {
                    "expiry": sd["expiry"],
                    "strike": float(info["fwd"] * np.exp(mid_lm)),
                    "lkf": float(mid_lm),
                    "tau": float(tau_sd),
                    "option_type": "C",
                    "price_norm_ub": np.nan,
                    "price_norm_lb": np.nan,
                    "weight": np.nan,
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values(["expiry", "strike"], ignore_index=True)

    def _force_monotonicity(df: pd.DataFrame, name: str = "price_norm_ub") -> pd.DataFrame:
        """Force the upper bound to be non-increasing in log-moneyness."""
        _df = df.set_index(["expiry", "lkf"])[name].to_frame().copy()

        pt = _df.pivot_table(index=["expiry"], columns=["lkf"], values=[name]).fillna(np.inf)
        pt = np.minimum.accumulate(pt, axis=1)
        pt.iloc[::-1, :] = np.minimum.accumulate(pt.iloc[::-1, :], axis=0)
        return pt.stack(future_stack=True).loc[_df.index].to_numpy()  # noqa: PD013

    # Apply monotonicity enforcement only to rows that carry a finite upper bound.
    ub_mask = df["price_norm_ub"].notna()
    if ub_mask.any():
        df.loc[ub_mask, "price_norm_ub"] = _force_monotonicity(df.loc[ub_mask], "price_norm_ub")

    return NoArbBounds(df)
