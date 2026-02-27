import numpy as np
import pandas as pd

from vol_risk.calibration.data.option_chain import OptionChain, OptionSlice
from vol_risk.models.black76 import implied_vol_jackel
from vol_risk.models.linear import LinearEquityMarket
from vol_risk.vol_surface.moneyness import DeltaMoneyness, Moneyness


def liquidity_filter(
    chain: OptionChain,
    oi_min: None | int = None,
    bid_min: None | float = None,
    mid_min: None | float = None,
    rel_bid_ask_max: None | float = None,
) -> OptionChain:
    """Filter an option chain for liquid contracts."""
    df = chain.df.copy()

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

    return chain.__class__(df.loc[mask, :].copy(), chain._calendar)


def make_otm_to_call(chain: OptionChain, le: LinearEquityMarket) -> OptionChain:
    """Create a call-only view of an option chain."""
    df = chain.df.copy()

    is_otm_c = (df["option_type"] == "C") & (df["strike"] >= le.spot)
    is_otm_p = (df["option_type"] == "P") & (df["strike"] <= le.spot)

    calls = df.loc[is_otm_c, :]

    # OTM puts to ITM calls using put-call parity
    puts = df.loc[is_otm_p, :]
    mid = chain.mid[is_otm_p]
    tau = chain.tau[is_otm_p]
    fwd_contract = le.df(tau) * (le.fwd(tau) - puts["strike"])

    puts = puts.assign(
        option_type="C",
        bid=np.nan,
        ask=np.nan,
        mid=fwd_contract + mid,
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
    mask = np.concatenate(
        [range(max(idx_closest - k, 0), idx_closest), range(idx_closest, min(idx_closest + k, len(strike)))]
    )
    z = np.polyfit(strike[mask], price[mask], 2)
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
