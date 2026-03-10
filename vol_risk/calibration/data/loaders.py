"""Data loading helpers for option chain construction."""

from __future__ import annotations

import pandas as pd

from vol_risk.calibration.data.option_chain import OptionChain
from vol_risk.utils.calendar import Actual365Fixed

_CBOE_RENAME: dict[str, str] = {
    "quote_date": "anchor",
    "expiration": "expiry",
    "trade_volume": "volume",
    "bid_eod": "bid",
    "ask_eod": "ask",
}


def make_cboe_chain(
    df: pd.DataFrame,
    underlying_symbol: str,
    root: str,
) -> OptionChain:
    """Build an :class:`OptionChain` from a raw CBOE EOD CSV DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame as loaded from a CBOE EOD CSV file.
    underlying_symbol:
        Value to match against the ``underlying_symbol`` column (e.g. ``"^SPX"``).
    root:
        Value to match against the ``root`` column (e.g. ``"SPX"``).

    Returns:
    -------
    OptionChain
        Validated option chain for the requested underlying.
    """
    df = df.rename(columns=_CBOE_RENAME)

    if "spot" not in df.columns:
        df = df.assign(spot=lambda x: 0.5 * (x["underlying_bid_eod"] + x["underlying_ask_eod"]))

    for col in ("anchor", "expiry"):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="raise")

    if "mid" not in df.columns:
        df = df.assign(mid=lambda x: 0.5 * (x["bid"] + x["ask"]))

    mask = (
        (df["underlying_symbol"] == underlying_symbol)
        & (df["root"] == root)
        & ((df["ask"] - df["bid"]) >= 0)
        & (df["mid"] > 0)
    )
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        msg = f"No options found for underlying_symbol={underlying_symbol!r}, root={root!r}."
        raise ValueError(msg)

    return OptionChain(df_filtered, Actual365Fixed)
