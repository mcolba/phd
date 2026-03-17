"""Data loading helpers for option chain construction."""

from __future__ import annotations

import pandas as pd

from vol_risk.calibration.option_chain import OptionChain
from vol_risk.utils.calendar import Actual365Fixed

_CBOE_RENAME: dict[str, str] = {
    "quote_date": "anchor",
    "expiration": "expiry",
    "trade_volume": "volume",
    "bid_eod": "bid",
    "ask_eod": "ask",
}

_OPTIONMETRICS_REQUIRED_COLS: tuple[str, ...] = (
    "date",
    "exdate",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "cp_flag",
    "underlying_close",
)


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


def make_optionmetrics_chain(
    df: pd.DataFrame,
    *,
    exclude_weeklies: bool = True,
) -> OptionChain:
    """Build an :class:`OptionChain` from a raw OptionMetrics DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame as loaded from OptionMetrics parquet.
    exclude_weeklies:
        If ``True``, remove rows where ``symbol`` matches the weekly pattern.
    weekly_symbol_pattern:
        Pattern used with ``str.contains`` against the ``symbol`` column.

    Returns:
    -------
    OptionChain
        Validated option chain with standardized column names and types.
    """
    missing = [col for col in _OPTIONMETRICS_REQUIRED_COLS if col not in df.columns]
    if missing:
        msg = f"OptionMetrics DataFrame is missing required columns: {missing}."
        raise ValueError(msg)

    df_chain = df.copy()
    if exclude_weeklies and "symbol" in df_chain.columns:
        symbol = df_chain["symbol"].fillna("").astype(str)
        df_chain = df_chain.loc[~symbol.str.contains("SPXW", na=False)]

    df_chain = df_chain.assign(
        strike=df_chain["strike_price"].astype(float) / 1000.0,
        anchor=pd.to_datetime(df_chain["date"]),
        expiry=pd.to_datetime(df_chain["exdate"]),
        bid=df_chain["best_bid"].astype(float),
        ask=df_chain["best_offer"].astype(float),
        mid=lambda x: 0.5 * (x["bid"] + x["ask"]),
        volume=df_chain["volume"].fillna(0).astype(int),
        option_type=df_chain["cp_flag"].astype(str).str.strip().str.upper(),
        spot=df_chain["underlying_close"].astype(float),
    )
    return OptionChain(df_chain, Actual365Fixed)
