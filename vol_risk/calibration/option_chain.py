import datetime as dt
from collections.abc import Generator
from dataclasses import InitVar, dataclass

import numpy as np
import pandas as pd
import pandera as pa
from pandera.pandas import Check, Column, DataFrameSchema

from vol_risk.protocols import Array, DayCountCalendar, OptionChainLike


def _ask_ge_bid_if_present(df: pd.DataFrame) -> bool:
    """Check ask >= bid only where both are present and non-null."""
    mask = df["bid"].notna() & df["ask"].notna()
    if not bool(mask.any()):
        return True
    return bool((df.loc[mask, "ask"] >= df.loc[mask, "bid"]).all())


def _expiry_ge_anchor(df: pd.DataFrame) -> bool:
    """Check that expiry >= anchor for all rows.

    Time information is omitted as PM options might not have a timestamp attached.
    """
    return bool((df["expiry"].dt.date >= df["anchor"].dt.date).all())


option_chain_schema = DataFrameSchema(
    columns={
        "anchor": Column(pa.DateTime, required=True),
        "spot": Column(float, required=True),
        "strike": Column(float, Check.ge(0), required=True),
        "expiry": Column(pa.DateTime, required=True),
        "volume": Column("Int64", required=True, nullable=True),
        "option_type": Column(str, Check.isin(["C", "P"]), required=True),
        "mid": Column(float, required=True),
        "bid": Column(float, required=True, nullable=True),
        "ask": Column(float, required=True, nullable=True),
        "quote_type": Column(str, Check.isin(["quote", "synthetic", "parity"]), required=False, nullable=True),
        "repair_adj": Column(float, required=False, nullable=True),
    },
    checks=[
        Check(_ask_ge_bid_if_present, error="ask must be >= bid"),
        Check(_expiry_ge_anchor, error="expiry must be >= anchor"),
        Check(lambda df: df["spot"].nunique() == 1, error="all spot values must be the same"),
        Check(lambda df: df["anchor"].nunique() == 1, error="OptionChain must have a single anchor date"),
    ],
    unique=["strike", "expiry", "option_type"],
    coerce=True,
    strict=False,  # allows extra columns
)


@dataclass(frozen=True)
class OptionChain(OptionChainLike):
    """Option chain data."""

    _df: pd.DataFrame
    _calendar: DayCountCalendar
    validate: InitVar[bool] = True

    def __post_init__(self, validate: bool) -> None:
        if validate:
            object.__setattr__(
                self,
                "_df",
                (
                    option_chain_schema.validate(self._df).sort_values(
                        ["expiry", "strike", "option_type"], ignore_index=True
                    )
                ),
            )

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self):
        return self._group_by_expiry()

    def __getitem__(self, key: str) -> Array:
        """Get a slice by expiry date."""
        mask = self._df["expiry"] == pd.Timestamp(key)
        return OptionSlice(self._df[mask].copy(), self._calendar)

    def _to_array(self, x: pd.Series) -> Array:
        arr = x.to_numpy(copy=False)
        arr.flags.writeable = False
        return arr

    @property
    def df(self) -> pd.DataFrame:
        """Return a shallow copy of the underlying DataFrame."""
        return self._df.copy(deep=False)

    @property
    def spot(self) -> float:
        """Return the unique spot in the chain."""
        return float(self._df["spot"].iloc[0])

    @property
    def k(self) -> Array:
        """Return the array of strikes in the chain."""
        return self._to_array(self._df["strike"])

    @property
    def expiry(self) -> Array:
        """Return the array of maturities in the chain."""
        return self._to_array(self._df["expiry"])

    @property
    def tau(self) -> Array:
        """Calculate time to expiry in years based."""
        year_fraction = self._calendar.year_fraction(self._df["anchor"], self._df["expiry"])
        return self._to_array(year_fraction)

    @property
    def mid(self) -> Array:
        """Return the mid prices of options in the chain."""
        if "mid" in self._df.columns:
            return self._to_array(self._df["mid"])
        mid = (self._df["bid"] + self._df["ask"]) / 2.0
        return self._to_array(mid)

    @property
    def bid(self) -> Array:
        """Return the bid prices of options in the chain."""
        return self._to_array(self._df["bid"])

    @property
    def ask(self) -> Array:
        """Return the ask prices of options in the chain."""
        return self._to_array(self._df["ask"])

    @property
    def option_type(self) -> Array:
        """Return the option types in the chain."""
        return self._to_array(self._df["option_type"])

    def _group_by_expiry(self) -> Generator[tuple[dt.datetime, "OptionChain"], None, None]:
        """Yield (expiry, OptionChain) pairs grouped by expiry date."""
        for expiry, group_df in self._df.groupby("expiry"):
            yield expiry, OptionSlice(group_df.copy(), self._calendar)


@dataclass(frozen=True)
class OptionSlice(OptionChain):
    """Option chain data."""

    validate: InitVar[bool] = False

    def __post_init__(self, validate: bool) -> None:
        super().__post_init__(validate)

        n_expiries = self._df["expiry"].nunique()
        if n_expiries != 1:
            msg = f"OptionSlice expects a single expiry. The input data contains {n_expiries}."
            raise ValueError(msg)

        object.__setattr__(self, "_slice_expiry", self._df["expiry"].iloc[0].date())

    def __iter__(self) -> Array:
        msg = "OptionSlice does not support iteration."
        raise NotImplementedError(msg)

    @property
    def slice_tau(self) -> float:
        year_fraction = self.tau[0]
        return float(year_fraction)


def _has_positive_time_value(df: pd.DataFrame) -> bool:
    """Require normalized prices to stay strictly above intrinsic value."""
    if df.empty:
        return True

    m = df["lkf"].to_numpy(dtype=float)
    is_call = df["option_type"].eq("C").to_numpy()

    intrinsic = np.where(
        is_call,
        np.maximum(1.0 - np.exp(m), 0.0),
        np.maximum(np.exp(m) - 1.0, 0.0),
    )

    eps_tv = 1e-10
    for col in ["price_norm_ub", "price_norm_lb"]:
        mask = df[col].notna().to_numpy()
        if not mask.any():
            continue

        px = df.loc[mask, col].to_numpy(dtype=float)
        if (px <= intrinsic[mask] + eps_tv).any():
            return False

    return True


def _has_no_calendar_arb_upper_bound(df: pd.DataFrame, tollerance: float = 1e-8) -> bool:
    """Require next maturity more-ITM call to dominate earlier."""
    if df.empty:
        return True

    not_na = df["price_norm_ub"].notna()
    ub = df.loc[not_na, ["expiry", "lkf", "price_norm_ub", "option_type"]].copy()

    if ub.empty:
        return True

    for option_type, opt_df in ub.groupby("option_type", sort=False):
        opt_df = opt_df.sort_values(["expiry", "lkf"], ignore_index=True)

        seen_m = np.array([], dtype=float)
        seen_p = np.array([], dtype=float)

        for _, grp in ub.groupby("expiry", sort=True):
            m = grp["lkf"].to_numpy(dtype=float)
            p = grp["price_norm_ub"].to_numpy(dtype=float)

            if seen_m.size:
                required = np.empty_like(p)
                for i, mi in enumerate(m):
                    mask = seen_m >= mi if option_type == "C" else seen_m <= mi
                    required[i] = np.max(seen_p[mask]) if mask.any() else -np.inf

                if (p < required - tollerance).any():
                    return False

            seen_m = np.concatenate([seen_m, m])
            seen_p = np.concatenate([seen_p, p])

    return True


no_arb_schema = DataFrameSchema(
    columns={
        "strike": Column(float, Check.ge(0), required=True),
        "lkf": Column(float, required=True),
        "expiry": Column(pa.DateTime, required=True),
        "option_type": Column(str, Check.isin(["C", "P"]), required=True),
        "price_norm_ub": Column(float, required=True, nullable=True),
        "price_norm_lb": Column(float, required=True, nullable=True),
        "weight": Column(float, required=True, nullable=True, default=1.0),
    },
    checks=[
        Check(_has_positive_time_value, error="Normalized prices must have positive time value."),
        Check(
            _has_no_calendar_arb_upper_bound,
            error="Upper bounds violate calendar monotonicity in moneyness.",
        ),
    ],
    unique=["strike", "expiry", "option_type"],
    coerce=True,
    strict=False,
)


@dataclass(frozen=True)
class NoArbBounds:
    """Data structure for storing no-arbitrage bounds on option prices."""

    _df: pd.DataFrame

    def __post_init__(self):
        object.__setattr__(
            self,
            "_df",
            (no_arb_schema.validate(self._df).sort_values(["expiry", "strike", "option_type"], ignore_index=True)),
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key: str) -> Array:
        mask = self._df["expiry"] == pd.Timestamp(key)
        return NoArbBounds(self._df[mask].copy())

    @property
    def call_ub(self) -> pd.DataFrame:
        mask = self._df["price_norm_ub"].notna() & self._df["option_type"].eq("C")
        return self._df.loc[mask, ["expiry", "strike", "price_norm_ub", "weight"]]

    @property
    def call_lb(self) -> pd.DataFrame:
        mask = self._df["price_norm_lb"].notna() & self._df["option_type"].eq("C")
        return self._df.loc[mask, ["expiry", "strike", "price_norm_lb", "weight"]]
