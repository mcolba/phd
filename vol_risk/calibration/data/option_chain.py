import datetime as dt
from collections.abc import Generator
from dataclasses import dataclass

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


option_chain_schema = DataFrameSchema(
    columns={
        "anchor": Column(pa.DateTime, required=True),
        "spot": Column(float, required=True),
        "strike": Column(float, Check.ge(0), required=True),
        "expiry": Column(pa.DateTime, required=True),
        "mid": Column(float, required=True),
        "volume": Column(int, required=True),
        "option_type": Column(str, Check.isin(["C", "P"]), required=True),
        "bid": Column(float, required=True, nullable=True),
        "ask": Column(float, required=True, nullable=True),
    },
    checks=[
        Check(lambda df: df["expiry"] >= df["anchor"], error="expiry must be >= anchor"),
        Check(lambda df: df["spot"].nunique() == 1, error="all spot values must be the same"),
        Check(lambda df: df["anchor"].nunique() == 1, error="OptionChain must have a single anchor date"),
        Check(_ask_ge_bid_if_present, error="ask must be >= bid"),
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

    def __post_init__(self):
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

    def __getitem__(self, column: str) -> Array:
        """Get a column as an immutable array."""
        try:
            return self._to_array(self._df[column])
        except KeyError as e:
            msg = f"Column {column!r} not found in OptionChain."
            raise KeyError(msg) from e

    def _to_array(self, x: pd.Series) -> Array:
        arr = x.to_numpy(copy=False)
        arr.flags.writeable = False
        return arr

    @property
    def df(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._df.copy()

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

    def __post_init__(self) -> None:
        super().__post_init__()

        n_expiries = self._df["expiry"].nunique()
        if n_expiries != 1:
            msg = f"OptionSlice expects a single expiry. The input data contains {n_expiries}."
            raise ValueError(msg)

        object.__setattr__(self, "_slice_expiry", self._df["expiry"].iloc[0].date())

    def __iter__(self) -> Array:
        raise NotImplementedError("OptionSlice does not support iteration.")

    @property
    def slice_tau(self) -> float:
        year_fraction = self.tau[0]
        return float(year_fraction)
