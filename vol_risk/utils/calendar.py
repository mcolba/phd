import pandas as pd

from vol_risk.protocols import DayCountCalendar


class Actual365Fixed(DayCountCalendar):
    """Actual/365 fixed day count convention."""

    @staticmethod
    def year_fraction(start: pd.Series, end: pd.Series) -> pd.Series:
        return (end - start).dt.days / 365.0
