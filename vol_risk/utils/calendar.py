import datetime as dt
from abc import abstractmethod

import pandas as pd

from vol_risk.protocols import DayCountCalendar


class Actual365Fixed(DayCountCalendar):
    """Actual/365 fixed day count convention."""

    @abstractmethod
    def year_fraction(start: dt.date, end: dt.date) -> pd.Series:
        return (end - start).dt.days / 365.0
