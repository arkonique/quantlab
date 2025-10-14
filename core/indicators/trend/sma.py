# core/indicators/trend/sma.py
from ..base import Indicator, IndicatorType
import pandas as pd

class SMA(Indicator):
    category = "trend"
    slug = "sma"
    indicator_type = IndicatorType.LINE  # exposes .type = "line"

    def __init__(self, window: int = 20, column: str = "close"):
        self.window = int(window)
        self.column = column

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        s = df[self.column].rolling(window=self.window, min_periods=self.window).mean()
        s.name = f"SMA({self.window})"
        return s
