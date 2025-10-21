from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class HMA(Indicator):
    category = "trend"
    slug = "hma"
    name = "Hull Moving Average"
    indicator_type = IndicatorType.LINE  # plotted as a line
    plot_row = 0  # typically plotted on price chart

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Hull Moving Average (HMA)

        Args:
            window (int): base period for the HMA.
            column (str): which column to compute the HMA on (usually 'close').
        """
        self.window = int(window)
        self.column = column

    def required_columns(self):
        return [self.column]

    def _wma(self, series: pd.Series, window: int) -> pd.Series:
        """Helper function to compute Weighted Moving Average (WMA)."""
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the Hull Moving Average (HMA).

        Returns:
            pd.Series: HMA values aligned to df.index.
        """
        n = self.window
        half_length = int(n / 2)
        sqrt_length = int(np.sqrt(n))

        wma_half = self._wma(df[self.column], half_length)
        wma_full = self._wma(df[self.column], n)

        diff = 2 * wma_half - wma_full
        hma = self._wma(diff, sqrt_length)
        hma.name = f"HMA({n})"
        return hma
