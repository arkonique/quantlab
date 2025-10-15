from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class WMA(Indicator):
    category = "trend"
    slug = "wma"
    name = "Weighted Moving Average"
    indicator_type = IndicatorType.LINE  # plotted as a line

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Weighted Moving Average (WMA)

        Args:
            window (int): number of periods for the moving window.
            column (str): which column to compute WMA on (usually 'close').
        """
        self.window = int(window)
        self.column = column

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the WMA over the specified column.

        Returns:
            pd.Series: WMA values aligned to df.index.
        """
        weights = np.arange(1, self.window + 1)
        wma = (
            df[self.column]
            .rolling(window=self.window)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )
        wma.name = f"WMA({self.window})"
        return wma
