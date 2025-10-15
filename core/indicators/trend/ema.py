from ..base import Indicator, IndicatorType
import pandas as pd


class EMA(Indicator):
    category = "trend"
    slug = "ema"
    name = "Exponential Moving Average"
    indicator_type = IndicatorType.LINE  # this will be plotted as a line

    def __init__(self, window: int = 20, column: str = "close", adjust: bool = False):
        """
        Exponential Moving Average (EMA)

        Args:
            window (int): the smoothing window / period.
            column (str): which column to compute the EMA on (usually 'close').
            adjust (bool): passed directly to pandas ewm(); False for recursive weighting.
        """
        self.window = int(window)
        self.column = column
        self.adjust = adjust

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the EMA over the specified column.

        Returns:
            pd.Series: EMA values aligned to df.index.
        """
        ema = df[self.column].ewm(span=self.window, adjust=self.adjust).mean()
        ema.name = f"EMA({self.window})"
        return ema
