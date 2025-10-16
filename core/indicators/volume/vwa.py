from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class VWA(Indicator):
    """
    Volume-Weighted Average Price (VWAP).

    Formula:
        Typical Price (TP) = (High + Low + Close) / 3
        VWAP_t = (Σ (TP_i * Volume_i)) / (Σ Volume_i)

    Notes:
    - VWAP reflects the average traded price weighted by volume.
    - Cumulative across the entire DataFrame by default (no session reset).
    - To reset daily, pre-split the DataFrame per session or use a grouped apply.
    """
    category = "volume"
    slug = "vwa"
    name = "Volume Weighted Average Price"
    indicator_type = IndicatorType.LINE

    def __init__(
        self,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        min_periods: int | None = 1,
    ):
        """
        Args:
            high_col (str): Column name for high prices.
            low_col (str): Column name for low prices.
            close_col (str): Column name for close prices.
            volume_col (str): Column name for volume.
            min_periods (int | None): Minimum bars before emitting VWAP.
                                      Defaults to 1.
        """
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.min_periods = min_periods

    def required_columns(self):
        return [self.high_col, self.low_col, self.close_col, self.volume_col]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Volume-Weighted Average Price (VWAP).

        Returns:
            pd.Series: VWAP values aligned to df.index.
        """
        h = df[self.high_col].astype(float)
        l = df[self.low_col].astype(float)
        c = df[self.close_col].astype(float)
        v = df[self.volume_col].astype(float)

        # Typical price
        tp = (h + l + c) / 3.0

        # Cumulative sums
        pv = (tp * v).cumsum()
        vol = v.cumsum()

        # VWAP
        vwap = pv / vol
        if self.min_periods:
            vwap = vwap.where(df[self.close_col].expanding().count() >= self.min_periods)

        vwap.name = "VWA"
        return vwap
