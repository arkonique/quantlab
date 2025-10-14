from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np
from typing import Literal


class RMA(Indicator):
    """
    Wilder's Moving Average (RMA), also known as Wilder's Smoothing.

    Equivalent to an EMA with alpha = 1/n and adjust=False, but by default
    seeds with the SMA over the first n points (Wilder's original convention).
    """
    category = "trend"
    slug = "rma"
    indicator_type = IndicatorType.LINE  # plotted as a line

    def __init__(
        self,
        window: int = 14,
        column: str = "close",
        seed: Literal["sma", "first"] = "sma",
    ):
        """
        Args:
            window (int): lookback period n.
            column (str): which column to compute on (e.g., 'close' or 'gain').
            seed (str): 'sma' uses SMA(n) as the initial value; 'first' uses x0.
        """
        self.window = int(window)
        self.column = column
        self.seed = seed

    def required_columns(self):
        return [self.column]

    def _rma_sma_seed(self, s: pd.Series, n: int) -> pd.Series:
        """RMA with SMA(n) seed at index n-1, then recursive Wilder updates."""
        arr = s.to_numpy(dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)

        if len(arr) < n:
            return pd.Series(out, index=s.index)

        # Initial SMA over the first n values
        init = np.nanmean(arr[:n])
        out[n - 1] = init

        alpha = 1.0 / n
        # Recursive update
        for i in range(n, len(arr)):
            prev = out[i - 1]
            x = arr[i]
            if np.isnan(prev) or np.isnan(x):
                out[i] = np.nan
            else:
                out[i] = prev + alpha * (x - prev)

        return pd.Series(out, index=s.index)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Wilder's Moving Average (RMA).

        Returns:
            pd.Series: RMA values aligned to df.index.
        """
        n = self.window
        s = df[self.column].astype(float)

        if self.seed == "first":
            # Equivalent to EMA with alpha=1/n and recursive weighting
            rma = s.ewm(alpha=1.0 / n, adjust=False).mean()
        else:
            # Wilder's original: seed with SMA(n)
            rma = self._rma_sma_seed(s, n)

        rma.name = f"RMA({n})"
        return rma
