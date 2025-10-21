from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class DON(Indicator):
    """
    Donchian Channels (DON)

    Definitions (for a lookback window n):
      upper_t = max( high_{t-n+1..t} )    [or exclude current bar if include_current=False]
      lower_t = min(  low_{t-n+1..t} )
      mid_t   = (upper_t + lower_t) / 2

    Options:
      - Separate lookbacks for upper/lower via upper_window / lower_window
      - Exclude current bar (no-repaint) with include_current=False

    Common defaults: window=20, include_current=True
    """
    category = "volatility"
    slug = "don"
    name = "Donchian Channels"
    indicator_type = IndicatorType.BANDS
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        upper_window: int | None = None,
        lower_window: int | None = None,
        high_col: str = "high",
        low_col: str = "low",
        include_current: bool = True,     # False -> use previous bar only (shifted by 1)
        min_periods: int | None = None,
    ):
        if window <= 0:
            raise ValueError("window must be > 0")
        if upper_window is not None and upper_window <= 0:
            raise ValueError("upper_window must be > 0 when provided")
        if lower_window is not None and lower_window <= 0:
            raise ValueError("lower_window must be > 0 when provided")

        self.window = int(window)
        self.upper_window = int(upper_window) if upper_window is not None else None
        self.lower_window = int(lower_window) if lower_window is not None else None
        self.high_col = high_col
        self.low_col = low_col
        self.include_current = bool(include_current)

        # default min_periods = max(upper_len, lower_len)
        upper_len = self.upper_window or self.window
        lower_len = self.lower_window or self.window
        default_mp = max(upper_len, lower_len)
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.high_col, self.low_col]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df[self.high_col].astype(float)
        l = df[self.low_col].astype(float)

        ulen = self.upper_window or self.window
        llen = self.lower_window or self.window

        # Rolling extrema
        upper = h.rolling(window=ulen, min_periods=self.min_periods).max()
        lower = l.rolling(window=llen, min_periods=self.min_periods).min()

        # Optionally exclude current bar (no-repaint)
        if not self.include_current:
            upper = upper.shift(1)
            lower = lower.shift(1)

        mid = (upper + lower) / 2.0

        out = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower}, index=df.index)

        # Final mask for min_periods (protect edges when shifted)
        if self.min_periods and self.min_periods > 0:
            valid = h.expanding().count() >= self.min_periods + (0 if self.include_current else 1)
            out = out.where(valid, np.nan)

        return out
