from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class MACD(Indicator):
    """
    Moving Average Convergence/Divergence (MACD)

    Standard (EMA-based):
        macd   = EMA(fast) - EMA(slow)
        signal = EMA(macd, signal_window)

    Params:
        fast_window  (int): fast EMA length (default 12)
        slow_window  (int): slow EMA length (default 26)
        signal_window(int): signal EMA length (default 9)
        column       (str): source column (default 'close')
        adjust       (bool): passed to pandas ewm(); False = recursive weighting
        min_periods  (int|None): default = slow_window (starts when slow EMA valid)

    Returns:
        pd.DataFrame with columns: 'macd', 'signal'
    """
    category = "momentum"
    slug = "macd"
    name = "MACD"
    indicator_type = IndicatorType.LINE
    plot_row = 1  # typically plotted below price

    def __init__(
        self,
        fast_window: int = 12,
        slow_window: int = 26,
        signal_window: int = 9,
        column: str = "close",
        adjust: bool = False,
        min_periods: int | None = None,
    ):
        if fast_window <= 0 or slow_window <= 0 or signal_window <= 0:
            raise ValueError("All MACD windows must be > 0")
        if fast_window >= slow_window:
            # Traditional MACD expects fast < slow
            raise ValueError("fast_window must be < slow_window")

        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.signal_window = int(signal_window)
        self.column = column
        self.adjust = bool(adjust)
        self.min_periods = slow_window if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.column]

    # --- helpers ---
    def _ema(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column].astype(float)

        # EMAs
        ema_fast = self._ema(s, self.fast_window, self.fast_window)
        ema_slow = self._ema(s, self.slow_window, self.slow_window)

        macd_line = ema_fast - ema_slow
        signal = self._ema(macd_line, self.signal_window, self.signal_window)

        out = pd.DataFrame(
            {"macd": macd_line, "signal": signal},
            index=df.index,
        )

        # Global min_periods mask (defaults to slow_window)
        if self.min_periods and self.min_periods > 0:
            valid = s.expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out