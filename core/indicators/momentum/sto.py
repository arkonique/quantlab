from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class STO(Indicator):
    """
    Stochastic Oscillator (fast/slow)

    Raw %K_t = 100 * (C_t - LL_n) / (HH_n - LL_n)
      where HH_n = rolling max(High, n), LL_n = rolling min(Low, n)

    Smoothed:
      K  = smooth(%K, k_smooth, method)
      D  = smooth(K,   d_window, method)

    Parameters:
        k_window  (int): lookback for HH/LL (default 14)
        k_smooth  (int): smoothing for %K (1 => fast %K; 3 => slow %K) (default 3)
        d_window  (int): smoothing for %D (default 3)
        method    (str): 'sma' | 'ema' | 'rma' for smoothing (default 'sma')
        high_col  (str): column name for high (default 'high')
        low_col   (str): column name for low  (default 'low')
        close_col (str): column name for close (default 'close')
        min_periods (int|None): bars required before emitting (default: max(k_window, k_smooth, d_window))

    Returns:
        pd.DataFrame with columns:
            'k' : smoothed %K (0..100)
            'd' : %D = smoothed K (0..100)
    """
    category = "momentum"
    slug = "sto"
    name = "Stochastic Oscillator"
    indicator_type = IndicatorType.LINE

    def __init__(
        self,
        k_window: int = 14,
        k_smooth: int = 3,
        d_window: int = 3,
        method: str = "sma",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        min_periods: int | None = None,
        adjust: bool = False,  # for EMA only
    ):
        if k_window <= 0 or k_smooth <= 0 or d_window <= 0:
            raise ValueError("k_window, k_smooth, and d_window must be > 0")
        if method.lower() not in {"sma", "ema", "rma"}:
            raise ValueError("method must be one of {'sma','ema','rma'}")

        self.k_window = int(k_window)
        self.k_smooth = int(k_smooth)
        self.d_window = int(d_window)
        self.method = method.lower()
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.adjust = bool(adjust)

        default_mp = max(self.k_window, self.k_smooth, self.d_window)
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.high_col, self.low_col, self.close_col]

    # --- smoothing helpers ---
    def _sma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.rolling(window=n, min_periods=mp or n).mean()

    def _ema(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()

    def _rma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        # Wilder's smoothing
        return s.ewm(alpha=1 / n, adjust=False, min_periods=mp or n).mean()

    def _smooth(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        if self.method == "sma":
            return self._sma(s, n, mp)
        elif self.method == "ema":
            return self._ema(s, n, mp)
        else:
            return self._rma(s, n, mp)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df[self.high_col].astype(float)
        l = df[self.low_col].astype(float)
        c = df[self.close_col].astype(float)

        # Rolling highest high / lowest low
        hh = h.rolling(window=self.k_window, min_periods=self.k_window).max()
        ll = l.rolling(window=self.k_window, min_periods=self.k_window).min()

        # Raw %K (handle division-by-zero when hh == ll)
        rng = (hh - ll)
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_k = 100.0 * (c - ll) / rng
        raw_k = raw_k.where(rng != 0, 0.0).clip(lower=0.0, upper=100.0)

        # Smooth %K and %D
        k = self._smooth(raw_k, self.k_smooth, self.k_smooth)
        d = self._smooth(k, self.d_window, self.d_window)

        out = pd.DataFrame({"k": k, "d": d}, index=df.index)

        # Apply global min_periods mask for consistency with your framework
        if self.min_periods and self.min_periods > 0:
            valid = c.expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
