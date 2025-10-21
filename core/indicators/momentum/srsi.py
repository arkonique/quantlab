from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class SRSI(Indicator):
    """
    Stochastic RSI (SRSI)

    Steps:
      1) Compute RSI over `rsi_window` using Wilder's smoothing (RMA).
      2) %K_raw = 100 * (RSI - LL_k) / (HH_k - LL_k), where HH/LL are rolling
         max/min of RSI over `k_window`.
      3) Smooth %K_raw -> K with `k_smooth` and method ('sma'|'ema'|'rma').
      4) %D = smooth(K, d_window, method).

    Typical defaults: rsi_window=14, k_window=14, k_smooth=3, d_window=3.
    """
    category = "momentum"
    slug = "srsi"
    name = "Stochastic RSI"
    indicator_type = IndicatorType.LINE
    plot_row = 1  # typically plotted below price

    def __init__(
        self,
        rsi_window: int = 14,
        k_window: int = 14,
        k_smooth: int = 3,
        d_window: int = 3,
        method: str = "sma",     # 'sma' | 'ema' | 'rma' for smoothing
        column: str = "close",
        min_periods: int | None = None,
        adjust: bool = False,    # EMA adjust flag
    ):
        if rsi_window <= 0 or k_window <= 0 or k_smooth <= 0 or d_window <= 0:
            raise ValueError("All windows must be > 0")
        if method.lower() not in {"sma", "ema", "rma"}:
            raise ValueError("method must be one of {'sma','ema','rma'}")

        self.rsi_window = int(rsi_window)
        self.k_window = int(k_window)
        self.k_smooth = int(k_smooth)
        self.d_window = int(d_window)
        self.method = method.lower()
        self.column = column
        self.adjust = bool(adjust)

        # Conservative default: need enough bars to form RSI and then the k-window
        default_mp = self.rsi_window + self.k_window
        default_mp = max(default_mp, self.k_smooth, self.d_window)
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.column]

    # --- helpers ---
    def _rma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        """Wilder's smoothing."""
        return s.ewm(alpha=1 / n, adjust=False, min_periods=mp or n).mean()

    def _sma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.rolling(window=n, min_periods=mp or n).mean()

    def _ema(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()

    def _smooth(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        if self.method == "sma":
            return self._sma(s, n, mp)
        elif self.method == "ema":
            return self._ema(s, n, mp)
        else:
            return self._rma(s, n, mp)

    def _rsi(self, s: pd.Series, n: int) -> pd.Series:
        """RSI using Wilder's RMA, consistent with your RSI indicator."""
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = self._rma(gain, n, n)
        avg_loss = self._rma(loss, n, n)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        src = df[self.column].astype(float)

        # 1) RSI
        rsi = self._rsi(src, self.rsi_window)

        # 2) Stochastic of RSI
        hh = rsi.rolling(window=self.k_window, min_periods=self.k_window).max()
        ll = rsi.rolling(window=self.k_window, min_periods=self.k_window).min()
        rng = (hh - ll)
        with np.errstate(invalid="ignore", divide="ignore"):
            k_raw = 100.0 * (rsi - ll) / rng
        k_raw = k_raw.where(rng != 0, 0.0).clip(lower=0.0, upper=100.0)

        # 3) Smooth K, then 4) D
        K = self._smooth(k_raw, self.k_smooth, self.k_smooth)
        D = self._smooth(K, self.d_window, self.d_window)

        out = pd.DataFrame({"k": K, "d": D}, index=df.index)

        # Global min_periods mask (ensures full warmup across RSI + Stoch windows)
        if self.min_periods and self.min_periods > 0:
            valid = src.expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
