from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class KEL(Indicator):
    """
    Keltner Channels (KEL)

    Classic form:
        mid   = EMA(column, window)          # optionally SMA or raw close
        TR_t  = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR_t = Wilder RMA(TR, atr_window)
        upper = mid + mult * ATR_t
        lower = mid - mult * ATR_t

    Defaults: window=20, atr_window=14, mult=2.0, method='ema'
    """
    category = "volatility"
    slug = "kel"
    name = "Keltner Channels"
    indicator_type = IndicatorType.BANDS
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,            # middle MA window
        atr_window: int = 14,        # ATR lookback
        mult: float = 2.0,           # ATR multiplier
        column: str = "close",       # source for mid when using SMA/EMA
        method: str = "ema",         # 'ema' | 'sma' | 'close'
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        min_periods: int | None = None,
        adjust: bool = False,        # EMA adjust
    ):
        self.window = int(window)
        self.atr_window = int(atr_window)
        self.mult = float(mult)
        self.column = column
        self.method = method.lower()
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.adjust = bool(adjust)

        if self.method not in {"ema", "sma", "close"}:
            raise ValueError("KEL: method must be one of {'ema','sma','close'}.")

        # Emit after both mid and ATR windows are satisfied
        default_mp = max(self.window if self.method in {"sma", "ema"} else 1, self.atr_window)
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        cols = [self.high_col, self.low_col, self.close_col]
        if self.method in {"sma", "ema"} and self.column not in cols:
            cols.append(self.column)
        return cols

    # --- helpers ---
    def _rma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        """Wilder's smoothing for ATR."""
        return s.ewm(alpha=1 / n, adjust=False, min_periods=mp or n).mean()

    def _mid(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        if self.method == "ema":
            return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()
        elif self.method == "sma":
            return s.rolling(window=n, min_periods=mp or n).mean()
        else:  # 'close'
            return s.astype(float)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df[self.high_col].astype(float)
        l = df[self.low_col].astype(float)
        c = df[self.close_col].astype(float)

        # True Range
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

        # ATR via Wilder RMA
        atr = self._rma(tr, self.atr_window, self.atr_window)

        # Middle line (EMA/SMA/close)
        price_for_mid = df[self.column].astype(float) if self.method in {"sma", "ema"} else c
        mid = self._mid(price_for_mid, self.window, self.window if self.method in {"sma", "ema"} else 1)

        upper = mid + self.mult * atr
        lower = mid - self.mult * atr

        out = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower}, index=df.index)

        # Apply min_periods mask so channels appear after sufficient data
        if self.min_periods and self.min_periods > 0:
            valid = c.expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
