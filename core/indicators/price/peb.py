from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class PEB(Indicator):
    """
    Percentage Bands (PEB)

    Bands = middle line +/- (percent * middle line)

    Middle line options:
      - 'sma': SMA(column, window)
      - 'ema': EMA(column, window)
      - 'close': raw close (ignores window)

    Args:
        window (int): lookback for SMA/EMA middle (ignored if method='close')
        column (str): source column for SMA/EMA (default 'close')
        method (str): 'sma' | 'ema' | 'close'
        percent (float): band distance as a fraction (e.g., 0.02 = 2%)
        min_periods (int|None): defaults to window for SMA/EMA, else 1
        adjust (bool): EMA adjust flag
    """
    category = "price"
    slug = "peb"
    name = "Percentage Bands"
    indicator_type = IndicatorType.BANDS
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        method: str = "sma",
        percent: float = 0.02,   # 2%
        min_periods: int | None = None,
        adjust: bool = False,
    ):
        self.window = int(window)
        self.column = column
        self.method = method.lower()
        self.percent = float(percent)
        self.adjust = bool(adjust)

        if self.method not in {"sma", "ema", "close"}:
            raise ValueError("PEB: method must be one of {'sma','ema','close'}.")

        default_mp = self.window if self.method in {"sma", "ema"} else 1
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        # Need the source column for SMA/EMA; for 'close' method we just need 'close'
        need = [self.column] if self.method in {"sma", "ema"} else ["close"]
        return need

    # --- helpers ---
    def _mid(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        if self.method == "sma":
            return s.rolling(window=n, min_periods=mp or n).mean()
        elif self.method == "ema":
            return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()
        else:  # 'close'
            return s.astype(float)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column if self.method in {"sma", "ema"} else "close"].astype(float)
        n, mp = self.window, self.min_periods

        mid = self._mid(s, n, mp)
        upper = mid * (1.0 + self.percent)
        lower = mid * (1.0 - self.percent)

        out = pd.DataFrame(
            {
                "mid": mid,
                "upper": upper,
                "lower": lower,
            },
            index=df.index,
        )

        if self.min_periods and self.min_periods > 0:
            valid = s.expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
