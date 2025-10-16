from ..base import Indicator, IndicatorType
import pandas as pd


class DMA(Indicator):
    """
    Displaced Moving Average (DMA).

    Computes a moving average (SMA/EMA/RMA) and shifts it by `displacement` bars.

    Parameters:
        window (int): Lookback for the base moving average.
        column (str): Price column to use (e.g., 'close').
        method (str): 'sma' | 'ema' | 'rma'  (default: 'sma').
        displacement (int): Bars to shift the MA.
            - Positive -> moves the curve forward in time (future bars) which
              introduces lookahead if used for trading rules.
            - Negative -> moves the curve backward (earlier bars).
        adjust (bool): Passed to EMA only (pandas ewm adjust parameter).
        min_periods (int | None): Minimum observations to produce a value.
                                  Defaults to `window` for SMA/RMA consistency.

    Notes:
        - Uses pandas rolling/ewm. RMA uses Wilder's smoothing (alpha=1/window).
        - Name: DMA-{METHOD}(window, displacement)
    """
    category = "trend"
    slug = "dma"
    name = "Displaced Moving Average"
    indicator_type = IndicatorType.LINE

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        method: str = "sma",
        displacement: int = 0,
        adjust: bool = False,
        min_periods: int | None = None,
    ):
        self.window = int(window)
        self.column = column
        self.method = str(method).lower()
        self.displacement = int(displacement)
        self.adjust = bool(adjust)
        self.min_periods = min_periods

        if self.method not in {"sma", "ema", "rma"}:
            raise ValueError("DMA: method must be one of {'sma','ema','rma'}.")

    def required_columns(self):
        return [self.column]

    # --- base MA helpers ---
    def _sma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.rolling(window=n, min_periods=mp or n).mean()

    def _ema(self, s: pd.Series, n: int, mp: int | None, adjust: bool) -> pd.Series:
        return s.ewm(span=n, adjust=adjust, min_periods=mp or n).mean()

    def _rma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        # Wilder's smoothing
        return s.ewm(alpha=1 / n, adjust=False, min_periods=mp or n).mean()

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = self.window
        mp = self.min_periods if self.min_periods is not None else n
        s = df[self.column].astype(float)

        if self.method == "sma":
            ma = self._sma(s, n, mp)
        elif self.method == "ema":
            ma = self._ema(s, n, mp, self.adjust)
        else:  # "rma"
            ma = self._rma(s, n, mp)

        # Displace the moving average
        dma = ma.shift(self.displacement)
        dma.name = f"DMA-{self.method.upper()}({n}, {self.displacement})"
        return dma
