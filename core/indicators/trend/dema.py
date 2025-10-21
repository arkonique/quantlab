from ..base import Indicator, IndicatorType
import pandas as pd


class DEMA(Indicator):
    """
    Double Exponential Moving Average (DEMA) — Patrick Mulloy (1994).

    Formula:
        DEMA(n) = 2 * EMA(price, n) - EMA(EMA(price, n), n)

    Notes:
    - Uses pandas' ewm() for EMA.
    - By default, we set min_periods=window so the first non-NaN appears at index n-1,
      which keeps it consistent with indicators that require a full window (e.g., SMA/RMA).
      You can override this by passing min_periods=None to start from the first bar.
    """
    category = "trend"
    slug = "dema"
    name = "Double Exponential Moving Average"
    indicator_type = IndicatorType.LINE  # plotted as a line
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        adjust: bool = False,
        min_periods: int | None = None,
    ):
        """
        Args:
            window (int): lookback period n.
            column (str): which column to compute on (e.g., 'close').
            adjust (bool): passed to pandas ewm(); False -> recursive weighting.
            min_periods (int | None): required observations in window to produce a value.
                                      Defaults to `window` if None (consistent with SMA/RMA).
        """
        self.window = int(window)
        self.column = column
        self.adjust = bool(adjust)
        self.min_periods = min_periods

    def required_columns(self):
        return [self.column]

    def _ema(self, s: pd.Series, n: int, mp: int | None, adjust: bool) -> pd.Series:
        return s.ewm(span=n, adjust=adjust, min_periods=mp).mean()

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute DEMA over `self.column`.

        Returns:
            pd.Series: DEMA values aligned to df.index.
        """
        n = self.window
        s = df[self.column].astype(float)
        mp = self.min_periods if self.min_periods is not None else n

        ema1 = self._ema(s, n, mp, self.adjust)
        ema2 = self._ema(ema1, n, mp, self.adjust)
        dema = 2.0 * ema1 - ema2

        dema.name = f"DEMA({n})"
        return dema
