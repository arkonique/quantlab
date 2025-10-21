from ..base import Indicator, IndicatorType
import pandas as pd


class TEMA(Indicator):
    """
    Triple Exponential Moving Average (TEMA) — Patrick Mulloy (1994).

    Formula:
        TEMA(n) = 3 * EMA(price, n)
                  - 3 * EMA(EMA(price, n), n)
                  + EMA(EMA(EMA(price, n), n), n)

    Notes:
    - Reduces lag compared to traditional EMA and DEMA.
    - Uses pandas' ewm() for EMA computation.
    - By default, min_periods=window for consistency with SMA/RMA;
      set to None if you want TEMA to start from the first bar.
    """
    category = "trend"
    slug = "tema"
    name = "Triple Exponential Moving Average"
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
            adjust (bool): passed to pandas ewm(); False = recursive weighting.
            min_periods (int | None): required observations for value.
                                      Defaults to window for SMA/RMA consistency.
        """
        self.window = int(window)
        self.column = column
        self.adjust = bool(adjust)
        self.min_periods = min_periods

    def required_columns(self):
        return [self.column]

    def _ema(self, s: pd.Series, n: int, mp: int | None, adjust: bool) -> pd.Series:
        """Helper: compute EMA."""
        return s.ewm(span=n, adjust=adjust, min_periods=mp).mean()

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute TEMA over `self.column`.

        Returns:
            pd.Series: TEMA values aligned to df.index.
        """
        n = self.window
        s = df[self.column].astype(float)
        mp = self.min_periods if self.min_periods is not None else n

        ema1 = self._ema(s, n, mp, self.adjust)
        ema2 = self._ema(ema1, n, mp, self.adjust)
        ema3 = self._ema(ema2, n, mp, self.adjust)

        tema = 3 * ema1 - 3 * ema2 + ema3
        tema.name = f"TEMA({n})"
        return tema
