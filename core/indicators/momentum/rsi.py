from ..base import Indicator, IndicatorType
import pandas as pd


class RSI(Indicator):
    """
    Relative Strength Index (RSI) — J. Welles Wilder (1978).

    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = (average gain) / (average loss)
    Uses Wilder's smoothing method (RMA) internally.

    Notes:
    - RSI measures the magnitude of recent price changes
      to evaluate overbought or oversold conditions.
    - Common thresholds: >70 (overbought), <30 (oversold).
    """
    category = "momentum"
    slug = "rsi"
    name = "Relative Strength Index"
    indicator_type = IndicatorType.LINE  # plotted as a line

    def __init__(
        self,
        window: int = 14,
        column: str = "close",
        min_periods: int | None = None,
    ):
        """
        Args:
            window (int): lookback period (commonly 14).
            column (str): which column to compute RSI on.
            min_periods (int | None): minimum periods for valid values.
                                      Defaults to `window` for consistency.
        """
        self.window = int(window)
        self.column = column
        self.min_periods = min_periods

    def required_columns(self):
        return [self.column]

    def _rma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        """Helper: Wilder's RMA smoothing."""
        return s.ewm(alpha=1 / n, adjust=False, min_periods=mp or n).mean()

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute RSI over the specified column.

        Returns:
            pd.Series: RSI values aligned to df.index.
        """
        n = self.window
        mp = self.min_periods if self.min_periods is not None else n
        s = df[self.column].astype(float)

        # Price changes
        delta = s.diff()

        # Separate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's smoothing (RMA)
        avg_gain = self._rma(gain, n, mp)
        avg_loss = self._rma(loss, n, mp)

        # Relative Strength (RS) and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        rsi.name = f"RSI({n})"
        return rsi
