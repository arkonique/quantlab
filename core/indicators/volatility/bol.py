from ..base import Indicator, IndicatorType
import pandas as pd


class BOL(Indicator):
    """
    Bollinger Bands (BOL)

    Middle Band: SMA or EMA of the selected column
    Upper Band : Middle + mult * rolling_std
    Lower Band : Middle - mult * rolling_std

    Args:
        window (int): lookback period (default 20)
        column (str): which column to use (default 'close')
        mult (float): standard deviation multiplier (default 2.0)
        method (str): 'sma' (classic) or 'ema'
        ddof (int): degrees of freedom for std (0=population, 1=sample)
        min_periods (int|None): required observations for value (default=window)
        adjust (bool): whether to adjust EMA (default=False)

    Returns:
        pd.DataFrame: with columns ['BOL.mid', 'BOL.upper', 'BOL.lower']
    """
    category = "volatility"
    slug = "bol"
    name = "Bollinger Bands"
    indicator_type = IndicatorType.BANDS  # plural
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        mult: float = 2.0,
        method: str = "sma",
        ddof: int = 0,
        min_periods: int | None = None,
        adjust: bool = False,
    ):
        self.window = int(window)
        self.column = column
        self.mult = float(mult)
        self.method = method.lower()
        self.ddof = int(ddof)
        self.min_periods = self.window if min_periods is None else int(min_periods)
        self.adjust = bool(adjust)

        if self.method not in {"sma", "ema"}:
            raise ValueError("BOL: method must be one of {'sma', 'ema'}.")

    def required_columns(self):
        return [self.column]

    def _mid(self, s: pd.Series) -> pd.Series:
        """Compute the middle band (SMA or EMA)."""
        n, mp = self.window, self.min_periods
        if self.method == "sma":
            return s.rolling(window=n, min_periods=mp).mean()
        else:
            return s.ewm(span=n, adjust=self.adjust, min_periods=mp).mean()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Bands.

        Returns:
            pd.DataFrame: Columns ['BOL.mid', 'BOL.upper', 'BOL.lower'] aligned to df.index.
        """
        s = df[self.column].astype(float)
        n, mp = self.window, self.min_periods

        mid = self._mid(s)
        std = s.rolling(window=n, min_periods=mp).std(ddof=self.ddof)

        upper = mid + self.mult * std
        lower = mid - self.mult * std

        out = pd.DataFrame(
            {
                "mid": mid,
                "upper": upper,
                "lower": lower,
            },
            index=df.index,
        )
        return out
