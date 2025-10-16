from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class PEV(Indicator):
    """
    Price Envelope (PEV)

    Classic moving-average envelope:
        mid   = SMA/EMA(column, window)
        upper = mid * (1 + up_pct)
        lower = mid * (1 - dn_pct)

    Notes:
    - Use symmetric `percent` OR asymmetric `up_percent` / `down_percent`.
    - If `method='close'`, the mid is the raw price (no smoothing).
    """
    category = "price"
    slug = "pev"
    name = "Price Envelope"
    indicator_type = IndicatorType.BANDS

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        method: str = "sma",          # 'sma' | 'ema' | 'close'
        percent: float | None = 0.02, # 2% symmetric default
        up_percent: float | None = None,
        down_percent: float | None = None,
        min_periods: int | None = None,
        adjust: bool = False,         # for EMA
    ):
        self.window = int(window)
        self.column = column
        self.method = method.lower()
        if self.method not in {"sma", "ema", "close"}:
            raise ValueError("PEV: method must be one of {'sma','ema','close'}.")

        # Resolve symmetric/asymmetric percentages
        if up_percent is None and down_percent is None:
            if percent is None:
                raise ValueError("Provide either percent or up/down_percent.")
            up_percent = down_percent = float(percent)
        self.up_percent = float(up_percent if up_percent is not None else percent)
        self.down_percent = float(down_percent if down_percent is not None else percent)

        self.adjust = bool(adjust)
        default_mp = self.window if self.method in {"sma", "ema"} else 1
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.column]

    # --- helpers ---
    def _mid(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        if self.method == "sma":
            return s.rolling(window=n, min_periods=mp or n).mean()
        elif self.method == "ema":
            return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()
        else:  # 'close'
            return s.astype(float)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column].astype(float)
        n, mp = self.window, self.min_periods

        mid = self._mid(s, n, mp)
        upper = mid * (1.0 + self.up_percent)
        lower = mid * (1.0 - self.down_percent)

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
