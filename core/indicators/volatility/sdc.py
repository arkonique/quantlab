from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class SDC(Indicator):
    """
    Standard Deviation Channels (SDC)

    Methods:
      - 'reg' : Rolling OLS regression over `window` bars. mid = fitted value at the
                current bar; width = mult * std(residuals over window).
      - 'sma' : mid = SMA(window); width = mult * std(price over window).
      - 'ema' : mid = EMA(window); width = mult * std(price over window).

    Notes:
      - For 'reg', channels are parallel to the regression line (slope-aware).
      - For 'sma'/'ema', channels are horizontal relative to the mid at each bar.
      - `ddof=0` (population) by default; set `ddof=1` for sample std.
    """
    category = "volatility"
    slug = "sdc"
    name = "Standard Deviation Channels"
    indicator_type = IndicatorType.BANDS
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        mult: float = 2.0,
        method: str = "reg",          # 'reg' | 'sma' | 'ema'
        ddof: int = 0,
        adjust: bool = False,         # EMA adjust
        min_periods: int | None = None,
    ):
        if window <= 1:
            raise ValueError("window must be >= 2")
        if method not in {"reg", "sma", "ema"}:
            raise ValueError("SDC: method must be one of {'reg','sma','ema'}.")

        self.window = int(window)
        self.column = column
        self.mult = float(mult)
        self.method = method
        self.ddof = int(ddof)
        self.adjust = bool(adjust)
        self.min_periods = window if min_periods is None else int(min_periods)

        # Precompute OLS constants for 'reg'
        w = self.window
        self._sumx = w * (w - 1) / 2.0
        self._sumx2 = w * (w - 1) * (2 * w - 1) / 6.0
        self._denom = w * self._sumx2 - self._sumx**2  # > 0 since w>=2

    def required_columns(self):
        return [self.column]

    def _ema(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.ewm(span=n, adjust=self.adjust, min_periods=mp or n).mean()

    def _sma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        return s.rolling(window=n, min_periods=mp or n).mean()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column].astype(float).to_numpy()
        n = len(s)
        w = self.window

        mid = np.full(n, np.nan, dtype=float)
        upper = np.full(n, np.nan, dtype=float)
        lower = np.full(n, np.nan, dtype=float)

        if self.method == "reg":
            sumx = self._sumx
            sumx2 = self._sumx2
            denom = self._denom
            x = np.arange(w, dtype=float)

            for i in range(w - 1, n):
                ys = s[i - w + 1 : i + 1]
                sumy = float(np.sum(ys))
                sumxy = float(np.dot(x, ys))

                # OLS slope/intercept
                b = (w * sumxy - sumx * sumy) / denom
                a = (sumy - b * sumx) / w

                yhat_last = a + b * (w - 1)
                mid[i] = yhat_last

                resid = ys - (a + b * x)
                sigma = float(resid.std(ddof=self.ddof))

                width = self.mult * sigma
                upper[i] = yhat_last + width
                lower[i] = yhat_last - width

        else:
            # SMA/EMA mid, std of price over same window
            ps = pd.Series(s, index=df.index)
            if self.method == "sma":
                m = self._sma(ps, w, self.min_periods)
            else:
                m = self._ema(ps, w, self.min_periods)

            sigma = ps.rolling(window=w, min_periods=self.min_periods).std(ddof=self.ddof)

            mid[:] = m.to_numpy()
            width = self.mult * sigma.to_numpy()
            upper[:] = mid + width
            lower[:] = mid - width

        out = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower}, index=df.index)

        # Enforce min_periods
        if self.min_periods and self.min_periods > 0:
            valid = df[self.column].expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
