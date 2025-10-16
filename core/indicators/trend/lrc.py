from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class LRC(Indicator):
    """
    Linear Regression Channel (trendlines)

    Computes a rolling least-squares regression over the last `window` bars:
      - mid   : fitted value of the regression at the current bar
      - upper : mid + mult * std(residuals over window)
      - lower : mid - mult * std(residuals over window)

    Notes:
      - Uses classic OLS with x = 0..window-1 within each rolling window.
      - Channels are parallel to the regression line; band width is residual std.
      - ddof=0 by default (population std); set ddof=1 for sample std, TradingView-like.
    """
    category = "trend"
    slug = "lrc"
    name = "Linear Regression Channel"
    indicator_type = IndicatorType.BANDS

    def __init__(
        self,
        window: int = 100,
        column: str = "close",
        mult: float = 2.0,
        ddof: int = 0,
        min_periods: int | None = None,
    ):
        if window <= 1:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self.column = column
        self.mult = float(mult)
        self.ddof = int(ddof)
        self.min_periods = window if min_periods is None else int(min_periods)

        # Precompute constants for speed (sum x, sum x^2 for x=0..w-1)
        w = self.window
        self._sumx = w * (w - 1) / 2.0
        self._sumx2 = w * (w - 1) * (2 * w - 1) / 6.0
        self._denom = w * self._sumx2 - self._sumx ** 2  # > 0 since w>=2

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column].astype(float).to_numpy()
        n = len(s)
        w = self.window
        sumx = self._sumx
        sumx2 = self._sumx2
        denom = self._denom
        x = np.arange(w, dtype=float)

        mid = np.full(n, np.nan, dtype=float)
        upper = np.full(n, np.nan, dtype=float)
        lower = np.full(n, np.nan, dtype=float)

        for i in range(w - 1, n):
            ys = s[i - w + 1 : i + 1]
            sumy = float(np.sum(ys))
            sumxy = float(np.dot(x, ys))

            # OLS slope/intercept for the window
            b = (w * sumxy - sumx * sumy) / denom
            a = (sumy - b * sumx) / w

            # Fitted value at the last x in the window (current bar)
            yhat_last = a + b * (w - 1)
            mid[i] = yhat_last

            # Residuals and channel width
            yhat_all = a + b * x
            resid = ys - yhat_all
            sigma = float(resid.std(ddof=self.ddof))

            width = self.mult * sigma
            upper[i] = yhat_last + width
            lower[i] = yhat_last - width

        out = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower}, index=df.index)

        # Enforce min_periods (default == window)
        if self.min_periods and self.min_periods > 0:
            valid = df[self.column].expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
