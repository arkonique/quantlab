from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class PHL(Indicator):
    """
    Pivot High / Pivot Low (swing points)

    A Pivot High at index i occurs when:
        high[i] is strictly greater than highs of the preceding `left` bars
        AND strictly greater than highs of the following `right` bars.

    Pivot Low analogously with lows and strict less-than.

    Outputs:
        DataFrame with columns:
          - 'ph' : pivot-high value at pivot bars, NaN elsewhere
          - 'pl' : pivot-low  value at pivot bars, NaN elsewhere
    """
    category = "price"
    slug = "phl"
    name = "Pivot High-Low"
    indicator_type = IndicatorType.MARKERS  # discrete swing markers

    def __init__(
        self,
        left: int = 2,
        right: int = 2,
        high_col: str = "high",
        low_col: str = "low",
        min_periods: int | None = None,
    ):
        if left < 1 or right < 1:
            raise ValueError("PHL: 'left' and 'right' must be >= 1.")
        self.left = int(left)
        self.right = int(right)
        self.high_col = high_col
        self.low_col = low_col
        # by default, emit after we have enough bars to evaluate a full window
        default_mp = left + right + 1
        self.min_periods = default_mp if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.high_col, self.low_col]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df[self.high_col].astype(float).to_numpy()
        l = df[self.low_col].astype(float).to_numpy()
        n = len(h)

        ph = np.full(n, np.nan, dtype=float)
        pl = np.full(n, np.nan, dtype=float)

        L, R = self.left, self.right

        for i in range(n):
            li = i - L
            ri = i + R
            if li < 0 or ri >= n:
                continue

            # windows excluding the center bar
            h_left = h[li:i]
            h_right = h[i + 1 : ri + 1]
            l_left = l[li:i]
            l_right = l[i + 1 : ri + 1]

            # Pivot High: strictly greater than all neighbors
            if h_left.size and h_right.size and np.all(h[i] > h_left) and np.all(h[i] > h_right):
                ph[i] = h[i]

            # Pivot Low: strictly less than all neighbors
            if l_left.size and l_right.size and np.all(l[i] < l_left) and np.all(l[i] < l_right):
                pl[i] = l[i]

        out = pd.DataFrame({"high": ph, "low": pl}, index=df.index)

        # Apply min_periods: avoid emitting near edges
        if self.min_periods and self.min_periods > 0:
            valid = df[self.high_col].expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
