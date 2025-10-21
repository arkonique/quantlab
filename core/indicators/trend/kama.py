from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class KAMA(Indicator):
    """
    Kaufman Adaptive Moving Average (KAMA) — Perry J. Kaufman.

    Steps:
        1) Efficiency Ratio (ER_t) = |price_t - price_{t-n}| / sum_{i=1..n} |price_{t-i+1} - price_{t-i}|
        2) Smoothing Constant (SC_t) = [ ER_t * (fastSC - slowSC) + slowSC ]^2
           where fastSC = 2 / (fast_window + 1), slowSC = 2 / (slow_window + 1)
        3) KAMA_t = KAMA_{t-1} + SC_t * (price_t - KAMA_{t-1})

    Defaults per common convention: n=10, fast=2, slow=30.
    We seed KAMA at the first valid index with the SMA of the initial ER window.
    """
    category = "trend"
    slug = "kama"
    name = "Kaufman Adaptive Moving Average"
    indicator_type = IndicatorType.LINE
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        er_window: int = 10,
        fast_window: int = 2,
        slow_window: int = 30,
        column: str = "close",
        min_periods: int | None = None,
    ):
        """
        Args:
            er_window (int): Lookback n for ER (common: 10).
            fast_window (int): Fast EMA equivalent period (common: 2).
            slow_window (int): Slow EMA equivalent period (common: 30).
            column (str): Input price column.
            min_periods (int | None): Minimum periods before emitting values.
                                      Defaults to `er_window` for consistency.
        """
        if er_window <= 0:
            raise ValueError("er_window must be > 0")
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("fast_window and slow_window must be > 0")

        self.er_window = int(er_window)
        self.fast_window = int(fast_window)
        self.slow_window = int(slow_window)
        self.column = column
        self.min_periods = er_window if min_periods is None else int(min_periods)

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = self.er_window
        s = df[self.column].astype(float)

        # Efficiency Ratio (ER)
        change = (s - s.shift(n)).abs()
        volatility = s.diff().abs().rolling(window=n, min_periods=n).sum()
        er = change / volatility
        er = er.clip(lower=0, upper=1)

        # Smoothing constant SC_t
        fast_sc = 2 / (self.fast_window + 1)
        slow_sc = 2 / (self.slow_window + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Initialize KAMA with SMA of first n points (at index n-1)
        kama = pd.Series(index=s.index, dtype="float64")
        first_valid = sc.first_valid_index()

        if first_valid is None:
            # Not enough data to compute ER/SC anywhere
            kama.name = f"KAMA({n},{self.fast_window},{self.slow_window})"
            return kama

        seed_idx = first_valid
        # Seed using SMA of the initial n-window ending at seed_idx
        seed_val = s.loc[:seed_idx].tail(n).mean()
        kama.iloc[s.index.get_loc(seed_idx)] = seed_val

        # Recursive calculation forward from seed
        idx_pos = s.index.get_loc(seed_idx)
        for i in range(idx_pos + 1, len(s)):
            prev = kama.iat[i - 1]
            if np.isnan(prev):
                # In case of gaps, backfill previous with last valid
                j = i - 1
                while j >= 0 and np.isnan(kama.iat[j]):
                    j -= 1
                prev = kama.iat[j] if j >= 0 else s.iat[i]

            sc_t = sc.iat[i]
            price_t = s.iat[i]
            if np.isnan(sc_t) or np.isnan(price_t):
                kama.iat[i] = np.nan
            else:
                kama.iat[i] = prev + sc_t * (price_t - prev)

        # Apply min_periods mask (emit values only after min_periods)
        if self.min_periods is not None and self.min_periods > 0:
            kama = kama.where(s.expanding().count() >= self.min_periods)

        kama.name = f"KAMA({n},{self.fast_window},{self.slow_window})"
        return kama
