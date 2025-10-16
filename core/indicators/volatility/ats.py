from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class ATS(Indicator):
    """
    ATR Trailing Stops (ATS)

    Idea:
        Build trailing stop lines using ATR:
          base_long  = src - mult * ATR
          base_short = src + mult * ATR

        Then "trail" them:
          long[i]  = max(base_long[i],  long[i-1])  if close[i] >  long[i-1]  else base_long[i]
          short[i] = min(base_short[i], short[i-1]) if close[i] < short[i-1] else base_short[i]

        'active' is the stop currently in effect:
          - if close >= short[i-1] -> use long[i]  (long regime)
          - if close <= long[i-1]  -> use short[i] (short regime)
          - else keep previous regime

    Params:
        atr_window (int): ATR lookback (default 14)
        mult (float): ATR multiplier (default 3.0)
        source (str): price column for offset ('close' by default)
        high_col/low_col/close_col: OHLC column names
        min_periods (int|None): bars required before emitting values (default = atr_window)
    """
    category = "volatility"
    slug = "ats"
    name = "ATR Trailing Stops"
    indicator_type = IndicatorType.BANDS

    def __init__(
        self,
        atr_window: int = 14,
        mult: float = 3.0,
        source: str = "close",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        min_periods: int | None = None,
    ):
        if atr_window <= 0:
            raise ValueError("atr_window must be > 0")
        if mult <= 0:
            raise ValueError("mult must be > 0")

        self.atr_window = int(atr_window)
        self.mult = float(mult)
        self.source = source
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.min_periods = atr_window if min_periods is None else int(min_periods)

    def required_columns(self):
        need = [self.high_col, self.low_col, self.close_col]
        if self.source not in need:
            need.append(self.source)
        return need

    # --- helpers ---
    def _rma(self, s: pd.Series, n: int, mp: int | None) -> pd.Series:
        """Wilder's smoothing for ATR."""
        return s.ewm(alpha=1 / n, adjust=False, min_periods=mp or n).mean()

    def _atr(self, h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
        prev_c = c.shift(1)
        tr = pd.concat([
            (h - l),
            (h - prev_c).abs(),
            (l - prev_c).abs()
        ], axis=1).max(axis=1)
        return self._rma(tr, n, n)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df[self.high_col].astype(float)
        l = df[self.low_col].astype(float)
        c = df[self.close_col].astype(float)
        src = df[self.source].astype(float)

        atr = self._atr(h, l, c, self.atr_window)

        base_long = src - self.mult * atr
        base_short = src + self.mult * atr

        n = len(df)
        long_ts = np.full(n, np.nan, dtype=float)
        short_ts = np.full(n, np.nan, dtype=float)
        active = np.full(n, np.nan, dtype=float)

        # Seed at first ATR value
        first_valid = atr.first_valid_index()
        if first_valid is None:
            return pd.DataFrame({"long": long_ts, "short": short_ts, "active": active}, index=df.index)

        i0 = df.index.get_loc(first_valid)
        long_ts[i0] = base_long.iat[i0]
        short_ts[i0] = base_short.iat[i0]
        # pick regime based on where price sits
        active[i0] = long_ts[i0] if c.iat[i0] >= short_ts[i0] else short_ts[i0]

        for i in range(i0 + 1, n):
            # Trail the long stop upwards only
            if c.iat[i] > long_ts[i - 1]:
                long_ts[i] = max(base_long.iat[i], long_ts[i - 1])
            else:
                long_ts[i] = base_long.iat[i]

            # Trail the short stop downwards only
            if c.iat[i] < short_ts[i - 1]:
                short_ts[i] = min(base_short.iat[i], short_ts[i - 1])
            else:
                short_ts[i] = base_short.iat[i]

            # Determine active stop (regime)
            prev_active = active[i - 1]
            # Switch to long regime if price clears the short stop
            if c.iat[i] >= short_ts[i - 1]:
                active[i] = long_ts[i]
            # Switch to short regime if price falls through the long stop
            elif c.iat[i] <= long_ts[i - 1]:
                active[i] = short_ts[i]
            else:
                # maintain regime: follow whichever side was active previously
                if np.isfinite(prev_active):
                    # decide which side we were on
                    was_long = abs(prev_active - long_ts[i - 1]) < abs(prev_active - short_ts[i - 1])
                    active[i] = long_ts[i] if was_long else short_ts[i]
                else:
                    # fallback if unknown: choose the closer stop
                    active[i] = long_ts[i] if (c.iat[i] - long_ts[i]) < (short_ts[i] - c.iat[i]) else short_ts[i]

        out = pd.DataFrame(
            {"long": long_ts, "short": short_ts, "active": active},
            index=df.index,
        )

        # Apply min_periods mask
        if self.min_periods and self.min_periods > 0:
            valid = c.expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
