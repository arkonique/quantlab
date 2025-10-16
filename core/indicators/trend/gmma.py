from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class GMMA(Indicator):
    """
    Guppy Multiple Moving Average (GMMA) — Daryl Guppy

    Two bundles of EMAs:
      Short-term:  3, 5, 8, 10, 12, 15
      Long-term:  30, 35, 40, 45, 50, 60

    Output:
      DataFrame with columns:
        s3, s5, s8, s10, s12, s15, l30, l35, l40, l45, l50, l60
      (each a separate EMA line)

    Notes:
      - Uses pandas ewm() with adjust=False by default (recursive weighting).
      - Each line uses its own window-specific min_periods, so it starts only
        after enough data for that window. You can still enforce a global
        min_periods via the constructor if you prefer masking early values.
    """
    category = "trend"
    slug = "gmma"
    name = "Guppy Multiple Moving Average"
    indicator_type = IndicatorType.LINE  # multiple line series

    def __init__(
        self,
        column: str = "close",
        short_windows: list[int] | None = None,
        long_windows: list[int] | None = None,
        adjust: bool = False,
        min_periods: int | None = None,
    ):
        """
        Args:
            column (str): Source price column (default 'close').
            short_windows (list[int]): Short-term EMA periods.
            long_windows (list[int]): Long-term EMA periods.
            adjust (bool): Passed to pandas ewm(); False = recursive weighting.
            min_periods (int|None): Optional global mask; if provided, values
                                    before this many observations are set NaN.
        """
        self.column = column
        self.short_windows = short_windows or [3, 5, 8, 10, 12, 15]
        self.long_windows = long_windows or [30, 35, 40, 45, 50, 60]
        self.adjust = bool(adjust)
        self.min_periods = min_periods  # global mask; per-line windows still apply

        # Basic validation
        for w in self.short_windows + self.long_windows:
            if w <= 0:
                raise ValueError("All GMMA windows must be > 0")

    def required_columns(self):
        return [self.column]

    def _ema(self, s: pd.Series, n: int) -> pd.Series:
        # per-line min_periods = n so each line starts after its own window
        return s.ewm(span=n, adjust=self.adjust, min_periods=n).mean()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        s = df[self.column].astype(float)

        out_cols: dict[str, pd.Series] = {}

        # Short-term bundle
        for n in self.short_windows:
            out_cols[f"s{n}"] = self._ema(s, n)

        # Long-term bundle
        for n in self.long_windows:
            out_cols[f"l{n}"] = self._ema(s, n)

        out = pd.DataFrame(out_cols, index=df.index)

        # Optional global min_periods mask (in addition to per-line windows)
        if self.min_periods is not None and self.min_periods > 0:
            mask = s.expanding().count() >= int(self.min_periods)
            out = out.where(mask, np.nan)

        return out
