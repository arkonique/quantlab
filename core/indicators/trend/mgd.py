from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class MGD(Indicator):
    """
    McGinley Dynamic (MD) — John R. McGinley, Jr.

    Adaptive smoother that automatically adjusts to market speed:
        MGD_t = MGD_{t-1} + (Price_t - MGD_{t-1}) / (N * (Price_t / MGD_{t-1})^4)

    Notes:
    - Reduces whipsaws and lag compared to standard MAs.
    - Common N values: 10–20 (higher = smoother, slower).
    - The first valid value is seeded using an SMA(N) by default.
    """
    category = "trend"
    slug = "mgd"
    name = "McGinley Dynamic"
    indicator_type = IndicatorType.LINE
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        min_periods: int | None = None,
        seed: str = "sma",  # 'sma' or 'price'
    ):
        """
        Args:
            window (int): N in the McGinley formula (typ. 10–20).
            column (str): Input price column (e.g., 'close').
            min_periods (int | None): Minimum bars before emitting a value.
                                      Defaults to `window` for SMA-like behavior.
            seed (str): How to seed the first MGD value at the first valid index:
                        'sma' -> use SMA(window) at that index,
                        'price' -> use the raw price at that index.
        """
        if window <= 0:
            raise ValueError("window must be > 0")
        if seed not in {"sma", "price"}:
            raise ValueError("seed must be 'sma' or 'price'")

        self.window = int(window)
        self.column = column
        self.min_periods = window if min_periods is None else int(min_periods)
        self.seed = seed

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = self.window
        s = df[self.column].astype(float)

        # Seed value at first full window
        sma_n = s.rolling(window=n, min_periods=n).mean()
        seed_idx = sma_n.first_valid_index()
        mgd = pd.Series(np.nan, index=s.index, dtype="float64")

        if seed_idx is None:
            mgd.name = f"MGD({n})"
            return mgd

        seed_pos = s.index.get_loc(seed_idx)
        mgd.iat[seed_pos] = sma_n.loc[seed_idx] if self.seed == "sma" else s.loc[seed_idx]

        # Recursive calculation
        for i in range(seed_pos + 1, len(s)):
            prev = mgd.iat[i - 1]
            price = s.iat[i]
            if np.isnan(price) or np.isnan(prev):
                mgd.iat[i] = np.nan
                continue

            ratio = price / prev if prev != 0 else 1.0
            denom = n * (ratio ** 4) if np.isfinite(ratio) else n
            denom = denom if denom != 0 and np.isfinite(denom) else n
            mgd.iat[i] = prev + (price - prev) / denom

        # Apply min_periods mask
        if self.min_periods and self.min_periods > 0:
            mgd = mgd.where(s.expanding().count() >= self.min_periods)

        mgd.name = f"MGD({n})"
        return mgd
