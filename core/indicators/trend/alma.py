from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class ALMA(Indicator):
    """
    Arnaud Legoux Moving Average (ALMA)

    Parameters:
        window (int): lookback period (default 20)
        offset (float): center offset in [0, 1], typical 0.85
        sigma (float): controls Gaussian width (typical 6)

    Formula:
        m = offset × (window − 1)
        s = window / sigma
        w_i = exp(−(i − m)² / (2 × s²)), i = 0..window−1 (normalized to sum = 1)
        ALMA_t = Σ w_i × price_{t−window+1+i}

    Notes:
        - Smoother than SMA/EMA with reduced lag.
        - `offset` controls how far the center of mass is shifted toward recent data.
        - `sigma` controls smoothness (larger = smoother).
    """
    category = "trend"
    slug = "alma"
    name = "Arnaud Legoux Moving Average"
    indicator_type = IndicatorType.LINE
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        window: int = 20,
        column: str = "close",
        offset: float = 0.85,
        sigma: float = 6.0,
        min_periods: int | None = None,
    ):
        if window <= 1:
            raise ValueError("ALMA: window must be ≥ 2")
        if not (0.0 <= offset <= 1.0):
            raise ValueError("ALMA: offset must be within [0, 1]")
        if sigma <= 0:
            raise ValueError("ALMA: sigma must be > 0")

        self.window = int(window)
        self.column = column
        self.offset = float(offset)
        self.sigma = float(sigma)
        self.min_periods = window if min_periods is None else int(min_periods)

        # Pre-compute normalized Gaussian weights
        n = self.window
        m = self.offset * (n - 1)
        s = n / self.sigma
        i = np.arange(n, dtype=float)
        w = np.exp(-((i - m) ** 2) / (2.0 * (s ** 2)))
        w /= w.sum()
        self._weights = w.astype(float)

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.Series:
        s = df[self.column].astype(float)

        # Rolling weighted sum
        def _alma(x: np.ndarray) -> float:
            return float(np.dot(x, self._weights))

        alma = s.rolling(window=self.window, min_periods=self.min_periods).apply(_alma, raw=True)
        alma.name = f"ALMA({self.window},{self.offset},{self.sigma})"
        return alma
