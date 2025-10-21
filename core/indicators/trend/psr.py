from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class PSR(Indicator):
    """
    Parabolic SAR (Stop and Reverse) — Welles Wilder.

    Parameters:
        step (float): AF (acceleration factor) increment per new extreme point. Default 0.02
        max_af (float): AF cap. Default 0.2
        start (str): 'auto' | 'long' | 'short'  (initial trend detection)
        high_col, low_col: columns for highs/lows
        min_periods (int|None): bars required before emitting SAR (default 2)

    Notes:
        - Uptrend update:  SAR_{t} = SAR_{t-1} + AF * (EP - SAR_{t-1}),  EP = highest high
          Bound: SAR_t <= min(low_{t-1}, low_{t-2})
          Reverse to downtrend if low_t < SAR_t; on reversal SAR_t := prior EP
        - Downtrend update: SAR_{t} = SAR_{t-1} + AF * (EP - SAR_{t-1}),  EP = lowest low
          Bound: SAR_t >= max(high_{t-1}, high_{t-2})
          Reverse to uptrend if high_t > SAR_t; on reversal SAR_t := prior EP
    """
    category = "trend"
    slug = "psr"
    name = "Parabolic SAR"
    indicator_type = IndicatorType.LINE
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        step: float = 0.02,
        max_af: float = 0.2,
        start: str = "auto",      # 'auto' | 'long' | 'short'
        high_col: str = "high",
        low_col: str = "low",
        min_periods: int | None = 2,
    ):
        if step <= 0 or max_af <= 0:
            raise ValueError("step and max_af must be positive.")
        if start not in {"auto", "long", "short"}:
            raise ValueError("start must be one of {'auto','long','short'}.")

        self.step = float(step)
        self.max_af = float(max_af)
        self.start = start
        self.high_col = high_col
        self.low_col = low_col
        self.min_periods = int(min_periods) if min_periods is not None else None

    def required_columns(self):
        return [self.high_col, self.low_col]

    def _detect_initial_trend(self, h: pd.Series, l: pd.Series) -> tuple[int, bool]:
        """
        Returns:
            start_idx (int): index position to start SAR updates
            uptrend (bool): True if initial trend is up
        """
        n = len(h)
        if self.start == "long":
            return 1, True
        if self.start == "short":
            return 1, False

        # 'auto': look for the first directional bar that breaks prior extremes
        for i in range(1, n):
            if h.iat[i] > h.iat[i - 1] and l.iat[i] >= l.iat[i - 1]:
                return i, True
            if l.iat[i] < l.iat[i - 1] and h.iat[i] <= h.iat[i - 1]:
                return i, False
        # Fallback if flat: assume uptrend from bar 1
        return 1, True

    def compute(self, df: pd.DataFrame) -> pd.Series:
        h = df[self.high_col].astype(float)
        l = df[self.low_col].astype(float)
        n = len(df)

        sar = np.full(n, np.nan, dtype=float)
        if n < 2:
            out = pd.Series(sar, index=df.index, name="Parabolic SAR")
            return out

        start_idx, up = self._detect_initial_trend(h, l)

        # Initialize EP (extreme point), SAR (initial), AF
        if up:
            ep = h.iloc[: start_idx + 1].max()  # highest high so far
            sar_val = l.iat[start_idx - 1]      # prior low
        else:
            ep = l.iloc[: start_idx + 1].min()  # lowest low so far
            sar_val = h.iat[start_idx - 1]      # prior high

        af = self.step

        # Write the first SAR
        sar[start_idx] = sar_val

        # Main loop
        for i in range(start_idx + 1, n):
            prev_sar = sar[i - 1]
            if np.isnan(prev_sar):
                prev_sar = sar_val

            # Update SAR toward EP
            sar_candidate = prev_sar + af * (ep - prev_sar)

            # Apply bounds (prevent SAR from intruding into the last 1–2 bars of the opposite extreme)
            if up:
                # SAR cannot be above the last two lows
                if i - 1 >= 0:
                    sar_candidate = min(sar_candidate, l.iat[i - 1])
                if i - 2 >= 0:
                    sar_candidate = min(sar_candidate, l.iat[i - 2])
            else:
                # SAR cannot be below the last two highs
                if i - 1 >= 0:
                    sar_candidate = max(sar_candidate, h.iat[i - 1])
                if i - 2 >= 0:
                    sar_candidate = max(sar_candidate, h.iat[i - 2])

            # Check for reversal
            if up and l.iat[i] < sar_candidate:
                # reverse to downtrend
                up = False
                sar_val = ep                     # on reversal, SAR = prior EP
                sar[i] = sar_val
                af = self.step                   # reset AF
                ep = l.iat[i]                    # new EP = current low
            elif (not up) and h.iat[i] > sar_candidate:
                # reverse to uptrend
                up = True
                sar_val = ep
                sar[i] = sar_val
                af = self.step
                ep = h.iat[i]
            else:
                # no reversal: accept candidate
                sar_val = sar_candidate
                sar[i] = sar_val

                # Update EP and AF in trend direction
                if up:
                    if h.iat[i] > ep:
                        ep = h.iat[i]
                        af = min(af + self.step, self.max_af)
                else:
                    if l.iat[i] < ep:
                        ep = l.iat[i]
                        af = min(af + self.step, self.max_af)

        out = pd.Series(sar, index=df.index, name="Parabolic SAR")

        # Enforce min_periods behavior
        if self.min_periods is not None and self.min_periods > 0:
            mask = pd.Series(np.arange(n), index=df.index).expanding().count() >= self.min_periods
            out = out.where(mask, np.nan)

        return out
