from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class MAMA(Indicator):
    """
    MESA Adaptive Moving Average (MAMA) — John Ehlers, with companion FAMA.

    Outputs:
        - MAMA: fast, adaptive MA
        - FAMA: following/confirming MA (slower than MAMA)

    Parameters:
        fastlimit (float): upper bound on adaptivity (typ. 0.5)
        slowlimit (float): lower bound on adaptivity (typ. 0.05)
        column (str): input price column (e.g., 'close')
        min_periods (int|None): minimum bars before emitting values (default ~ 10)

    Notes:
        - Implements Ehlers' Hilbert-transform method to estimate the dominant cycle
          and adapt the smoothing coefficient each bar.
        - Returns a DataFrame with columns ["MAMA(n/a)", "FAMA(n/a)"] aligned to df.index.
        - For plotting systems that expect a Series, select one column (e.g., df["MAMA(...)"]).
    """
    category = "trend"
    slug = "mama"           # 4-char slug, as requested
    name = "MESA Adaptive Moving Average"
    indicator_type = IndicatorType.LINE  # plotted as lines
    plot_row = 0  # typically plotted on price chart

    def __init__(
        self,
        fastlimit: float = 0.5,
        slowlimit: float = 0.05,
        column: str = "close",
        min_periods: int | None = 10,
    ):
        if not (0 < slowlimit < fastlimit < 1):
            raise ValueError("Require 0 < slowlimit < fastlimit < 1 (e.g., 0.05 < 0.5).")
        self.fastlimit = float(fastlimit)
        self.slowlimit = float(slowlimit)
        self.column = column
        self.min_periods = int(min_periods) if min_periods is not None else None

    def required_columns(self):
        return [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        price = df[self.column].astype(float).to_numpy()
        n = len(price)

        # Pre-allocate arrays
        mama = np.full(n, np.nan, dtype=float)
        fama = np.full(n, np.nan, dtype=float)

        smooth = np.full(n, np.nan, dtype=float)
        detrender = np.full(n, np.nan, dtype=float)
        i1 = np.full(n, np.nan, dtype=float)
        q1 = np.full(n, np.nan, dtype=float)
        ji = np.full(n, np.nan, dtype=float)
        jq = np.full(n, np.nan, dtype=float)
        i2 = np.full(n, np.nan, dtype=float)
        q2 = np.full(n, np.nan, dtype=float)
        re = np.full(n, np.nan, dtype=float)
        im = np.full(n, np.nan, dtype=float)
        period = np.full(n, np.nan, dtype=float)
        phase = np.full(n, np.nan, dtype=float)
        dphase = np.full(n, np.nan, dtype=float)
        alpha = np.full(n, np.nan, dtype=float)

        # Constants per Ehlers' HT kernel
        # Smoothing (4-sample WMA-like): s = (p + 2p[-1] + 2p[-2] + p[-3]) / 6
        # HT kernel coeffs:
        a = 0.0962
        b = 0.5769

        # Reasonable initial period guess
        per = 10.0

        for t in range(n):
            # 4-sample smoothing
            if t >= 3:
                smooth[t] = (price[t] + 2*price[t-1] + 2*price[t-2] + price[t-3]) / 6.0
            else:
                smooth[t] = price[t]

            # Adaptive HT "constant"
            q = 0.075 * per + 0.54

            # Detrender via HT kernel with lags 2,4,6
            if t >= 6:
                detrender[t] = (a * smooth[t] + b * smooth[t-2] - b * smooth[t-4] - a * smooth[t-6]) * q
            else:
                detrender[t] = 0.0

            # In-phase/Quadrature components
            if t >= 3:
                i1[t] = detrender[t-3]
            else:
                i1[t] = 0.0

            if t >= 6:
                q1[t] = (a * detrender[t] + b * detrender[t-2] - b * detrender[t-4] - a * detrender[t-6]) * q
            else:
                q1[t] = 0.0

            # Advance I/Q by 90 degrees
            if t >= 3:
                ji[t] = i1[t-3]
                jq[t] = q1[t-3]
            else:
                ji[t] = 0.0
                jq[t] = 0.0

            # Homodyne discriminator
            i2[t] = i1[t] - jq[t]
            q2[t] = q1[t] + ji[t]

            # Smooth i2/q2
            if t >= 1:
                i2[t] = 0.2 * i2[t] + 0.8 * i2[t-1]
                q2[t] = 0.2 * q2[t] + 0.8 * q2[t-1]

                re[t] = 0.2 * (i2[t] * i2[t-1] + q2[t] * q2[t-1]) + 0.8 * (re[t-1] if np.isfinite(re[t-1]) else 0.0)
                im[t] = 0.2 * (i2[t] * q2[t-1] - q2[t] * i2[t-1]) + 0.8 * (im[t-1] if np.isfinite(im[t-1]) else 0.0)
            else:
                re[t] = 0.0
                im[t] = 0.0

            # Period estimation (protect against division edge cases)
            if re[t] != 0 and im[t] != 0:
                per_est = 2 * np.pi / np.arctan2(im[t], re[t])  # radians
                # Limit and smooth the period
                per_est = np.clip(per_est, 6.0, 50.0)
                per = 0.2 * per_est + 0.8 * per
            per = np.clip(per, 6.0, 50.0)
            period[t] = per

            # Phase and delta phase
            ph = np.degrees(np.arctan2(q1[t], i1[t])) if (i1[t] != 0 or q1[t] != 0) else 0.0
            phase[t] = ph
            if t >= 1:
                dp = phase[t-1] - phase[t]
                if dp < 1.0:
                    dp = 1.0
                if dp > 50.0:
                    dp = 50.0
                dphase[t] = dp
            else:
                dphase[t] = np.nan

            # Adaptive smoothing coefficient
            if np.isfinite(dphase[t]):
                a_t = self.fastlimit / dphase[t]
                a_t = min(max(a_t, self.slowlimit), self.fastlimit)
            else:
                a_t = self.slowlimit
            alpha[t] = a_t

            # MAMA / FAMA recursions
            if t == 0:
                mama[t] = price[t]
                fama[t] = price[t]
            else:
                mama[t] = a_t * price[t] + (1.0 - a_t) * mama[t-1]
                fama[t] = 0.5 * a_t * mama[t] + (1.0 - 0.5 * a_t) * fama[t-1]

        # Build result frame
        out = pd.DataFrame({
            f"MAMA": mama,
            f"FAMA": fama,
        }, index=df.index)

        # Apply min_periods mask for consistency with your other indicators
        if self.min_periods is not None and self.min_periods > 0:
            valid = pd.Series(np.arange(n), index=out.index).expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
