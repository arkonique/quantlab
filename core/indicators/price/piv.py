from ..base import Indicator, IndicatorType
import pandas as pd
import numpy as np


class PIV(Indicator):
    """
    Pivot Points (session-based levels)

    Supported methods:
      - 'classic'   : PP=(H+L+C)/3; R1=2PP-L; S1=2PP-H; R2=PP+(H-L); S2=PP-(H-L);
                      R3=H+2(PP-L); S3=L-2(H-PP)
      - 'fibonacci' : PP=(H+L+C)/3; Rk=PP+Fib_k*(H-L); Sk=PP-Fib_k*(H-L); k=1..3 with
                      Fib = [0.382, 0.618, 1.000]
      - 'woodie'    : PP=(H+L+2O)/4; then R/S like classic
      - 'camarilla' : R1=C+1.1*(H-L)/12; R2=C+1.1*(H-L)/6; R3=C+1.1*(H-L)/4; R4=C+1.1*(H-L)/2
                      S levels symmetrical (C minus ...)
      - 'demark'    : X = (H + L + 2*C) if C==O else (2*H + L + C) if C>O else (H + 2*L + C)
                      PP = X/4; R1 = X/2 - L; S1 = X/2 - H

    Behavior:
      - Levels for each bar are computed from the PREVIOUS period’s H/L/C(/O),
        where the period is defined by `freq` (e.g., daily pivots on intraday data).
      - Returns a DataFrame with columns:
          'pp','r1','r2','r3','r4','s1','s2','s3','s4'
        (some may be NaN depending on method).
    """
    category = "price"
    slug = "piv"
    name = "Pivot Points"
    indicator_type = IndicatorType.LEVELS  # horizontal levels/zones

    def __init__(
        self,
        method: str = "classic",
        freq: str = "D",             # pandas offset alias: 'D','W','M', etc.
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        open_col: str = "open",
        min_periods: int | None = None,
    ):
        self.method = method.lower()
        if self.method not in {"classic", "fibonacci", "woodie", "camarilla", "demark"}:
            raise ValueError("PIV: method must be one of {'classic','fibonacci','woodie','camarilla','demark'}.")
        self.freq = freq
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.open_col = open_col
        self.min_periods = min_periods

    def required_columns(self):
        cols = [self.high_col, self.low_col, self.close_col]
        if self.method in {"woodie", "demark"} and self.open_col not in cols:
            cols.append(self.open_col)
        return cols

    # --- per-period formulas ---
    def _compute_levels_row(self, row: pd.Series) -> pd.Series:
        H = float(row["H"])
        L = float(row["L"])
        C = float(row["C"])
        O = float(row["O"]) if "O" in row and pd.notna(row["O"]) else np.nan

        out = dict(pp=np.nan, r1=np.nan, r2=np.nan, r3=np.nan, r4=np.nan,
                   s1=np.nan, s2=np.nan, s3=np.nan, s4=np.nan)

        if self.method == "classic":
            pp = (H + L + C) / 3.0
            out["pp"] = pp
            out["r1"] = 2 * pp - L
            out["s1"] = 2 * pp - H
            out["r2"] = pp + (H - L)
            out["s2"] = pp - (H - L)
            out["r3"] = H + 2 * (pp - L)
            out["s3"] = L - 2 * (H - pp)

        elif self.method == "fibonacci":
            pp = (H + L + C) / 3.0
            rng = (H - L)
            out["pp"] = pp
            for k, fib in enumerate([0.382, 0.618, 1.0], start=1):
                out[f"r{k}"] = pp + fib * rng
                out[f"s{k}"] = pp - fib * rng

        elif self.method == "woodie":
            if np.isnan(O):
                O = C
            pp = (H + L + 2 * O) / 4.0
            out["pp"] = pp
            out["r1"] = 2 * pp - L
            out["s1"] = 2 * pp - H
            out["r2"] = pp + (H - L)
            out["s2"] = pp - (H - L)
            out["r3"] = H + 2 * (pp - L)
            out["s3"] = L - 2 * (H - pp)

        elif self.method == "camarilla":
            rng = (H - L)
            out["pp"] = C  # many platforms show close as the "basis"
            for k, div in zip([1, 2, 3, 4], [12.0, 6.0, 4.0, 2.0]):
                delta = 1.1 * rng / div
                out[f"r{k}"] = C + delta
                out[f"s{k}"] = C - delta

        else:  # 'demark'
            if np.isnan(O):
                O = C
            if C < O:
                X = H + 2 * L + C
            elif C > O:
                X = 2 * H + L + C
            else:
                X = H + L + 2 * C
            pp = X / 4.0
            out["pp"] = pp
            out["r1"] = X / 2.0 - L
            out["s1"] = X / 2.0 - H

        return pd.Series(out)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Group by session/period using pandas PeriodIndex of given freq
        pi = df.index.to_period(self.freq)

        H = df[self.high_col].astype(float).groupby(pi).max()
        L = df[self.low_col].astype(float).groupby(pi).min()
        C = df[self.close_col].astype(float).groupby(pi).last()

        if self.open_col in df.columns:
            O = df[self.open_col].astype(float).groupby(pi).first()
        else:
            # If open is missing, create an aligned NaN series
            O = pd.Series(index=H.index, dtype="float64")

        agg = pd.DataFrame({"H": H, "L": L, "C": C, "O": O})

        # Compute levels per period, then shift by 1 to use previous period levels
        levels = agg.apply(self._compute_levels_row, axis=1).astype(float)
        levels = levels.shift(1)

        # Map each row back to original timestamp index
        out = levels.reindex(pi).copy()
        out.index = df.index

        # Optional min_periods (default: show as soon as one prior period exists)
        if self.min_periods is not None and self.min_periods > 0:
            valid = pd.Series(np.arange(len(df)), index=df.index).expanding().count() >= self.min_periods
            out = out.where(valid, np.nan)

        return out
