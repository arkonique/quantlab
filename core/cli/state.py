from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class ShellState:
    mode: Optional[str] = None

    # === user inputs (shared) ===
    ticker: Optional[str] = None
    interval: str = "1day"
    outputsize: int = 252
    indicators: List[Dict[str, Any]] = field(default_factory=list)

    # === optional date-range mode ===
    # when both are set (non-None), we load via load_duration_data()
    range_start: Optional[str] = None   # keep original user string
    range_end: Optional[str] = None

    # === FAKE data flags ===
    is_fake: bool = False
    fake_path: Optional[str] = None

    # === cache: RAW data (fetched/loaded ONCE per signature) ===
    # signature encodes whether it's "RANGE" or "OUTPUTSIZE" mode
    raw_df: Any = None
    raw_sig: Optional[Tuple] = None  # e.g. ("RANGE", ticker, interval, start, end) or ("OUT", ticker, interval, outputsize)

    # === cache: DERIVED data (raw_df + indicators) ===
    df: Any = None
    indicator_cols: List[str] = field(default_factory=list)
    derived_dirty: bool = True  # recompute from raw when True

    def reset_chart_state(self):
        self.ticker = None
        self.interval = "1day"
        self.outputsize = 252
        self.indicators.clear()

        self.range_start = None
        self.range_end = None

        self.is_fake = False
        self.fake_path = None

        self.raw_df = None
        self.raw_sig = None

        self.df = None
        self.indicator_cols = []
        self.derived_dirty = True

    def prompt_prefix(self) -> str:
        if self.mode == "chart":
            if self.ticker:
                return f"(chart:{self.ticker}) "
            return "(chart) "
        if self.mode:
            return f"({self.mode}) "
        return ""
