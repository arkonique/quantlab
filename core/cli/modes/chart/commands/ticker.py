from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple
from ....registry import Command

# TwelveData-accepted intervals we normalize to
_ALLOWED_INTERVALS = {
    "1min", "5min", "15min", "30min", "45min",
    "1h", "2h", "4h", "8h",
    "1day",
    "1week",
    "1month",
}

_FAKE_RE = re.compile(r"^FAKE\d*$", re.IGNORECASE)

def _normalize_interval(user_text: str) -> str:
    s = (user_text or "").strip().lower().replace(" ", "")
    if not s:
        return "1day"
    m = re.fullmatch(r"(?:(\d+))?([a-z]+)?", s)
    if not m:
        raise ValueError(f"Could not parse interval '{user_text}'.")
    num_str, unit = m.groups()
    if unit is None and num_str is not None:
        unit = "min"
    n = int(num_str) if num_str is not None else 1
    unit_map = {
        "m": "min", "min": "min", "mins": "min", "minute": "min", "minutes": "min",
        "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
        "d": "day", "day": "day", "days": "day",
        "w": "week", "wk": "week", "wks": "week", "week": "week", "weeks": "week",
        "mo": "month", "mon": "month", "mth": "month", "mths": "month",
        "month": "month", "months": "month",
    }
    unit_norm = unit_map.get(unit, unit or "day")
    if unit_norm == "min":
        normalized = f"{n}min"
    elif unit_norm == "h":
        normalized = f"{n}h"
    elif unit_norm == "day":
        normalized = f"{n}day"
    elif unit_norm == "week":
        if n != 1:
            raise ValueError("Weeks must be '1week' for TwelveData.")
        normalized = "1week"
    elif unit_norm == "month":
        if n != 1:
            raise ValueError("Months must be '1month' for TwelveData.")
        normalized = "1month"
    else:
        raise ValueError(f"Unrecognized interval unit '{unit}'.")
    if normalized not in _ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval '{normalized}'. Allowed: {', '.join(sorted(_ALLOWED_INTERVALS))}")
    return normalized

def _parse_real_ticker_args(arg_blob: str) -> Tuple[str, Optional[str], Optional[int]]:
    parts = [p for p in arg_blob.split(":") if p != ""]
    if not parts:
        raise ValueError("Missing symbol.")
    symbol = parts[0].upper()
    interval = _normalize_interval(parts[1]) if len(parts) >= 2 and parts[1] else None
    outputsize = int(parts[2]) if len(parts) >= 3 and parts[2] else None
    return symbol, interval, outputsize

def _parse_fake_ticker_blob(arg_blob: str) -> Tuple[str, Optional[str]]:
    if ":" in arg_blob:
        sym, path = arg_blob.split(":", 1)
        return sym.upper(), path.strip() or None
    return arg_blob.upper(), None

def _load_fake_df(file_path: str):
    import pandas as pd
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Fake data file not found: {file_path}")
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, sheet_name=0)
    else:
        raise ValueError("Fake data must be .csv, .xlsx, or .xls")
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fake data missing required columns: {', '.join(missing)}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise", utc=False)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    df.index.name = "datetime"
    return df

def _apply_real_and_invalidate_if_needed(state, symbol: str, interval: Optional[str], outputsize: Optional[int]) -> None:
    prev_sig = state.raw_sig
    state.is_fake = False
    state.fake_path = None
    state.ticker = symbol
    if interval:
        state.interval = interval
    if outputsize is not None:
        state.outputsize = outputsize
    new_sig = (state.ticker, state.interval, state.outputsize)
    if prev_sig != new_sig:
        state.raw_df = None
        state.raw_sig = None
    state.df = None
    state.indicator_cols = []
    state.derived_dirty = True

def _prompt_for_fake_file() -> Optional[str]:
    while True:
        try:
            path = input(
                "Enter file path for FAKE data (CSV/XLS/XLSX; columns: datetime, open, high, low, close, volume)\n"
                "(or type 'cancel' to abort): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not path:
            continue
        if path.lower() in ("cancel", "quit", "q", "exit"):
            return None
        return path

def _apply_fake_with_file(state, symbol: str, file_path: str) -> None:
    df = _load_fake_df(file_path)
    state.ticker = symbol.upper()
    state.is_fake = True
    state.fake_path = str(Path(file_path))
    state.raw_df = df
    state.raw_sig = (state.ticker, "FAKE", 0)  # special signature to avoid API fetch
    state.df = None
    state.indicator_cols = []
    state.derived_dirty = True

class Ticker(Command):
    name = "ticker"
    aliases = ["t"]
    mode = "chart"
    help = (
        "Set ticker (and optional interval/outputsize). Real: "
        "ticker AAPL[:interval[:outputsize]] | t AAPL | tAAPL[:interval[:outputsize]]. "
        "FAKE data: ticker FAKE[:/path/file.csv] | t FAKE /path/file.xlsx | tFAKE:/path/file.csv | FAKE1, FAKE2, ...\n"
        "If FAKE is provided without a file path, you'll be prompted to enter one."
    )

    def match_raw(self, line: str, state) -> bool:
        if state.mode != "chart":
            return False
        m = re.fullmatch(r"t([^\s]+)(?:\s+(.*))?", line.strip())
        if not m or line.startswith("ticker"):
            return False

        arg1 = m.group(1)
        rest = (m.group(2) or "").strip()

        if _FAKE_RE.match(arg1):
            symbol, path_from_blob = _parse_fake_ticker_blob(arg1)
            file_path = path_from_blob or (rest if rest else None)
            if not file_path:
                while True:
                    maybe = _prompt_for_fake_file()
                    if maybe is None:
                        print("Canceled. Ticker unchanged.")
                        return True
                    try:
                        _apply_fake_with_file(state, symbol, maybe)
                        print(f"Ticker set to {state.ticker} (FAKE, file={state.fake_path})")
                        return True
                    except Exception as e:
                        print(f"Error loading fake data: {e}")
            else:
                try:
                    _apply_fake_with_file(state, symbol, file_path)
                    print(f"Ticker set to {state.ticker} (FAKE, file={state.fake_path})")
                except Exception as e:
                    print(f"Fake ticker error: {e}")
            return True

        try:
            symbol, interval, outputsize = _parse_real_ticker_args(arg1)
            _apply_real_and_invalidate_if_needed(state, symbol, interval, outputsize)
            print(f"Ticker set to {state.ticker} (interval={state.interval}, outputsize={state.outputsize})")
        except Exception as e:
            print(f"Ticker parse error: {e}")
        return True

    def run(self, args: List[str], state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return
        if not args:
            print(self.help)
            return

        if _FAKE_RE.match(args[0]):
            if len(args) == 1:
                blob = args[0]
                file_path = None
            else:
                blob = args[0]
                file_path = args[1]
            try:
                symbol, path_from_blob = _parse_fake_ticker_blob(blob)
            except Exception as e:
                print(f"Fake ticker error: {e}")
                return
            file_path = file_path or path_from_blob
            if not file_path:
                while True:
                    maybe = _prompt_for_fake_file()
                    if maybe is None:
                        print("Canceled. Ticker unchanged.")
                        return
                    try:
                        _apply_fake_with_file(state, symbol, maybe)
                        print(f"Ticker set to {state.ticker} (FAKE, file={state.fake_path})")
                        return
                    except Exception as e:
                        print(f"Error loading fake data: {e}")
            else:
                try:
                    _apply_fake_with_file(state, symbol, file_path)
                    print(f"Ticker set to {state.ticker} (FAKE, file={state.fake_path})")
                except Exception as e:
                    print(f"Fake ticker error: {e}")
            return

        if len(args) == 1:
            blob = args[0]
        else:
            blob = args[0]
            if len(args) >= 2 and args[1]:
                blob += f":{args[1]}"
            if len(args) >= 3 and args[2]:
                blob += f":{args[2]}"

        try:
            symbol, interval, outputsize = _parse_real_ticker_args(blob)
            _apply_real_and_invalidate_if_needed(state, symbol, interval, outputsize)
            print(f"Ticker set to {state.ticker} (interval={state.interval}, outputsize={state.outputsize})")
        except Exception as e:
            print(f"Ticker parse error: {e}")
