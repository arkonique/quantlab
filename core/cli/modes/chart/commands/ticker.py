from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional, Tuple
from ....registry import Command

# TwelveData-accepted intervals we normalize to
_ALLOWED_INTERVALS = {
    "1min", "5min", "15min", "30min", "45min",
    "1h", "2h", "4h", "8h",
    "1day", "1week", "1month",
}

_FAKE_RE = re.compile(r"^FAKE\d*$", re.IGNORECASE)

# Accept ONLY hyphenated date/time:
#  - YYYY-MM-DD
#  - YYYY-MM-DD-HH
#  - YYYY-MM-DD-HH-MM
#  - YYYY-MM-DD-HH-MM-SS
_DATE_HYPHEN_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:-(\d{2})(?:-(\d{2})(?:-(\d{2}))?)?)?$"
)

def _looks_like_date(s: str) -> bool:
    s = s.strip()
    # strictly forbid ':' inside date tokens
    if ":" in s or " " in s or "T" in s:
        return False
    return bool(_DATE_HYPHEN_RE.match(s))

def _normalize_date_token(s: str) -> str:
    """
    Normalize hyphenated date tokens to:
      - 'YYYY-MM-DD'           (if only date)
      - 'YYYY-MM-DD HH:MM:SS'  (if time present)
    ONLY accepts:
      'YYYY-MM-DD'
      'YYYY-MM-DD-HH'
      'YYYY-MM-DD-HH-MM'
      'YYYY-MM-DD-HH-MM-SS'
    """
    s = s.strip()
    m = _DATE_HYPHEN_RE.match(s)
    if not m:
        raise ValueError(
            f"Unrecognized date format: '{s}'. Use YYYY-MM-DD[-HH[-MM[-SS]]] (no colons)."
        )
    y, mo, d, hh, mm, ss = m.groups()
    if hh is None:
        return f"{y}-{mo}-{d}"
    mm = mm or "00"
    ss = ss or "00"
    return f"{y}-{mo}-{d} {hh}:{mm}:{ss}"

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

# ---------- FAKE helpers ----------
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
    df["datetime"] = __import__("pandas").to_datetime(df["datetime"], errors="raise", utc=False)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    df.index.name = "datetime"
    return df

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
    # clear date-range/outsizing (irrelevant for FAKE)
    state.range_start = None
    state.range_end = None
    state.raw_df = df
    state.raw_sig = ("FAKE", state.ticker, state.fake_path)
    state.df = None
    state.indicator_cols = []
    state.derived_dirty = True

# ---------- REAL ticker parsing ----------
def _parse_real_ticker_args(arg_blob: str):
    """
    Returns a dict with either:
      {'mode': 'OUT', 'symbol': S, 'interval': I or None, 'outputsize': N or None}
    or
      {'mode': 'RANGE', 'symbol': S, 'start': A, 'end': B, 'interval': I or None}
    where A/B are normalized to 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
    """
    parts = [p for p in arg_blob.split(":") if p != ""]
    if not parts:
        raise ValueError("Missing symbol.")
    symbol = parts[0].upper()

    # RANGE forms (strict hyphenated date tokens):
    #   S:START:END
    #   S:START:END:INTERVAL
    if len(parts) >= 3 and _looks_like_date(parts[1]) and _looks_like_date(parts[2]):
        start = _normalize_date_token(parts[1])
        end = _normalize_date_token(parts[2])
        interval = None
        if len(parts) >= 4 and parts[3]:
            interval = _normalize_interval(parts[3])
        return {"mode": "RANGE", "symbol": symbol, "start": start, "end": end, "interval": interval}

    # OUT forms (legacy):
    #   S:INTERVAL
    #   S:INTERVAL:OUTPUTSIZE
    interval = _normalize_interval(parts[1]) if len(parts) >= 2 and parts[1] else None
    outputsize = int(parts[2]) if len(parts) >= 3 and parts[2] else None
    return {"mode": "OUT", "symbol": symbol, "interval": interval, "outputsize": outputsize}

def _apply_real_and_invalidate_if_needed(state, parsed: dict) -> None:
    prev_sig = state.raw_sig
    symbol = parsed["symbol"]
    state.is_fake = False
    state.fake_path = None
    state.ticker = symbol

    if parsed["mode"] == "RANGE":
        if parsed.get("interval"):
            state.interval = parsed["interval"]
        state.range_start = parsed["start"]
        state.range_end = parsed["end"]
        new_sig = ("RANGE", state.ticker, state.interval, state.range_start, state.range_end)
    else:
        if parsed.get("interval"):
            state.interval = parsed["interval"]
        if parsed.get("outputsize") is not None:
            state.outputsize = parsed["outputsize"]
        state.range_start = None
        state.range_end = None
        new_sig = ("OUT", state.ticker, state.interval, state.outputsize)

    if prev_sig != new_sig:
        state.raw_df = None
        state.raw_sig = None

    state.df = None
    state.indicator_cols = []
    state.derived_dirty = True

def _print_set_message(state):
    if state.range_start and state.range_end:
        print(f"Ticker set to {state.ticker} (range {state.range_start} → {state.range_end}, interval={state.interval})")
    else:
        print(f"Ticker set to {state.ticker} (interval={state.interval}, outputsize={state.outputsize})")

class Ticker(Command):
    name = "ticker"
    aliases = ["t"]
    mode = "chart"
    help = (
        "Set ticker.\n"
        "Real (outputsize): ticker AAPL[:interval[:outputsize]] | t AAPL | tAAPL[:interval[:outputsize]]\n"
        "Real (date range): ticker AAPL:START:END[:interval] | tAAPL:START:END[:interval]\n"
        "  START/END formats (no colons allowed):\n"
        "    - YYYY-MM-DD\n"
        "    - YYYY-MM-DD-HH\n"
        "    - YYYY-MM-DD-HH-MM\n"
        "    - YYYY-MM-DD-HH-MM-SS\n"
        "FAKE data: ticker FAKE[:/path/file.csv] | t FAKE /path/file.xlsx | tFAKE:/path/file.csv | FAKE1, FAKE2, ...\n"
        "  If FAKE is provided without a file path, you'll be prompted to enter one."
    )

    def match_raw(self, line: str, state) -> bool:
        if state.mode != "chart":
            return False
        m = re.fullmatch(r"t([^\s]+)(?:\s+(.*))?", line.strip())
        if not m or line.startswith("ticker"):
            return False

        arg1 = m.group(1)
        rest = (m.group(2) or "").strip()

        # FAKE branch
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

        # REAL (compact blob)
        try:
            parsed = _parse_real_ticker_args(arg1)
            _apply_real_and_invalidate_if_needed(state, parsed)
            _print_set_message(state)
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

        # FAKE route (space-variant)
        if _FAKE_RE.match(args[0]):
            blob = args[0]
            file_path = args[1] if len(args) > 1 else None
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

        # REAL route (space-separated variants):
        # - SYMBOL INTERVAL OUTPUTSIZE
        # - SYMBOL START END [INTERVAL]   (START/END must be hyphenated date/time; no colons)
        if len(args) == 1:
            blob = args[0]
        else:
            if len(args) >= 3 and _looks_like_date(args[1]) and _looks_like_date(args[2]):
                # normalize START/END then stitch colon form
                start_norm = _normalize_date_token(args[1])
                end_norm = _normalize_date_token(args[2])
                if len(args) >= 4:
                    blob = f"{args[0]}:{start_norm.replace(' ', ' ')}:{end_norm.replace(' ', ' ')}:{args[3]}"
                else:
                    blob = f"{args[0]}:{start_norm.replace(' ', ' ')}:{end_norm.replace(' ', ' ')}"
            else:
                # treat as OUT mode (interval/outputsize)
                blob = args[0]
                if len(args) >= 2 and args[1]:
                    blob += f":{args[1]}"
                if len(args) >= 3 and args[2]:
                    blob += f":{args[2]}"

        try:
            parsed = _parse_real_ticker_args(blob)
            _apply_real_and_invalidate_if_needed(state, parsed)
            _print_set_message(state)
        except Exception as e:
            print(f"Ticker parse error: {e}")
