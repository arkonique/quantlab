#!/usr/bin/env python3
"""
QuantLab Shell (v0.1)

Usage (example session):
    quantlab$ mode list
    quantlab$ mc
    quantlab$ tAAPL
    quantlab$ i sma:20
    quantlab$ i ema:50
    quantlab$ chart --out "aapl_candles_with_indicators.html"

Requirements:
  - Your project must have `chart.py` (with load_chart_data, add_indicator_columns, plot_candlestick_chart)
  - Your indicators package must be discoverable (e.g., `core.indicators` with Indicator registry)
  - Environment variable TWELVE_DATA_API_KEY set (for load_chart_data), or adjust chart.py to your data source
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    from chart import load_chart_data, add_indicator_columns, plot_candlestick_chart
except Exception as e:
    print("[QuantLab] Error: failed to import from chart.py. Make sure it's on PYTHONPATH.", file=sys.stderr)
    print(f"[QuantLab] Import error: {e}", file=sys.stderr)
    load_chart_data = add_indicator_columns = plot_candlestick_chart = None

try:
    from core.indicators import discover_all  # populates registry
except Exception as e:
    discover_all = None
    print("[QuantLab] Warning: core.indicators.discover_all not found. Indicators may not be available.", file=sys.stderr)

# --- Simple indicator spec parser ---
def parse_indicator_token(token: str) -> Dict[str, Any]:
    """
    Parse a token like 'sma:20' or 'ema:50' or 'rsi:14' into the indicator spec dict
    understood by chart.add_indicator_columns().
    Also accepts key=value pairs after a comma, e.g., 'sma:20,column=close'.
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty indicator token.")
    # allow forms: slug:window[,k=v[,k=v...]]
    parts = token.split(",", maxsplit=1)
    head = parts[0]
    extras_str = parts[1] if len(parts) == 2 else ""
    if ":" in head:
        slug, window_str = head.split(":", 1)
        slug = slug.strip().lower()
        window_str = window_str.strip()
        if not window_str.isdigit():
            raise ValueError(f"Window must be an integer in '{token}'")
        params: Dict[str, Any] = {"window": int(window_str), "column": "close"}
    else:
        # slug only, no window (indicator may not need a period)
        slug = head.strip().lower()
        params = {"column": "close"}  # default
    # parse extras key=value pairs
    if extras_str:
        for kv in extras_str.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                raise ValueError(f"Expected key=value in extras: '{kv}'")
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            # cast ints/floats when possible
            if re.fullmatch(r"-?\d+", v):
                v_cast: Any = int(v)
            elif re.fullmatch(r"-?\d+\.\d*", v):
                v_cast = float(v)
            else:
                v_cast = v
            params[k] = v_cast

    name = slug.upper()
    if "window" in params:
        name += f"({params['window']})"

    return {"slug": slug, "params": params, "name": name}

# --- Shell state ---

AVAILABLE_MODES = ["chart"]

@dataclass
class ShellState:
    mode: Optional[str] = None
    ticker: Optional[str] = None
    indicators: List[Dict[str, Any]] = field(default_factory=list)

    def reset_chart_state(self):
        self.ticker = None
        self.indicators.clear()

# --- Command handling ---

HELP_TEXT = """\
Commands:
  help                         Show this help
  mode list                    List available modes
  mode chart | mode c          Enter chart mode
  m c | mc                     Shorthands to enter chart mode
  t SYMBOL | ticker SYMBOL     Set ticker (chart mode only)
  tSYMBOL                      Compact ticker set, e.g., 'tAAPL' (chart mode only)
  i SPEC | indicator SPEC      Add indicator (e.g., 'i sma:20', 'i ema:50', 'i rsi:14')
                               You can append extras: 'i sma:20,column=close'
  clear i                      Clear all indicators
  show                         Show current state
  chart --out FILE.html        Render chart to HTML (requires ticker; indicators optional)
  quit | exit | q              Exit QuantLab
"""

PROMPT = "quantlab$ "

def in_chart_mode(state: ShellState) -> bool:
    return state.mode == "chart"

def handle_mode(tokens: List[str], state: ShellState):
    if len(tokens) == 1 or tokens[1] == "list":
        print("Available modes:", ", ".join(AVAILABLE_MODES))
        return
    val = tokens[1].lower()
    if val in ("chart", "c"):
        state.mode = "chart"
        print("Mode set to 'chart'.")
        return
    print(f"Unknown mode '{val}'. Try: mode list")

def handle_mode_short(tokens: List[str], state: ShellState):
    # supports 'm c' or 'mc'
    if len(tokens) == 2 and tokens[1].lower() == "c":
        state.mode = "chart"
        print("Mode set to 'chart'.")
        return
    if len(tokens) == 1 and tokens[0].lower() == "mc":
        state.mode = "chart"
        print("Mode set to 'chart'.")
        return
    print("Usage: 'm c' or 'mc' to enter chart mode.")

def handle_ticker(tokens: List[str], state: ShellState):
    if not in_chart_mode(state):
        print("Error: set mode to 'chart' first (use: mode chart | mc).")
        return
    if len(tokens) < 2:
        print("Usage: ticker SYMBOL  |  t SYMBOL  |  tSYMBOL")
        return
    symbol = tokens[1].upper()
    state.ticker = symbol
    print(f"Ticker set to {state.ticker}")

def handle_t_compact(line: str, state: ShellState):
    # pattern tAAPL (but NOT tickerAAPL)
    if not in_chart_mode(state):
        print("Error: set mode to 'chart' first (use: mode chart | mc).")
        return
    m = re.fullmatch(r"t([A-Za-z0-9\.\-\:_]+)", line.strip())
    if not m:
        print("Usage: tSYMBOL  (e.g., tAAPL)")
        return
    state.ticker = m.group(1).upper()
    print(f"Ticker set to {state.ticker}")

def handle_indicator(tokens: List[str], state: ShellState):
    if not in_chart_mode(state):
        print("Error: set mode to 'chart' first (use: mode chart | mc).")
        return
    if len(tokens) < 2:
        print("Usage: i slug:window[,k=v...]  (e.g., i sma:20  or  i rsi:14,smooth=3)")
        return
    spec_str = " ".join(tokens[1:]).strip()
    try:
        spec = parse_indicator_token(spec_str)
        state.indicators.append(spec)
        print(f"Indicator added: {spec['name']}")
    except Exception as e:
        print(f"Indicator parse error: {e}")

def handle_clear_indicators(state: ShellState):
    state.indicators.clear()
    print("Indicators cleared.")

def handle_show(state: ShellState):
    print(f"Mode   : {state.mode or '(none)'}")
    print(f"Ticker : {state.ticker or '(not set)'}")
    if state.indicators:
        print("Indicators:")
        for i, spec in enumerate(state.indicators, 1):
            print(f"  {i}. {spec.get('name')}  -> {spec}")
    else:
        print("Indicators: (none)")

def handle_chart(tokens: List[str], state: ShellState):
    if not in_chart_mode(state):
        print("Error: set mode to 'chart' first (use: mode chart | mc).")
        return
    # parse args: chart --out FILE.html
    out_file = None
    for i, tok in enumerate(tokens[1:], start=1):
        if tok == "--out" and i + 1 < len(tokens):
            out_file = tokens[i + 1]
            break
        if tok.startswith("--out="):
            out_file = tok.split("=", 1)[1]
            break
    if not out_file:
        print("Usage: chart --out FILE.html")
        return
    if not state.ticker:
        print("Error: ticker not set. Use: t SYMBOL  or  tSYMBOL")
        return
    if discover_all:
        discover_all()
    if not (load_chart_data and add_indicator_columns and plot_candlestick_chart):
        print("Error: chart helpers not available. Ensure chart.py is importable.")
        return
    try:
        df = load_chart_data(state.ticker, interval="1day", outputsize=252)
        df, indicator_cols = add_indicator_columns(df, state.indicators)
        plot_candlestick_chart(
            df,
            state.ticker,
            out_file,
            indicator_cols=indicator_cols,
            theme="plotly_dark",
        )
        print(f"Chart written to: {out_file}")
    except Exception as e:
        print(f"Chart error: {e}")

def dispatch(line: str, state: ShellState) -> bool:
    line = line.strip()
    if not line:
        return True
    # compact ticker form 'tAAPL' (but avoid 'tickerAAPL')
    if line.startswith("t") and not line.startswith("ticker") and re.fullmatch(r"t[A-Za-z0-9\.\-\:_]+", line):
        handle_t_compact(line, state)
        return True

    # tokenize for the rest
    try:
        tokens = shlex.split(line)
    except ValueError as e:
        print(f"Parse error: {e}")
        return True

    cmd = tokens[0].lower()

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "?"):
        print(HELP_TEXT)
        return True
    if cmd == "mode":
        handle_mode(tokens, state)
        return True
    if cmd == "m" or cmd == "mc":
        handle_mode_short(tokens, state)
        return True
    if cmd in ("ticker", "t"):
        handle_ticker(tokens, state)
        return True
    if cmd in ("indicator", "i"):
        handle_indicator(tokens, state)
        return True
    if cmd == "clear" and len(tokens) > 1 and tokens[1] in ("i", "indicators"):
        handle_clear_indicators(state)
        return True
    if cmd == "show":
        handle_show(state)
        return True
    if cmd == "chart":
        handle_chart(tokens, state)
        return True

    if cmd == "mode" and len(tokens) == 2 and tokens[1] == "list":
        handle_mode(tokens, state)
        return True

    print(f"Unknown command: {tokens[0]}. Type 'help' for commands.")
    return True

def main():
    state = ShellState()
    # greet
    # print("Welcome to QuantLab. Type 'help' for commands.")
    while True:
        try:
            # Build dynamic prompt
            if state.mode == "chart":
                if state.ticker:
                    prefix = f"(chart:{state.ticker}) "
                else:
                    prefix = "(chart) "
            elif state.mode:
                prefix = f"({state.mode}) "
            else:
                prefix = ""
            line = input(f"{prefix}quantlab$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        keep_going = dispatch(line, state)
        if not keep_going:
            break

if __name__ == "__main__":
    main()
