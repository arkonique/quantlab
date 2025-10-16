# Load last x time period data for a ticker

import os
import dotenv
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from core.indicators import discover_all, discover_category
from typing import Any, Dict, Iterable, List, Union
import warnings
from core.indicators.base import Indicator, IndicatorType
from dataclasses import dataclass

@dataclass
class IndicatorRenderSpec:
    indicator: Indicator                  # the actual instance used
    base: str                             # friendly base name, e.g., "Bollinger Bands(20)"
    columns: list[str]                    # the df columns created for this indicator
    indicator_type: IndicatorType | None  # from the class (LINE/BANDS/LEVELS/etc.)

IndicatorSpec = Union[
    str,                              # "sma"
    Indicator,                        # an already-constructed instance
    Dict[str, Any],                   # {"slug": "sma", "params": {...}, "name": "SMA(20)"}
]

def _build_indicator(spec: IndicatorSpec) -> tuple[str, Indicator, str, Dict[str, Any]]:
    """Normalize the user input spec into (slug, instance, name, params)."""
    if isinstance(spec, str):
        inst = Indicator.create(spec)
        return spec, inst, getattr(inst, "slug", spec), {}
    if isinstance(spec, Indicator):
        slug = getattr(spec, "slug", spec.__class__.__name__.lower())
        return slug, spec, getattr(spec, "name", slug), {}
    if isinstance(spec, dict):
        slug = spec.get("slug")
        if not slug:
            raise ValueError("Indicator dict must include 'slug'.")
        params = spec.get("params", {}) or {}
        name = spec.get("name", slug)
        inst = Indicator.create(slug, **params)
        return slug, inst, name, params
    raise TypeError(f"Unsupported indicator spec: {type(spec)}")

def _ensure_required_columns(df: pd.DataFrame, cols: Iterable[str], slug: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Indicator '{slug}' requires columns {missing} but got {list(df.columns)}")
def _primary_period_like(params: Dict[str, Any]):
    for k in ("window", "span", "period", "n", "length"):
        if k in params:
            return k, params[k]
    return None, None


def _friendly_base_name(indicator, params: dict) -> str:
    """
    Build a human-friendly base label for Series/DF outputs, like:
      'Bollinger Bands(20)' or 'Kaufman Adaptive Moving Average'
    """
    name = getattr(indicator, "name", None) or getattr(indicator, "slug", None) or indicator.__class__.__name__
    # pick a primary period-like param if present
    for key in ("window", "period", "length", "er_window"):
        if key in params and params[key] is not None:
            return f"{name}({params[key]})"
    return name

def _compact_param_sig(params: Dict[str, Any], max_items: int = 3) -> str | None:
    if not params:
        return None
    # Exclude noisy or non-identifying keys if you want
    ignore = {"column"}  # you might keep it if useful
    items = [(k, params[k]) for k in sorted(params.keys()) if k not in ignore]
    if not items:
        return None
    items = items[:max_items]
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)
    return "_".join(f"{k}={_fmt(v)}" for k, v in items)

def _choose_colname(slug: str, result: pd.Series | pd.DataFrame, name: str | None, params: Dict[str, Any]) -> str | None:
    # 1) explicit human label
    if isinstance(name, str) and name.strip():
        return name.strip() if isinstance(result, pd.Series) else None  # for DF, we suffix per-column later

    # 2) series-provided name
    if isinstance(result, pd.Series) and (getattr(result, "name", None)):
        return str(result.name)

    # 3) machine-friendly fallback
    period_key, period_val = _primary_period_like(params)
    if period_key is not None:
        return f"ind_{slug}_{period_val}"

    sig = _compact_param_sig(params)
    if sig:
        return f"ind_{slug}_{sig}"

    return f"ind_{slug}"

def add_indicator_columns(
    df: pd.DataFrame,
    indicators: List[IndicatorSpec]
) -> tuple[pd.DataFrame, list[str], list[IndicatorRenderSpec]]:
    """
    Apply indicators and append their outputs to the DataFrame.

    Returns:
        df_out, added_cols, render_specs
          - df_out: DataFrame with added columns
          - added_cols: flat list of all column names added
          - render_specs: structured mapping of each indicator instance to the column(s) it produced
    """
    df = df.copy()
    added_cols: list[str] = []
    render_specs: list[IndicatorRenderSpec] = []

    for spec in indicators:
        slug, inst, name, params = _build_indicator(spec)
        _ensure_required_columns(df, inst.required_columns(), slug)
        result = inst.compute(df)

        # Friendly base name used for labeling
        base = _friendly_base_name(inst, params)
        produced: list[str] = []

        if isinstance(result, pd.Series):
            s = result.reindex(df.index).copy()
            s.name = base  # force friendly label
            df[s.name] = s
            produced.append(s.name)
            added_cols.append(s.name)

        elif isinstance(result, pd.DataFrame):
            out = result.reindex(df.index).copy()

            def _subkey(col: str) -> str:
                try:
                    return str(col).split(".")[-1]
                except Exception:
                    return str(col)

            renamed_cols = {c: f"{base}.{_subkey(c)}" for c in out.columns}
            out = out.rename(columns=renamed_cols)
            for c in out.columns:
                df[c] = out[c]
                produced.append(c)
                added_cols.append(c)
        else:
            warnings.warn(f"Indicator '{slug}' returned {type(result)}, not added as a column.")
            continue

        render_specs.append(IndicatorRenderSpec(
            indicator=inst,
            base=base,
            columns=produced,
            indicator_type=getattr(inst, "indicator_type", None)
        ))

    return df, added_cols, render_specs



dotenv.load_dotenv()

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

def load_chart_data(ticker, interval="1min", outputsize=30):

    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if "values" in data:
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        df = df.sort_index()              # ascending=True by default
        df = df.astype(float)
        return df
    else:
        raise ValueError(f"Error fetching data: {data.get('message', 'Unknown error')}")
    

# Map TwelveData-style interval strings to pandas Timedelta for validation
_INTERVAL_TO_DELTA = {
    "1min":  pd.Timedelta(minutes=1),
    "5min":  pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "45min": pd.Timedelta(minutes=45),
    "1h":    pd.Timedelta(hours=1),
    "2h":    pd.Timedelta(hours=2),
    "4h":    pd.Timedelta(hours=4),
    "8h":    pd.Timedelta(hours=8),
    "1day":  pd.Timedelta(days=1),
    "1week": pd.Timedelta(weeks=1),
    "1month": pd.Timedelta(days=28),  # lower bound sanity check (months vary)
}

_SUPPORTED_INTERVALS = set(_INTERVAL_TO_DELTA.keys())

def _parse_dt_any(x) -> pd.Timestamp:
    """
    Accepts:
      - str in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
      - datetime / pandas Timestamp (naive or tz-aware)
    Returns a pandas.Timestamp (tz-naive, second precision).
    """
    if x is None:
        raise ValueError("start_date and end_date must be provided.")
    ts = pd.to_datetime(x)
    # If tz-aware, convert to UTC then drop tz (TwelveData expects local-like strings)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC") if hasattr(ts, "tz_convert") else ts.tz_localize(None)
        ts = ts.tz_localize(None)
    # Round down to seconds for stable string formatting
    return ts.floor("S")

def _fmt_td_api(ts: pd.Timestamp) -> str:
    # Twelve Data accepts 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
    # Always send full resolution to seconds.
    return ts.strftime("%Y-%m-%d %H:%M:%S")

def load_duration_data(
    ticker: str,
    interval: str = "1min",
    start_date=None,
    end_date=None,
):
    """
    Fetch OHLCV for `ticker` between [start_date, end_date] at `interval`.

    Args
    ----
    ticker : str
    interval : str
        One of: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 8h, 1day, 1week, 1month
    start_date, end_date :
        str ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'), or datetime-like.

    Returns
    -------
    pd.DataFrame indexed by datetime (ascending) with float columns:
    open, high, low, close, volume.
    """
    # --- Input validation ---
    interval = str(interval).lower().strip()
    if interval not in _SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {sorted(_SUPPORTED_INTERVALS)}"
        )

    start_ts = _parse_dt_any(start_date)
    end_ts   = _parse_dt_any(end_date)

    if end_ts <= start_ts:
        raise ValueError(f"end_date ({end_ts}) must be after start_date ({start_ts}).")

    # Check that the requested span is at least one bar long
    min_span = _INTERVAL_TO_DELTA[interval]
    if (end_ts - start_ts) < min_span:
        raise ValueError(
            f"Requested duration {end_ts - start_ts} is shorter than one '{interval}' bar "
            f"({min_span})."
        )

    # --- Build API request ---
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ticker,
        "interval": interval,
        "start_date": _fmt_td_api(start_ts),
        "end_date": _fmt_td_api(end_ts),
        "apikey": TWELVE_DATA_API_KEY,
        # You can add "timezone": "America/Toronto" or similar if you prefer.
    }

    resp = requests.get(base_url, params=params, timeout=30)
    # Network / HTTP sanity
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"TwelveData HTTP error {resp.status_code}: {resp.text}") from e

    data = resp.json()

    # TwelveData sometimes nests the payload; handle both common shapes.
    # Expected success shape: {"meta": {...}, "values": [ { "datetime": "...", "open": "...", ... }, ... ] }
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected TwelveData response shape (not a JSON object).")

    if "status" in data and data.get("status") == "error":
        # Common error payloads: {"status":"error","message":"..."}
        raise ValueError(f"TwelveData error: {data.get('message','Unknown error')}")

    if "values" not in data or not isinstance(data["values"], list) or len(data["values"]) == 0:
        # Some errors are embedded differently:
        msg = data.get("message") or data.get("note") or "No 'values' returned."
        raise ValueError(f"Error fetching data: {msg}")

    df = pd.DataFrame(data["values"])

    # Standardize columns & dtypes
    if "datetime" not in df.columns:
        raise RuntimeError("TwelveData response missing 'datetime' in values.")

    # Coerce datetimes and numeric columns
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()  # ascending

    # The API returns strings for numbers; coerce safely
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Some intervals may omit 'volume'; create if absent
            if col == "volume":
                df[col] = pd.NA

    # Clip to the exact requested window (server-side may include edges)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    # Ensure strictly increasing index and float dtype where possible
    df = df.astype(
        {c: "float64" for c in ("open", "high", "low", "close") if c in df.columns}
    )

    return df

def _marker_style_for_suffix(suffix: str) -> dict:
    """
    Map a column suffix to a Plotly marker style.
    Recognizes: high, low, neutral, and any Plotly marker symbol names.
    Returns: {"symbol": str, "color": str, "label": str, "size": int}
    """
    s = (suffix or "").strip().lower()

    # canonical special cases
    if s == "low":
        return {"symbol": "triangle-up", "color": "#16a34a", "label": "low", "size": 9}   # green
    if s == "high":
        return {"symbol": "triangle-down", "color": "#dc2626", "label": "high", "size": 9}  # red
    if s == "neutral":
        return {"symbol": "circle", "color": "#9ca3af", "label": "neutral", "size": 8}     # gray

    # known plotly marker symbols (partial but common set; extend as needed)
    plotly_symbols = {
        "circle","circle-open","circle-dot","circle-open-dot",
        "square","square-open","square-dot","square-open-dot",
        "diamond","diamond-open","diamond-dot","diamond-open-dot",
        "cross","x","x-thin","cross-thin",
        "triangle-up","triangle-up-open","triangle-up-dot","triangle-up-open-dot",
        "triangle-down","triangle-down-open","triangle-down-dot","triangle-down-open-dot",
        "triangle-left","triangle-right",
        "star","hexagon","hexagon2","hourglass","bowtie"
    }

    if s in plotly_symbols:
        # give a pleasant default color for generic symbols
        return {"symbol": s, "color": "#3b82f6", "label": s, "size": 8}  # blue

    # fallback
    return {"symbol": "circle", "color": "#94a3b8", "label": s or "marker", "size": 8}


def plot_candlestick_chart(
    df,
    ticker,
    filename,
    indicator_cols: list[str] | None = None,         # legacy path
    indicator_specs: list[IndicatorRenderSpec] | None = None,  # new path
    theme: str = "plotly_dark",
    connect_line: bool = True,
    line_column: str = "close",
):
    x_vals = pd.to_datetime(df.index).strftime("%Y-%m-%d %H:%M:%S")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x_vals, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"
    ))

    if connect_line and line_column in df.columns:
        fig.add_trace(go.Scatter(
            x=x_vals, y=df[line_column], mode="lines",
            name=f"{line_column.capitalize()} (connect)",
            line=dict(width=1, dash="dot"), hoverinfo="skip",
            connectgaps=False,
        ))

    # --- New: render by indicator object when provided ---
    if indicator_specs:
        for spec in indicator_specs:
            itype = spec.indicator_type
            cols = [c for c in spec.columns if c in df.columns]

            # BANDS (e.g., Bollinger upper/lower, fill between)
            if itype == IndicatorType.BANDS:
                # Try to find upper/lower/mid by suffix
                upper = next((c for c in cols if c.endswith(".upper")), None)
                lower = next((c for c in cols if c.endswith(".lower")), None)
                mid   = next((c for c in cols if c.endswith(".mid")),   None)

                # Fallback: if no canonical names, just draw all as separate lines
                if not (upper and lower):
                    for c in cols:
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=df[c], mode="lines", name=c, connectgaps=False
                        ))
                else:
                    # draw lower first
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=df[lower], mode="lines",
                        name=lower, connectgaps=False
                    ))
                    # then upper, filled to previous
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=df[upper], mode="lines",
                        name=upper, fill="tonexty", opacity=0.2, connectgaps=False
                    ))
                    if mid and mid in df.columns:
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=df[mid], mode="lines",
                            name=mid, line=dict(width=2, dash="dash"), connectgaps=False
                        ))

            # LEVELS (horizontal levels / zones)
            elif itype == IndicatorType.LEVELS:
                for c in cols:
                    s = df[c].astype(float)                    # ensure float (NaN-friendly)
                    valid = s.notna()
                    changes = s.ne(s.shift()) | ~valid         # indices where value changes or is invalid
                    y = s.mask(changes)                        # set change points to NaN -> breaks the line

                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y,
                        mode="lines",
                        name=c,                                 # exactly one legend entry per column
                        line=dict(width=1),
                        connectgaps=False,                      # don’t bridge NaNs
                        line_shape="hv",                        # horizontal then vertical (vertical suppressed by NaN)
                        hoverinfo="skip"                        # hover only on candles
                    ))

            elif itype == IndicatorType.MARKERS:
                # Each column is NaN except at the event; plot-only those points.
                for c in cols:
                    suffix = c.split(".")[-1] if "." in c else c
                    style = _marker_style_for_suffix(suffix)

                    # We purposely don't "connectgaps": NaNs ensure only event points show.
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=df[c],
                        mode="markers",
                        name=f"{spec.base} · {style['label']}",
                        marker=dict(
                            symbol=style["symbol"],
                            size=style["size"],
                            color=style["color"],
                            line=dict(width=0)  # clean, no outline; adjust if you want contrast
                        ),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Time: %{x}<br>"
                            "Price: %{y:.4f}<extra></extra>"
                        ),
                        text=[f"{spec.base} · {style['label']}"] * len(df),
                    ))

            # TABLE/SIGNAL or default LINE -> line series per column
            else:
                for c in cols:
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=df[c], mode="lines", name=c, connectgaps=False
                    ))

    # --- Legacy: flat column list still works ---
    elif indicator_cols:
        for col in indicator_cols:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=x_vals, y=df[col], mode="lines", name=col, connectgaps=False
                ))

    fig.update_layout(
        template=theme,
        title=dict(text=f"Candlestick Chart for {ticker}", x=0.5, xanchor="center", yanchor="top", pad=dict(t=15, b=10)),
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)" if "dark" in theme else "rgba(255,255,255,0)"),
        hovermode="x unified",
    )

    if filename.endswith((".html", ".htm")):
        fig.write_html(filename)
        with open(filename, "r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("<body>", '<body style="margin:0">')
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
    else:
        fig.update_layout(font=dict(size=30))
        fig.write_image(filename, width=3840, height=2160, scale=1)


# Example usage:
if __name__ == "__main__":
    # Get daily data for the last 1 year for AAPL
    discover_all()

    # Define what you want to compute
    indicators = [
        {"slug": "sma", "params": {"window": 20, "column": "close"}, "name": "SMA(20)"},
        {"slug": "ema", "params": {"window": 50, "column": "close"}, "name": "EMA(50)"},
        {"slug": "rsi", "params": {"window": 14, "column": "close"}, "name": "RSI(14)"},
        {"slug": "phl", "params": {"left": 3, "right": 3, "high_col": "high", "low_col": "low"}, "name": "Pivot High-Low(3,3)"},
        {"slug": "ats", "params": {"atr_window": 14, "mult": 3.0, "source": "close"}, "name": "ATR Trailing Stops(14,3.0)"}
    ]

    # Load your price data
    df = load_chart_data("AAPL", interval="2h", outputsize=252)

    # Add indicators as columns
    df, indicator_cols, render_specs = add_indicator_columns(df, indicators)

    print(df.head())
    print("Indicator columns added:", indicator_cols)

    print(render_specs[-2])

    plot_candlestick_chart(
        df,
        "AAPL",
        "aapl_candles_with_indicators22.html",
        indicator_specs=render_specs,
    )