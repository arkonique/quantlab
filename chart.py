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

def add_indicator_columns(df: pd.DataFrame, indicators: List[IndicatorSpec]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    added_cols: list[str] = []

    for spec in indicators:
        slug, inst, name, params = _build_indicator(spec)
        _ensure_required_columns(df, inst.required_columns(), slug)
        result = inst.compute(df)

        if isinstance(result, pd.Series):
            colname = _choose_colname(slug, result, name, params)
            df[colname] = result.reindex(df.index)
            added_cols.append(colname)

        elif isinstance(result, pd.DataFrame):
            base = _choose_colname(slug, result, name, params) or f"ind_{slug}"
            for c in result.columns:
                colname = f"{base}.{c}"
                df[colname] = result[c].reindex(df.index)
                added_cols.append(colname)
        else:
            warnings.warn(f"Indicator '{slug}' returned {type(result)}, not added as a column.")

    return df, added_cols

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
    


def plot_candlestick_chart(df: pd.DataFrame,
                           ticker: str,
                           filename: str,
                           indicators: List[IndicatorSpec] | None = None):
    """
    Plot OHLC candles + optional indicators.

    indicators: list of specs, each can be one of:
      - "sma"
      - Indicator instance (e.g., Indicator.create("sma", window=20))
      - {"slug": "sma", "params": {"window": 20, "column": "close"}, "name": "SMA(20)"}
    """
    # Base candlestick
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC"
            )
        ]
    )
    fig.update_layout(
        title=f'Candlestick chart for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    # Early exit if no indicators requested
    if not indicators:
        fig.write_html(filename)
        return

    # Add indicators
    for spec in indicators:
        slug, inst, name, _ = _build_indicator(spec)

        # Validate required columns
        _ensure_required_columns(df, inst.required_columns(), slug)

        # Compute the indicator
        result = inst.compute(df)

        # Route by indicator type (for now we implement LINE only)
        itype = getattr(inst.__class__, "type", "line")  # string alias
        # or: itype_enum = getattr(inst.__class__, "indicator_type", IndicatorType.LINE)

        if itype == "line":
            # Expect a pd.Series aligned to df.index
            if isinstance(result, pd.DataFrame):
                # If a DataFrame was returned, try to pick a sensible column
                series = result.iloc[:, 0]
                warnings.warn(f"Indicator '{slug}' returned a DataFrame; using first column '{result.columns[0]}' for line plot.")
            else:
                series = result

            # Align index & drop NaNs at the front to avoid gaps
            series = series.reindex(df.index).astype(float)

            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=name
                )
            )

        else:
            # Future: add LEVELS/BANDS/MARKERS rendering here
            warnings.warn(f"Indicator '{slug}' has type '{itype}', which is not yet implemented in plotting.")

    fig.write_html(filename)

def plot_candlestick_chart(
    df, ticker, filename, indicator_cols=None, theme="plotly_dark",
    connect_line=True, line_column="close",
):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"
    ))

    if connect_line and line_column in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[line_column], mode="lines",
            name=f"{line_column.capitalize()} (connect)",
            line=dict(width=1, dash="dot"), hoverinfo="skip",
            connectgaps=False,
        ))

    # draw indicator columns
    if indicator_cols:
        for col in indicator_cols:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], mode="lines", name=col, line=dict(width=2),
                    connectgaps=False,
                ))

    fig.update_layout(
        template=theme,
        title=dict(
            text=f"Candlestick Chart for {ticker}",
            x=0.5,
            xanchor="center",
            yanchor="top",
            pad=dict(t=15, b=10),
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)" if "dark" in theme else "rgba(255,255,255,0)",
        ),
        hovermode="x unified",
    )
    fig.write_html(filename)

    # remove margins from the saved HTML file body
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()

    # Remove the margins from the HTML body
    html = html.replace('<body>', '<body style="margin:0">')

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

# Example usage:
if __name__ == "__main__":
    # Get daily data for the last 1 year for AAPL
    discover_all()

    # Define what you want to compute
    indicators = [
        {"slug": "sma", "params": {"window": 20, "column": "close"}, "name": "SMA(20)"},
        {"slug": "sma", "params": {"window": 50, "column": "close"}, "name": "SMA(50)"},
        {"slug": "ema", "params": {"window": 20, "column": "close"}, "name": "EMA(20)"},
        {"slug": "ema", "params": {"window": 50, "column": "close"}, "name": "EMA(50)"},
        {"slug": "wma", "params": {"window": 20, "column": "close"}, "name": "WMA(20)"},
        {"slug": "wma", "params": {"window": 50, "column": "close"}, "name": "WMA(50)"},
        {"slug": "hma", "params": {"window": 20, "column": "close"}, "name": "HMA(20)"},
        {"slug": "hma", "params": {"window": 50, "column": "close"}, "name": "HMA(50)"},
        {"slug": "rma", "params": {"window": 20, "column": "close"}, "name": "RMA(20)"},
        {"slug": "rma", "params": {"window": 50, "column": "close"}, "name": "RMA(50)"},
    ]

    # Load your price data
    df = load_chart_data("AAPL", interval="1day", outputsize=252)

    # Add indicators as columns
    df, indicator_cols = add_indicator_columns(df, indicators)

    print(df.head())
    print("Indicator columns added:", indicator_cols)

    plot_candlestick_chart(
        df,
        "AAPL",
        "aapl_candles_with_indicators.html",
        indicator_cols=indicator_cols,   # use the ones actually added
    )