from __future__ import annotations
from pathlib import Path
from typing import List
from ....registry import Command

# ===== helpers =====

def _current_sig(state):
    if not state.ticker:
        raise RuntimeError("Ticker not set.")
    return (state.ticker, state.interval, state.outputsize)

def _ensure_loaded(state) -> None:
    """
    - REAL: fetch raw once per (ticker, interval, outputsize).
    - FAKE: never call API; raw_df is set by ticker command.
    - DERIVED: rebuild from raw + indicators only when needed.
    """
    if state.mode != "chart":
        raise RuntimeError("Not in chart mode.")
    if not state.ticker:
        raise RuntimeError("Ticker not set. Use: t SYMBOL or ticker SYMBOL[:interval[:outputsize]].")

    if state.is_fake:
        if state.raw_df is None:
            src = state.fake_path or "(no file provided)"
            raise RuntimeError(f"FAKE ticker has no data loaded. Provide a file. Current: {src}")
    else:
        sig = _current_sig(state)
        if state.raw_df is None or state.raw_sig != sig:
            try:
                from core.indicators import discover_all
                discover_all()
            except Exception:
                pass
            from chart import load_chart_data
            state.raw_df = load_chart_data(state.ticker, interval=state.interval, outputsize=state.outputsize)
            state.raw_sig = sig
            state.df = None
            state.indicator_cols = []
            state.derived_dirty = True

    if state.df is None or state.derived_dirty:
        from chart import add_indicator_columns
        base = state.raw_df.copy(deep=True)  # avoid mutating raw
        state.df, state.indicator_cols = add_indicator_columns(base, state.indicators)
        state.derived_dirty = False

def _save_chart(state, file_path: str) -> None:
    from chart import plot_candlestick_chart
    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_candlestick_chart(
        state.df,
        state.ticker,
        str(out),
        indicator_cols=state.indicator_cols,
        theme="plotly_dark",
    )

# --- custom CSS injection for dataframe->HTML ---
# Fixed small typo: 'Motnserrat' -> 'Montserrat'
_DATAFRAME_EMBED_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
body {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #ccd7e8;
}

table {
  border-collapse: collapse;
  background-color: white;
  overflow: hidden;
  width: 500px;
  border-radius: 10px;
}

th, td {
  font-family: 'Montserrat', sans-serif;
  text-align: left;
  font-size: 12px;
  padding: 10px;
}

th {
  background-color: #7691ab;
  color: white;
}
</style>
"""

def _wrap_table_html_with_style(table_html: str, title: str = "Dataframe Export") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
{_DATAFRAME_EMBED_CSS}
</head>
<body>
{table_html}
</body>
</html>
"""

def _save_dataframe(state, file_path: str) -> None:
    import pandas as pd
    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = out.suffix.lower().lstrip(".")
    df = state.df

    if ext in ("csv",):
        df.to_csv(out, index=False)
    elif ext in ("xlsx", "xls"):
        try:
            df.to_excel(out, index=False)
        except Exception as e:
            raise RuntimeError(f"Excel writer not available. Install openpyxl or xlsxwriter. ({e})")
    elif ext in ("parquet",):
        try:
            df.to_parquet(out, index=False)
        except Exception as e:
            raise RuntimeError(f"Parquet writer not available. Install pyarrow. ({e})")
    elif ext in ("feather",):
        try:
            df.to_feather(out)
        except Exception as e:
            raise RuntimeError(f"Feather writer not available. Install pyarrow. ({e})")
    elif ext in ("json",):
        df.to_json(out, orient="records")
    elif ext in ("ndjson", "jsonl"):
        df.to_json(out, orient="records", lines=True)
    elif ext in ("html", "htm"):
        # Build a full HTML document and inject a <style> in <head>, then the table in <body>
        table_html = df.to_html(index=False)
        doc_html = _wrap_table_html_with_style(table_html, title=f"{state.ticker} dataframe")
        out.write_text(doc_html, encoding="utf-8")
    elif ext in ("md", "markdown"):
        try:
            md = df.to_markdown(index=False)  # requires tabulate
        except Exception as e:
            raise RuntimeError(f"Markdown export requires 'tabulate'. pip install tabulate. ({e})")
        out.write_text(md, encoding="utf-8")
    elif ext in ("pkl", "pickle"):
        df.to_pickle(out)
    else:
        raise RuntimeError(
            f"Unsupported dataframe format '.{ext}'. "
            "Try: csv, xlsx, xls, parquet, feather, json, ndjson/jsonl, html, md/markdown, pkl/pickle."
        )

# ===== commands =====

class Output(Command):
    name = "output"
    aliases = ["o"]
    mode = "chart"
    help = (
        "Universal exporter.\n"
        "Usage:\n"
        "  output --chart FILE     | output chart FILE      (save chart: html/png/svg/pdf)\n"
        "  output --dataframe FILE | output dataframe FILE  (save df: csv/xlsx/html/md/json/...)\n"
        "  output FILE             (infers by extension: .html/.htm/.png/.svg/.pdf -> chart; otherwise dataframe)"
    )

    def run(self, args: List[str], state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return
        if not args:
            print(self.help)
            return

        intent = None
        file_path = None
        a0 = args[0].lower()
        if a0 in ("--chart", "chart"):
            intent = "chart"; file_path = args[1] if len(args) > 1 else None
        elif a0 in ("--dataframe", "dataframe"):
            intent = "dataframe"; file_path = args[1] if len(args) > 1 else None
        else:
            file_path = args[0]
            ext = Path(file_path).suffix.lower()
            intent = "chart" if ext in (".html", ".htm", ".png", ".svg", ".pdf") else "dataframe"

        if not file_path:
            print("Missing output file. Example: output --chart out.html  |  output --dataframe data.csv")
            return

        try:
            _ensure_loaded(state)
        except Exception as e:
            print(f"[output] {e}")
            return

        try:
            if intent == "chart":
                _save_chart(state, file_path)
            else:
                _save_dataframe(state, file_path)
        except Exception as e:
            print(f"[output] Error writing '{file_path}': {e}")
            return

        print(f"Saved {intent} -> {file_path}")

class Oc(Command):
    name = "oc"
    mode = "chart"
    help = "Shortcut: save chart. Usage: oc FILE"
    def run(self, args: List[str], state) -> None:
        if not args:
            print("Usage: oc FILE"); return
        Output().run(["--chart", args[0]], state)

class Od(Command):
    name = "od"
    mode = "chart"
    help = "Shortcut: save dataframe. Usage: od FILE"
    def run(self, args: List[str], state) -> None:
        if not args:
            print("Usage: od FILE"); return
        Output().run(["--dataframe", args[0]], state)
