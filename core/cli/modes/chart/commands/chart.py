from __future__ import annotations
from ....registry import Command

class RenderChart(Command):
    name = 'chart'
    mode = 'chart'
    help = 'Render chart with indicators: chart --out FILE.html'

    def run(self, args, state) -> None:
        if state.mode != 'chart':
            print("Error: set mode to 'chart' first (use: mode chart | mc)." )
            return

        out_file = None
        for i, tok in enumerate(args):
            if tok == '--out' and i + 1 < len(args):
                out_file = args[i + 1]
                break
            if tok.startswith('--out='):
                out_file = tok.split('=', 1)[1]
                break

        if not out_file:
            print('Usage: chart --out FILE.html')
            return
        if not state.ticker:
            print('Error: ticker not set. Use: t SYMBOL  or  tSYMBOL')
            return

        try:
            from core.indicators import discover_all
            discover_all()
        except Exception:
            pass

        try:
            from chart import load_chart_data, add_indicator_columns, plot_candlestick_chart
        except Exception as e:
            print(f'Error: chart helpers not available. Ensure chart.py is importable. ({e})')
            return

        try:
            df = load_chart_data(state.ticker, interval='1day', outputsize=252)
            df, indicator_cols = add_indicator_columns(df, state.indicators)
            plot_candlestick_chart(
                df,
                state.ticker,
                out_file,
                indicator_cols=indicator_cols,
                theme='plotly_dark',
            )
            print(f'Chart written to: {out_file}')
        except Exception as e:
            print(f'Chart error: {e}')
