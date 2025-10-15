from __future__ import annotations
import re
from typing import List
from ....registry import Command

# Reuse the loader that honors one-fetch caching
from .output import _ensure_loaded

# ---- pretty table helper (rich → tabulate → plain text) ----
def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(show_header=True, header_style="bold")
        for h in headers:
            table.add_column(h)
        for r in rows:
            table.add_row(*[str(x) for x in r])
        console.print(table)
        return
    except Exception:
        pass
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="github"))
        return
    except Exception:
        pass
    widths = [max(len(str(x)) for x in col) for col in zip(*([headers] + rows))] if rows else [len(h) for h in headers]
    def fmt(row: List[str]) -> str:
        return "  ".join(str(x).ljust(w) for x, w in zip(row, widths))
    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))
# -----------------------------------------------------------

class PrintDF(Command):
    name = "print"
    aliases = ["p"]
    mode = "chart"
    help = "Print the current dataframe head as a table. Usage: print [N]  |  pN  |  p"

    def match_raw(self, line: str, state) -> bool:
        """Support compact 'p5' style."""
        if state.mode != "chart":
            return False
        m = re.fullmatch(r"p(\d+)", line.strip())
        if not m:
            return False
        n = int(m.group(1))
        self._do_print(n, state)
        return True

    def run(self, args: List[str], state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return

        # Parse N from 'print 10' or default to 5
        n = 5
        if args:
            try:
                n = int(args[0])
            except ValueError:
                print("Usage: print [N]  |  pN  |  p")
                return

        self._do_print(n, state)

    def _do_print(self, n: int, state) -> None:
        try:
            _ensure_loaded(state)  # uses cached raw_df; recomputes derived only if needed
        except Exception as e:
            print(f"[print] {e}")
            return

        df = state.df
        if df is None:
            print("[print] No dataframe available.")
            return

        # Take head and include datetime index if present
        head = df.head(max(1, n))
        # If the index looks like a datetime index, include it as a column for display
        idx_name = head.index.name or "datetime"
        if hasattr(head.index, "inferred_type") and "date" in str(head.index.inferred_type):
            display = head.reset_index()
            if not display.columns[0]:
                display.rename(columns={display.columns[0]: idx_name}, inplace=True)
        else:
            display = head

        headers = [str(c) for c in display.columns.tolist()]
        rows = [[display.iat[i, j] for j in range(display.shape[1])] for i in range(display.shape[0])]
        _print_table(headers, rows)
