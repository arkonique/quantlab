from __future__ import annotations
from ....registry import Command

def _mark_derived_dirty(state) -> None:
    state.derived_dirty = True
    state.df = None
    state.indicator_cols = []

class ClearIndicators(Command):
    name = "clear"
    mode = "chart"
    help = "Clear indicators: clear i"

    def run(self, args, state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return
        if args and args[0] in ("i", "indicators"):
            state.indicators.clear()
            _mark_derived_dirty(state)
            print("Indicators cleared.")
        else:
            print("Usage: clear i")
