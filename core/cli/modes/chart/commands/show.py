from __future__ import annotations
from ....registry import Command

class ShowChart(Command):
    name = 'show'
    mode = 'chart'
    help = 'Show state (chart-mode alias)'

    def run(self, args, state) -> None:
        print(f"Mode   : {state.mode or '(none)'}")
        print(f"Ticker : {state.ticker or '(not set)'}")
        if getattr(state, 'indicators', None):
            print('Indicators:')
            for i, spec in enumerate(state.indicators, 1):
                print(f"  {i}. {spec.get('name')}  -> {spec}")
        else:
            print('Indicators: (none)')
