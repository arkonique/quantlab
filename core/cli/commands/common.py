from __future__ import annotations
from ..registry import CommandRegistry, Command, CommandSpec

def _help_text(registry: CommandRegistry) -> str:
    modes = ', '.join(registry.all_modes() or ['chart'])
    return f"""
Commands:
  help                         Show this help
  mode list                    List available modes
  mode <name>                  Enter a mode (available: {modes})
  mc                           Shortcut to enter chart mode
  m c                          Shortcut to enter chart mode
  t SYMBOL | ticker SYMBOL     Set ticker (chart mode only)
  tSYMBOL                      Compact ticker set, e.g., 'tAAPL' (chart mode only)
  i SPEC | indicator SPEC      Add indicator (colon positionals + ,k=v extras)
  clear i                      Clear all indicators
  show                         Show current state
  chart --out FILE.html        Render chart to HTML
  quit | exit | q              Exit QuantLab
""".strip()

class Help(Command):
    name = 'help'
    aliases = ['?']
    help = 'Show help message'

    def run(self, args, state) -> None:
        print(_help_text_placeholder)

_help_text_placeholder = ''

class Quit(Command):
    name = 'quit'
    aliases = ['exit', 'q']
    help = 'Exit the shell'

    def run(self, args, state) -> None:
        raise SystemExit

class Mode(Command):
    name = 'mode'
    help = 'Set or list modes'

    def run(self, args, state) -> None:
        if not args or args[0] == 'list':
            print('Available modes: chart')
            return
        val = args[0].lower()
        if val in ('chart', 'c'):
            state.mode = 'chart'
            print("Mode set to 'chart'.")
        else:
            state.mode = val
            print(f"Mode set to '{val}'. (Commands will load if defined under core/cli/modes/{val}/commands)")

class ModeShort(Command):
    name = 'mc'
    help = 'Shortcut for chart mode'

    def run(self, args, state) -> None:
        state.mode = 'chart'
        print("Mode set to 'chart'.")

class MShort(Command):
    name = 'm'
    help = 'Shortcut: m c'

    def run(self, args, state) -> None:
        if args and args[0].lower() == 'c':
            state.mode = 'chart'
            print("Mode set to 'chart'.")
        else:
            print('Usage: m c')

class Show(Command):
    name = 'show'
    help = 'Show current state'

    def run(self, args, state) -> None:
        print(f"Mode   : {state.mode or '(none)'}")
        print(f"Ticker : {state.ticker or '(not set)'}")
        if getattr(state, 'indicators', None):
            print('Indicators:')
            for i, spec in enumerate(state.indicators, 1):
                print(f"  {i}. {spec.get('name')}  -> {spec}")
        else:
            print('Indicators: (none)')

def load_common_commands(registry: CommandRegistry) -> None:
    global _help_text_placeholder
    _help_text_placeholder = _help_text(registry)
    for cls in (Help, Quit, Mode, ModeShort, MShort, Show):
        registry.register(CommandSpec(name=cls.name, handler=cls(), aliases=getattr(cls, 'aliases', []), mode=None))
