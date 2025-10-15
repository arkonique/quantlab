from __future__ import annotations
import shlex, sys
from .registry import CommandRegistry
from .loader import discover_commands
from .state import ShellState
from .commands.common import load_common_commands

def run(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    registry = CommandRegistry()
    load_common_commands(registry)
    discover_commands(registry)

    state = ShellState()

    while True:
        try:
            line = input(f"{state.prompt_prefix()}quantlab$ ")
        except (EOFError, KeyboardInterrupt):
            print(file=stdout)
            break

        line = line.strip()
        if not line:
            continue

        handled = False
        for spec in registry.get_for_mode(state.mode).values():
            try:
                if hasattr(spec.handler, 'match_raw') and spec.handler.match_raw(line, state):
                    handled = True
                    break
            except Exception as e:
                print(f"[error] {e}")
        if handled:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError as e:
            print(f"Parse error: {e}")
            continue

        if not tokens:
            continue

        cmd_name = tokens[0].lower()
        cmd_map = registry.get_for_mode(state.mode)
        spec = cmd_map.get(cmd_name)
        if not spec:
            print(f"Unknown command: {cmd_name}. Try 'help'.")
            continue

        try:
            spec.handler.run(tokens[1:], state)
        except SystemExit:
            break
        except Exception as e:
            print(f"[Command Error] {e}")
