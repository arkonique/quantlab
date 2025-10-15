from __future__ import annotations
import importlib
import pkgutil
from .registry import CommandRegistry, Command, CommandSpec

def discover_commands(registry: CommandRegistry, package_root: str = 'core.cli.modes') -> None:
    pkg = importlib.import_module(package_root)
    for modinfo in pkgutil.iter_modules(pkg.__path__):
        mode_name = modinfo.name
        commands_pkg_name = f"{package_root}.{mode_name}.commands"
        try:
            commands_pkg = importlib.import_module(commands_pkg_name)
        except Exception:
            continue
        for subinfo in pkgutil.iter_modules(commands_pkg.__path__):
            module_name = f"{commands_pkg_name}.{subinfo.name}"
            module = importlib.import_module(module_name)
            for attr in dir(module):
                obj = getattr(module, attr)
                try:
                    if isinstance(obj, type) and issubclass(obj, Command) and obj is not Command:
                        cmd = obj()
                        registry.register(CommandSpec(name=cmd.name, handler=cmd, aliases=getattr(cmd, 'aliases', []), mode=cmd.mode))
                except Exception:
                    continue
