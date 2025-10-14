# core/indicators/_auto_init.py
import importlib, pkgutil, inspect, sys
from types import ModuleType

def auto_init(package_name: str, path) -> None:
    """
    Import all modules in this package and expose classes defined in them
    as top-level names of the package (EMA, MACD, RSI, ...).
    """
    for mod in pkgutil.iter_modules(path, package_name + "."):
        module: ModuleType = importlib.import_module(mod.name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                # inject class into the package's namespace
                sys.modules[package_name].__dict__[name] = obj
