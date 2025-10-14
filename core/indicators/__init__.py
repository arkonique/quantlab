# core/indicators/__init__.py
"""
Discovery utilities for indicator packages.

- discover_all():     recursively import EVERYTHING under core.indicators.*
- discover_category(): import only a specific subpackage (e.g., 'trend', 'volume')
"""

from __future__ import annotations
import importlib
import pkgutil
from typing import Iterable, List


def _iter_module_names(pkg) -> Iterable[str]:
    """Yield fully qualified module names under a package (recursively)."""
    # walk_packages recurses into subpackages automatically
    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        yield modinfo.name


def discover_all(package_name: str = __name__) -> List[str]:
    """
    Recursively import all subpackages and modules under core.indicators
    so every Indicator subclass gets registered.

    Returns a list of imported module names (useful for logging/tests).
    """
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):
        return []
    imported = []
    for mod_name in _iter_module_names(pkg):
        importlib.import_module(mod_name)
        imported.append(mod_name)
    return imported


def discover_category(category: str, package_name: str = __name__) -> List[str]:
    """
    Import only one category subpackage (e.g., 'trend', 'momentum', 'volume').

    Example:
        discover_category("trend")  # imports core.indicators.trend.*

    Returns a list of imported module names.
    """
    if not category or "." in category:
        raise ValueError("category must be a simple subpackage name like 'trend'")

    base = f"{package_name}.{category}"
    pkg = importlib.import_module(base)
    if not hasattr(pkg, "__path__"):
        # It's a single module, not a package with children.
        return [base]

    imported = []
    for mod_name in _iter_module_names(pkg):
        importlib.import_module(mod_name)
        imported.append(mod_name)
    return imported
