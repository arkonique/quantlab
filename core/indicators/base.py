# core/indicators/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable
from enum import Enum
import inspect, re

class IndicatorType(str, Enum):
    LINE    = "line"      # continuous line (SMA, EMA, RSI, ...)
    BANDS   = "bands"     # upper/lower envelopes (Bollinger, KC)
    LEVELS  = "levels"    # horizontal levels/zones (Fibo, pivots, S/R)
    MARKERS = "markers"   # discrete points/arrows (fractals, swing highs)
    SIGNAL  = "signal"    # boolean/ternary signals (buy/sell/hold)
    TABLE   = "table"     # tabular stats (roll stats, factor exposures)

class Indicator(ABC):
    _registry: Dict[str, type["Indicator"]] = {}

    # Default presentation type (subclasses override)
    indicator_type: IndicatorType = IndicatorType.LINE
    # Optional: developer sets category in subclass ("trend", "volume", etc.)
    category: str = "uncategorized"

    # Convenience alias to satisfy “variable called type”
    # (kept as a string to be UI-friendly and not fight with Enum types)
    type: str = indicator_type.value

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            slug = getattr(cls, "slug", None) or _to_slug(cls.__name__)
            if slug in Indicator._registry and Indicator._registry[slug] is not cls:
                raise RuntimeError(f"Duplicate indicator slug: {slug}")
            cls.slug = slug
            # keep the alias in sync if subclass overrode indicator_type
            if hasattr(cls, "indicator_type"):
                cls.type = cls.indicator_type.value  # string alias for UIs
            Indicator._registry[slug] = cls

    @classmethod
    def create(cls, slug: str, **params) -> "Indicator":
        if slug not in cls._registry:
            raise KeyError(f"Unknown indicator: {slug}")
        return cls._registry[slug](**params)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def by_category(cls) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for slug, klass in cls._registry.items():
            cat = getattr(klass, "category", "uncategorized")
            out.setdefault(cat, []).append(slug)
        for v in out.values():
            v.sort()
        return out

    @classmethod
    def meta(cls) -> dict:
        """Lightweight metadata for UIs."""
        return {
            "slug": getattr(cls, "slug", _to_slug(cls.__name__)),
            "name": cls.__name__,
            "category": getattr(cls, "category", "uncategorized"),
            "type": getattr(cls, "type", getattr(cls, "indicator_type", IndicatorType.LINE)).value
                    if isinstance(getattr(cls, "indicator_type", IndicatorType.LINE), IndicatorType)
                    else getattr(cls, "type", "line"),
        }

    @abstractmethod
    def required_columns(self) -> Iterable[str]: ...
    @abstractmethod
    def compute(self, df) -> Any: ...

def _to_slug(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
