from __future__ import annotations
import re
import inspect
from typing import Dict, Any, List
from ....registry import Command

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

def _cast_atom(v: str) -> Any:
    v = v.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d*", v):
        return float(v)
    lv = v.lower()
    if lv in ("true", "false"):
        return lv == "true"
    if lv in ("none", "null"):
        return None
    return v

def _get_param_order_from_init(func) -> List[str]:
    sig = inspect.signature(func)
    names = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        names.append(name)
    return names

def _introspect_param_order(slug: str) -> List[str]:
    # Ensure registry is populated
    try:
        from core.indicators import discover_all
        discover_all()
    except Exception:
        pass

    Indicator = None
    try:
        from core.indicators.base import Indicator as _Indicator
        Indicator = _Indicator
    except Exception:
        try:
            from core.indicators import Indicator as _Indicator  # type: ignore
            Indicator = _Indicator
        except Exception:
            Indicator = None  # type: ignore

    cls = None
    if Indicator is not None:
        for attr_name in ("registry", "_registry", "REGISTRY", "classes", "_classes"):
            reg = getattr(Indicator, attr_name, None)
            if isinstance(reg, dict) and slug in reg:
                cls = reg[slug]
                break
        if cls is None:
            getcls = getattr(Indicator, "get_class", None) or getattr(Indicator, "resolve_class", None)
            if callable(getcls):
                try:
                    cls = getcls(slug)
                except Exception:
                    cls = None
        if cls is None:
            create = getattr(Indicator, "create", None)
            if callable(create):
                try:
                    inst = create(slug)
                    cls = inst.__class__
                except Exception:
                    cls = None

    if cls is None:
        return ["window", "column"]

    init = getattr(cls, "__init__", None)
    if not callable(init):
        return ["window", "column"]
    names = _get_param_order_from_init(init)
    return names or ["window", "column"]

def parse_indicator_token(token: str) -> Dict[str, Any]:
    token = token.strip()
    if not token:
        raise ValueError("Empty indicator token.")

    extras_str = ""
    head = token
    if "," in token:
        head, extras_str = token.split(",", 1)

    head_parts = head.split(":")
    slug = head_parts[0].strip().lower()
    pos_vals = [h.strip() for h in head_parts[1:] if h.strip()]

    params: Dict[str, Any] = {}
    order = _introspect_param_order(slug)

    if len(pos_vals) > len(order):
        raise ValueError(
            f"Too many positional values for '{slug}'. Expected at most {len(order)}: {order}. "
            f"Provide extra args as key=value after a comma."
        )

    for name, val in zip(order, pos_vals):
        params[name] = _cast_atom(val)

    if extras_str:
        for kv in [s for s in extras_str.split(",") if s.strip()]:
            if "=" not in kv:
                raise ValueError(f"Expected key=value in extras: '{kv}'")
            k, v = kv.split("=", 1)
            params[k.strip()] = _cast_atom(v.strip())

    name = slug.upper()
    if "window" in params and params["window"] is not None:
        name += f"({params['window']})"

    try:
        from core.indicators import Indicator  # type: ignore
        create = getattr(Indicator, "create", None)
        if callable(create):
            create(slug, **params)
    except Exception:
        pass

    return {"slug": slug, "params": params, "name": name}

def _iter_indicator_classes():
    """Return list of (slug, cls) from the registry (best-effort)."""
    try:
        from core.indicators import discover_all
        discover_all()
    except Exception:
        pass
    Indicator = None
    try:
        from core.indicators.base import Indicator as _Indicator
        Indicator = _Indicator
    except Exception:
        try:
            from core.indicators import Indicator as _Indicator  # type: ignore
            Indicator = _Indicator
        except Exception:
            Indicator = None  # type: ignore
    if Indicator is None:
        return []
    for attr_name in ("registry", "_registry", "REGISTRY", "classes", "_classes"):
        reg = getattr(Indicator, attr_name, None)
        if isinstance(reg, dict) and reg:
            return list(reg.items())
    return []

def _format_params_signature(cls) -> str:
    init = getattr(cls, "__init__", None)
    if not callable(init):
        return ""
    sig = inspect.signature(init)
    parts = []
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            parts.append(name)
        else:
            parts.append(f"{name}={p.default!r}")
    return ", ".join(parts)

def _mark_derived_dirty(state) -> None:
    state.derived_dirty = True
    state.df = None
    state.indicator_cols = []

class Indicator(Command):
    name = "indicator"
    aliases = ["i"]
    mode = "chart"
    help = "Add indicator or list them: i slug[:pos1[:pos2...]][,k=v...] | i list"

    def run(self, args: List[str], state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return
        if not args:
            print("Usage: i slug[:pos1[:pos2...]][,k=v[,k=v...]]  |  i list")
            return

        # Listing mode
        if args[0].lower() in ("list", "ls"):
            items = _iter_indicator_classes()
            if not items:
                print("No indicator registry found. Ensure core.indicators exposes a registry or discover_all().")
                return
            rows = []
            for slug, cls in items:
                cat = getattr(cls, "category", "") or ""
                disp = getattr(cls, "name", slug.upper()) or slug.upper()
                sig = _format_params_signature(cls)
                rows.append((slug, disp, cat, sig))
            rows_sorted = sorted(rows, key=lambda x: x[0])
            headers = ["slug", "name", "category", "params"]
            _print_table(headers, [[s, d, c, sig] for s, d, c, sig in rows_sorted])
            return

        # Add an indicator (makes DERIVED dirty only; RAW stays cached)
        spec_str = " ".join(args).strip()
        try:
            spec = parse_indicator_token(spec_str)
            state.indicators.append(spec)
            _mark_derived_dirty(state)
            print(f"Indicator added: {spec['name']} -> {spec['params']}")
        except Exception as e:
            print(f"Indicator parse error: {e}")
