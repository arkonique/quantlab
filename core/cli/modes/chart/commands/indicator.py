from __future__ import annotations
import re
import sys
import os
import inspect
import pkgutil
import importlib
from typing import Dict, Any, List, Tuple, Optional
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

# ---------- indicator parsing for "i <slug>[:v1[:v2...]][,k=v...]" ----------
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

def _introspect_param_order_via_class(cls) -> List[str]:
    init = getattr(cls, "__init__", None)
    if not callable(init):
        return []
    return _get_param_order_from_init(init)

# --- package discovery helpers ---
def _ensure_core_on_sys_path():
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

def _import_indicator_base():
    IndicatorBase = None
    IndicatorType = None
    for mod in ("core.indicators.base", "core.indicators"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "Indicator") and IndicatorBase is None:
                IndicatorBase = getattr(m, "Indicator")
            if hasattr(m, "IndicatorType") and IndicatorType is None:
                IndicatorType = getattr(m, "IndicatorType")
        except Exception:
            continue
    return IndicatorBase, IndicatorType

def _walk_indicator_modules(debug: bool = False) -> List[str]:
    imported: List[str] = []
    try:
        pkg = importlib.import_module("core.indicators")
    except Exception as e:
        if debug:
            print(f"[i list] Cannot import 'core.indicators': {e}")
        return imported

    if not hasattr(pkg, "__path__"):
        if debug:
            print("[i list] 'core.indicators' has no __path__ (not a package?)")
        return imported

    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        mod_name = m.name
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            if debug:
                print(f"[i list] Skipped {mod_name}: {e}")
            continue
    return imported

def _derive_category_from_module(mod_name: str) -> str:
    parts = mod_name.split(".")
    try:
        idx = parts.index("indicators")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return ""

def _indicator_type_to_str(t, IndicatorTypeEnum=None) -> str:
    """
    Normalize indicator_type to a friendly string.
    Accepts:
      - Enum value (IndicatorType.*)  -> lowercased name
      - str ("line", "band", ...)     -> lowercased
      - anything else / missing       -> ""
    """
    if t is None:
        return ""
    # Enum from your base
    try:
        import enum
        if isinstance(t, enum.Enum):
            return str(t.name).lower()
    except Exception:
        pass
    # Match specific enum class if provided
    if IndicatorTypeEnum is not None:
        try:
            if isinstance(t, IndicatorTypeEnum):
                return str(t.name).lower()
        except Exception:
            pass
    # String fallback
    if isinstance(t, str):
        return t.strip().lower()
    # Last resort: repr tail
    s = str(t)
    if "." in s:
        s = s.split(".")[-1]
    return s.strip().lower()

def _collect_indicator_classes(debug: bool = False) -> List[Tuple[str, str, str, type]]:
    """
    Returns list of (slug, category, type_str, class).
    - slug from class.slug (lowercased)
    - category from class.category or derived from module path
    - type_str from class.indicator_type (enum or str) normalized
    """
    _ensure_core_on_sys_path()
    IndicatorBase, IndicatorTypeEnum = _import_indicator_base()
    if IndicatorBase is None:
        if debug:
            print("[i list] Could not find core.indicators.base.Indicator")
        return []

    imported = _walk_indicator_modules(debug=debug)

    def all_subclasses(cls):
        seen = set()
        stack = list(cls.__subclasses__())
        out = []
        while stack:
            sub = stack.pop()
            if sub in seen: continue
            seen.add(sub); out.append(sub); stack.extend(sub.__subclasses__())
        return out

    items: List[Tuple[str, str, str, type]] = []
    for cls in all_subclasses(IndicatorBase):
        slug = getattr(cls, "slug", None)
        if not slug:
            continue
        slug = str(slug).lower().strip()
        mod = getattr(cls, "__module__", "") or ""
        category = (getattr(cls, "category", None) or _derive_category_from_module(mod) or "").strip()
        ind_type = _indicator_type_to_str(getattr(cls, "indicator_type", None), IndicatorTypeEnum)
        items.append((slug, category, ind_type, cls))

    # Dedup by slug (last import wins)
    dedup: Dict[str, Tuple[str, str, type]] = {}
    for slug, cat, tstr, cls in items:
        dedup[slug] = (cat, tstr, cls)

    if debug:
        print(f"[i list] Imported modules: {len(imported)}")
        for name in imported:
            print(f"  - {name}")
        print(f"[i list] Discovered indicators: {len(dedup)}")

    # Return sorted by slug
    return [(slug, dedup[slug][0], dedup[slug][1], dedup[slug][2]) for slug in sorted(dedup.keys())]

def _resolve_indicator_class(slug: str) -> Optional[type]:
    slug = (slug or "").lower().strip()
    for s, _cat, _typ, cls in _collect_indicator_classes(debug=False):
        if s == slug:
            return cls
    return None

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

    cls = _resolve_indicator_class(slug)
    order = _introspect_param_order_via_class(cls) if cls else []
    if not order:
        order = ["window", "column"]

    if len(pos_vals) > len(order):
        raise ValueError(
            f"Too many positional values for '{slug}'. Expected at most {len(order)}: {order}. "
            f"Provide extra args as key=value after a comma."
        )

    params: Dict[str, Any] = {}
    for name, val in zip(order, pos_vals):
        params[name] = _cast_atom(val)

    if extras_str:
        for kv in [s for s in extras_str.split(",") if s.strip()]:
            if "=" not in kv:
                raise ValueError(f"Expected key=value in extras: '{kv}'")
            k, v = kv.split("=", 1)
            params[k.strip()] = _cast_atom(v.strip())

    display_name = (getattr(cls, "name", None) if cls else None) or slug.upper()
    if "window" in params and params["window"] is not None:
        display_name += f"({params['window']})"

    # optional sanity via IndicatorBase.create, if available
    try:
        from core.indicators.base import Indicator as IndicatorBase
        create = getattr(IndicatorBase, "create", None)
        if callable(create):
            try:
                create(slug, **params)
            except Exception:
                pass
    except Exception:
        pass

    return {"slug": slug, "params": params, "name": display_name}

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

# ======== Command ========
class Indicator(Command):
    name = "indicator"
    aliases = ["i"]
    mode = "chart"
    help = (
        "Add/list/remove indicators.\n"
        "  i <slug>[:p1[:p2...]][,k=v[,k=v...]]    Add indicator\n"
        "  i list [category] [--debug]              List AVAILABLE indicators\n"
        "  i current                                List ADDED indicators for this ticker/session\n"
        "    aliases: i active | i show | i added\n"
        "  i remove <idx|slug> [--all]              Remove indicator by 1-based index or slug (latest match). Use --all to remove all with that slug.\n"
        "    aliases: i rm | i del | i delete\n"
        "  i clear                                   Remove ALL added indicators (alias: i reset)\n"
    )

    def run(self, args: List[str], state) -> None:
        if state.mode != "chart":
            print("Error: set mode to 'chart' first (use: mode chart | mc).")
            return
        if not args:
            print(self.help)
            return

        cmd = args[0].lower()

        # ---- List ADDED (current session) ----
        if cmd in ("current", "active", "show", "added"):
            if not state.indicators:
                print("No indicators have been added yet.")
                return
            rows = []
            for idx, spec in enumerate(state.indicators, start=1):
                slug = spec.get("slug", "")
                name = spec.get("name", slug.upper())
                params = spec.get("params", {})
                ind_cls = _resolve_indicator_class(slug)
                ind_type = _indicator_type_to_str(getattr(ind_cls, "indicator_type", None)) if ind_cls else ""
                rows.append([idx, slug, name, ind_type, params])
            _print_table(["#", "slug", "name", "type", "params"], rows)
            try:
                cols = getattr(state, "indicator_cols", None)
                if cols:
                    print(f"\nComputed indicator columns available: {len(cols)}")
            except Exception:
                pass
            return

        # ---- List AVAILABLE (introspective) ----
        if cmd in ("list", "ls"):
            debug = any(a == "--debug" for a in args[1:])
            category_filter = None
            for a in args[1:]:
                if a != "--debug":
                    category_filter = a.lower().strip()

            items = _collect_indicator_classes(debug=debug)
            if category_filter:
                items = [(slug, cat, typ, cls) for (slug, cat, typ, cls) in items if cat.lower() == category_filter]

            if not items:
                msg = "No indicators discovered"
                if category_filter:
                    msg += f" in category '{category_filter}'"
                msg += ". Ensure:\n- 'core/indicators' and each subfolder (trend, momentum, etc.) has an __init__.py\n" \
                       "- You're running from the project root so 'core' is on PYTHONPATH\n" \
                       "- Required libs (e.g. pandas) are installed"
                print(msg)
                return

            rows = []
            for slug, cat, typ, cls in items:
                name = getattr(cls, "name", slug.upper()) or slug.upper()
                sig = _format_params_signature(cls)
                rows.append((slug, name, cat, typ, sig))
            _print_table(["slug", "name", "category", "type", "params"], [[s, n, c, t, sig] for s, n, c, t, sig in rows])
            return

        # ---- Remove ALL indicators ----
        if cmd in ("clear", "reset"):
            if not state.indicators:
                print("No indicators to clear.")
                return
            count = len(state.indicators)
            state.indicators.clear()
            _mark_derived_dirty(state)
            print(f"Cleared {count} indicator(s).")
            return

        # ---- Remove one / many by idx or slug ----
        if cmd in ("remove", "rm", "del", "delete"):
            if len(args) < 2:
                print("Usage: i remove <idx|slug> [--all]")
                return
            target = args[1].strip().lower()
            remove_all = any(a == "--all" for a in args[2:])
            if not state.indicators:
                print("No indicators to remove.")
                return

            removed = 0
            # numeric index? (1-based)
            if re.fullmatch(r"\d+", target):
                idx = int(target)
                if 1 <= idx <= len(state.indicators):
                    spec = state.indicators.pop(idx - 1)
                    removed = 1
                    print(f"Removed #{idx}: {spec.get('name', spec.get('slug',''))}")
                else:
                    print(f"No indicator at index {idx}. Use 'i current' to see indices.")
            else:
                # slug match: remove last (or all) matching entries
                slug = target
                matches = [i for i, sp in enumerate(state.indicators) if sp.get("slug","").lower() == slug]
                if not matches:
                    print(f"No indicator with slug '{slug}' found. Use 'i current' to see slugs.")
                else:
                    if remove_all:
                        for i in reversed(matches):
                            state.indicators.pop(i)
                            removed += 1
                        print(f"Removed {removed} indicator(s) with slug '{slug}'.")
                    else:
                        i = matches[-1]
                        spec = state.indicators.pop(i)
                        removed = 1
                        print(f"Removed last '{slug}': {spec.get('name', slug)}")

            if removed > 0:
                _mark_derived_dirty(state)
            return

        # ---- Add an indicator ----
        spec_str = " ".join(args).strip()
        try:
            spec = parse_indicator_token(spec_str)
            state.indicators.append(spec)
            _mark_derived_dirty(state)
            print(f"Indicator added: {spec['name']} -> {spec['params']}")
        except Exception as e:
            print(f"Indicator parse error: {e}")


from ....registry import Command as _BaseCommand  # reuse the same base

class IndicatorList(_BaseCommand):
    name = "il"
    aliases = []        # add more if you like, e.g. ["ilist"]
    mode = "chart"
    help = "Alias for 'indicator list'. Usage: il [category] [--debug]"

    def run(self, args, state):
        # forward to Indicator command with "list" subcommand
        try:
            Indicator().run(["list", *args], state)
        except NameError:
            # If this class lives in a separate file (il.py), import Indicator from sibling module
            from .indicator import Indicator as _Indicator
            _Indicator().run(["list", *args], state)