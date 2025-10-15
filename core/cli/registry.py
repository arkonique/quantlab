from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CommandSpec:
    name: str
    handler: "Command"
    aliases: List[str] = field(default_factory=list)
    mode: Optional[str] = None

class CommandRegistry:
    def __init__(self) -> None:
        self._by_name: Dict[str, CommandSpec] = {}
        self._globals: Dict[str, CommandSpec] = {}
        self._by_mode: Dict[str, Dict[str, CommandSpec]] = {}

    def register(self, spec: CommandSpec) -> None:
        target = self._globals if spec.mode is None else self._by_mode.setdefault(spec.mode, {})
        target[spec.name] = spec
        for a in spec.aliases:
            target[a] = spec
        self._by_name[spec.name] = spec

    def get_for_mode(self, mode: Optional[str]) -> Dict[str, CommandSpec]:
        merged = dict(self._globals)
        if mode and mode in self._by_mode:
            merged.update(self._by_mode[mode])
        return merged

    def all_modes(self) -> List[str]:
        return sorted(self._by_mode.keys())

class Command:
    name: str = ''
    aliases: List[str] = []
    mode: Optional[str] = None
    help: str = ''

    def match_raw(self, line: str, state) -> bool:
        return False

    def run(self, args: list[str], state) -> None:
        raise NotImplementedError('Command.run must be implemented')
