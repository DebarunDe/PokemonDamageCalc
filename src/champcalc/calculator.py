"""Runs the bundled @smogon/calc in an embedded V8 and exposes it to Python."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from functools import cached_property
from importlib import resources
from typing import Any

from py_mini_racer import MiniRacer

from .models import CalcError, Field, PokemonSet, Result


def _to_id(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


class Calculator:
    """Pokemon Champions damage calculator.

    Creating one loads the JS bundle (about 0.2s); reuse a single instance.
    Instances are safe to share between threads.
    """

    def __init__(self) -> None:
        data = resources.files("champcalc") / "data"
        self._ctx = MiniRacer()
        self._ctx.eval(data.joinpath("calc.js").read_text(encoding="utf-8"))
        champions = json.loads(data.joinpath("champions.json").read_text(encoding="utf-8"))
        self.showdown_commit: str = champions["showdown_commit"]
        self._learnsets: dict[str, list[str]] = champions["learnsets"]
        self._abilities: dict[str, list[str]] = champions["abilities"]
        self._lock = threading.Lock()

    def _call(self, function: str, payload: dict[str, Any]) -> Any:
        with self._lock:
            raw = self._ctx.call(f"champcalc.{function}", json.dumps(payload))
        response = json.loads(raw)
        if response["ok"]:
            return response["value"]
        if response["user_error"]:
            raise CalcError(response["error"])
        raise RuntimeError(f"@smogon/calc failed: {response['error']}")

    def calculate(
        self,
        attacker: PokemonSet,
        defender: PokemonSet,
        move: str,
        field: Field | None = None,
        is_crit: bool = False,
    ) -> Result:
        """Damage dealt by `attacker` using `move` on `defender` in a singles battle."""
        data = self._call(
            "calculate",
            {
                "attacker": asdict(attacker),
                "defender": asdict(defender),
                "move": move,
                "field": asdict(field or Field()),
                "is_crit": is_crit,
            },
        )
        warnings = []
        species = data["attacker"]["species"]
        if not self.can_learn(species, data["move"]["name"]):
            warnings.append(f"{species} cannot learn {data['move']['name']} in Pokemon Champions")
        for side in ("attacker", "defender"):
            mon = data[side]
            legal = self._abilities.get(mon["species"], [])
            if mon["ability"] and legal and mon["ability"] not in legal:
                warnings.append(f"{mon['species']} cannot have {mon['ability']}; it can have {', '.join(legal)}")
        return Result.from_js(data, warnings)

    def species(self, name: str) -> dict[str, Any]:
        """Types, base stats, abilities and formes of a Pokemon."""
        data = self._call("species", {"name": name})
        data["abilities"] = list(self._abilities.get(data["name"], []))
        return data

    def _species_name(self, name: str) -> str:
        return self._call("species", {"name": name})["name"]

    def move(self, name: str) -> dict[str, Any]:
        return self._call("move", {"name": name})

    def learnset(self, species: str) -> list[str]:
        """Every move the Pokemon can use in Champions, sorted by name."""
        return list(self._learnsets.get(self._species_name(species), []))

    def can_learn(self, species: str, move: str) -> bool:
        moves = self._learnsets.get(self._species_name(species), [])
        return _to_id(move) in {_to_id(m) for m in moves}

    @cached_property
    def all_species(self) -> list[str]:
        return self._call("list", {"kind": "species"})

    @cached_property
    def all_moves(self) -> list[str]:
        return self._call("list", {"kind": "moves"})

    @cached_property
    def all_items(self) -> list[str]:
        return self._call("list", {"kind": "items"})

    @cached_property
    def all_abilities(self) -> list[str]:
        return self._call("list", {"kind": "abilities"})
