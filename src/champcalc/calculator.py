"""Runs the bundled @smogon/calc in an embedded V8 and exposes it to Python."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from functools import cached_property
from importlib import resources
from typing import Any

from py_mini_racer import MiniRacer

from .models import CalcError, Field, MoveDamage, PokemonSet, Result


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
        self._legal: list[str] = champions["legal"]
        self._learnsets: dict[str, list[str]] = champions["learnsets"]
        self._abilities: dict[str, list[str]] = champions["abilities"]
        self._accuracy: dict[str, int | bool] = champions["accuracy"]
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
        spread: bool | None = None,
    ) -> Result:
        """Damage dealt by `attacker` using `move` on `defender`.

        Pass `Field(doubles=True)` for a doubles battle. There, moves that hit
        several targets (Earthquake, Heat Wave, ...) take the 0.75x spread
        penalty; set `spread=False` when only one target is left.
        """
        data = self._call(
            "calculate",
            {
                "attacker": asdict(attacker),
                "defender": asdict(defender),
                "move": move,
                "field": asdict(field or Field()),
                "is_crit": is_crit,
                "spread": spread,
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

    def calculate_many(
        self,
        attacker: PokemonSet,
        defender: PokemonSet,
        moves: list[str],
        field: Field | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[MoveDamage]]:
        """Damage ranges for several moves in one call, for bulk analysis.

        Returns the attacker's and defender's details and one `MoveDamage` per
        move, in order. Much faster than calling `calculate` per move, but skips
        KO-chance text and learnset or ability warnings.
        """
        data = self._call(
            "calculateMany",
            {
                "attacker": asdict(attacker),
                "defender": asdict(defender),
                "moves": list(moves),
                "field": asdict(field or Field()),
            },
        )
        return data["attacker"], data["defender"], [MoveDamage(**r) for r in data["results"]]

    def calculate_batch(
        self,
        moves: list[str],
        jobs: list[tuple[PokemonSet, PokemonSet, Field | None, list[int]]],
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """The fastest bulk path: many (attacker, defender, field, move indices)
        jobs in one call, sharing one `moves` list.

        Returns, per job, the defender's max HP and a (min, max) damage pair per
        move index. No names, descriptions or warnings.
        """
        data = self._call(
            "calculateBatch",
            {
                "moves": list(moves),
                "jobs": [
                    {"attacker": asdict(a), "defender": asdict(d), "field": asdict(f or Field()), "moves": list(idx)}
                    for a, d, f, idx in jobs
                ],
            },
        )
        return [(job["hp"], [tuple(pair) for pair in job["damage"]]) for job in data]

    def species(self, name: str) -> dict[str, Any]:
        """Types, base stats, abilities, formes and Mega Stone of a Pokemon."""
        data = self._call("species", {"name": name})
        data["abilities"] = list(self._abilities.get(data["name"], []))
        return data

    def _species_name(self, name: str) -> str:
        return self._call("species", {"name": name})["name"]

    def move(self, name: str) -> dict[str, Any]:
        """Type, category, power and stats used by a move.

        `accuracy` is a percentage, or True for moves that never miss.
        """
        data = self._call("move", {"name": name})
        data["accuracy"] = self._accuracy.get(data["name"], True)
        return data

    def learnset(self, species: str) -> list[str]:
        """Every move the Pokemon can use in Champions, sorted by name."""
        return list(self._learnsets.get(self._species_name(species), []))

    def can_learn(self, species: str, move: str) -> bool:
        moves = self._learnsets.get(self._species_name(species), [])
        return _to_id(move) in {_to_id(m) for m in moves}

    @property
    def legal_species(self) -> list[str]:
        """Pokemon and formes usable in Champions, Megas included, without
        battle-only formes or duplicates that play identically."""
        return list(self._legal)

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
