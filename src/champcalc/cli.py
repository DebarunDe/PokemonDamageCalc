"""Command line interface: `champcalc calc`, `champcalc moves`, `champcalc info`."""

from __future__ import annotations

import argparse
import json
import re
import sys

from .calculator import Calculator
from .models import STATS, CalcError, Field, PokemonSet

_STAT_ALIASES = {
    "hp": "hp", "atk": "atk", "attack": "atk", "def": "def", "defense": "def",
    "spa": "spa", "spatk": "spa", "spd": "spd", "spdef": "spd", "spe": "spe", "speed": "spe",
}


def parse_stat_spread(text: str) -> dict[str, int]:
    """Parse 'atk=32,spe=32' or Showdown-style '32 Atk / 32 Spe' (also '+1 Atk')."""
    spread: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in re.split(r"[,/]", text or ""))):
        match = re.fullmatch(r"([a-zA-Z. ]+?)\s*=\s*([+-]?\d+)", part) or re.fullmatch(
            r"([+-]?\d+)\s+([a-zA-Z. ]+)", part
        )
        if not match:
            raise CalcError(f"Could not read '{part}'; use 'atk=32' or '32 Atk'")
        a, b = match.groups()
        name, value = (a, b) if a[0].isalpha() else (b, a)
        stat = _STAT_ALIASES.get(re.sub(r"[^a-z]", "", name.lower()))
        if not stat:
            raise CalcError(f"Unknown stat '{name}'")
        spread[stat] = int(value)
    return spread


def _add_side_options(parser: argparse.ArgumentParser, prefix: str, who: str) -> None:
    group = parser.add_argument_group(f"{who} options")
    group.add_argument(f"--{prefix}-nature", help="e.g. Jolly")
    group.add_argument(f"--{prefix}-ability", help="defaults to the first ability")
    group.add_argument(f"--{prefix}-item", help="e.g. 'Life Orb'")
    group.add_argument(f"--{prefix}-sp", default="", help="Stat Points, e.g. '32 Atk / 32 Spe' or 'atk=32,spe=32'")
    group.add_argument(f"--{prefix}-boosts", default="", help="stat stages, e.g. '+1 Atk' or 'def=-1'")
    group.add_argument(f"--{prefix}-status", default="", help="brn, par, psn, tox, slp or frz")
    group.add_argument(f"--{prefix}-hp", type=float, help="current HP as a percentage")


def _side(args: argparse.Namespace, prefix: str, species: str) -> PokemonSet:
    get = lambda name: getattr(args, f"{prefix}_{name}")  # noqa: E731
    return PokemonSet(
        species=species,
        nature=get("nature"),
        ability=get("ability"),
        item=get("item"),
        sp=parse_stat_spread(get("sp")),
        boosts=parse_stat_spread(get("boosts")),
        status=get("status"),
        cur_hp_percent=get("hp"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="champcalc", description="Pokemon Champions damage calculator")
    commands = parser.add_subparsers(dest="command", required=True)

    calc = commands.add_parser(
        "calc",
        help="damage from one Pokemon's attack on another",
        description="Example: champcalc calc Garchomp Earthquake Incineroar --a-nature Jolly --a-sp '32 Atk' --d-sp '32 HP'",
    )
    calc.add_argument("attacker")
    calc.add_argument("move")
    calc.add_argument("defender")
    _add_side_options(calc, "a", "attacker")
    _add_side_options(calc, "d", "defender")
    conditions = calc.add_argument_group("field options")
    conditions.add_argument("--crit", action="store_true", help="the hit is a critical hit")
    conditions.add_argument("--weather", help="Sun, Rain, Sand or Snow")
    conditions.add_argument("--terrain", help="Electric, Grassy, Psychic or Misty")
    conditions.add_argument("--reflect", action="store_true", help="Reflect on the defender's side")
    conditions.add_argument("--light-screen", action="store_true", help="Light Screen on the defender's side")
    calc.add_argument("--rolls", action="store_true", help="show all damage rolls")
    calc.add_argument("--json", action="store_true", help="print the full result as JSON")

    moves = commands.add_parser("moves", help="list a Pokemon's Champions movepool")
    moves.add_argument("pokemon")

    info = commands.add_parser("info", help="show a Pokemon's types, abilities and base stats")
    info.add_argument("pokemon")
    return parser


def _run(args: argparse.Namespace, calc: Calculator) -> None:
    if args.command == "calc":
        result = calc.calculate(
            _side(args, "a", args.attacker),
            _side(args, "d", args.defender),
            args.move,
            Field(args.weather, args.terrain, args.reflect, args.light_screen),
            is_crit=args.crit,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
            return
        print(result.description)
        if args.rolls:
            print("Rolls: " + ", ".join(map(str, result.rolls)))
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    elif args.command == "moves":
        name = calc.species(args.pokemon)["name"]
        moves = calc.learnset(name)
        print(f"{name} ({len(moves)} moves)")
        for move in moves:
            print(f"  {move}")
    elif args.command == "info":
        s = calc.species(args.pokemon)
        print(f"{s['name']}  [{' / '.join(s['types'])}]")
        print(f"Abilities: {', '.join(s['abilities'])}")
        stats = s["base_stats"]
        print("Base stats: " + "  ".join(f"{k.upper()} {stats[k]}" for k in STATS) + f"  (BST {sum(stats.values())})")
        if s["other_formes"]:
            print(f"Other formes: {', '.join(s['other_formes'])}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _run(args, Calculator())
    except CalcError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
