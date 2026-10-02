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
    for stat, aliases, label in _BOOST_FLAGS:
        group.add_argument(
            *(f"--{prefix}-{name}" for name in (stat, *aliases)),
            dest=f"{prefix}_boost_{stat}",
            type=_stage,
            metavar="STAGE",
            help=f"{label} stage from -6 to +6, e.g. +2",
        )


_BOOST_FLAGS = [
    ("atk", (), "Attack"),
    ("def", (), "Defense"),
    ("spa", ("spatk",), "Sp. Atk"),
    ("spd", ("spdef",), "Sp. Def"),
    ("spe", (), "Speed"),
]


def _stage(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a stat stage; use a number like +2 or -1") from None
    if not -6 <= value <= 6:
        raise argparse.ArgumentTypeError(f"stat stages go from -6 to +6, got {text}")
    return value


def _boosts(args: argparse.Namespace, prefix: str) -> dict[str, int]:
    """Combine --x-boosts with the per-stat flags such as --x-def +2."""
    boosts = parse_stat_spread(getattr(args, f"{prefix}_boosts"))
    for stat, _, _ in _BOOST_FLAGS:
        value = getattr(args, f"{prefix}_boost_{stat}")
        if value is None:
            continue
        if boosts.get(stat, value) != value:
            raise CalcError(f"--{prefix}-boosts and --{prefix}-{stat} give different {stat} stages")
        boosts[stat] = value
    return boosts


def _side(args: argparse.Namespace, prefix: str, species: str) -> PokemonSet:
    get = lambda name: getattr(args, f"{prefix}_{name}")  # noqa: E731
    return PokemonSet(
        species=species,
        nature=get("nature"),
        ability=get("ability"),
        item=get("item"),
        sp=parse_stat_spread(get("sp")),
        boosts=_boosts(args, prefix),
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
    conditions.add_argument("--aurora-veil", action="store_true", help="Aurora Veil on the defender's side")
    doubles = calc.add_argument_group("doubles options")
    doubles.add_argument("--doubles", action="store_true", help="doubles battle: spread moves take 0.75x damage")
    doubles.add_argument("--single-target", action="store_true",
                         help="a spread move hits only one target (no spread penalty)")
    doubles.add_argument("--helping-hand", action="store_true", help="the attacker's ally used Helping Hand")
    doubles.add_argument("--friend-guard", action="store_true", help="the defender's ally has Friend Guard")
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
            Field(
                weather=args.weather,
                terrain=args.terrain,
                reflect=args.reflect,
                light_screen=args.light_screen,
                doubles=args.doubles,
                aurora_veil=args.aurora_veil,
                helping_hand=args.helping_hand,
                friend_guard=args.friend_guard,
            ),
            is_crit=args.crit,
            spread=False if args.single_target else None,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
            return
        print(result.description + (" (spread)" if result.move["spread"] else ""))
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
