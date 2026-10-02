"""Which Mega one-hit KOs the most other Megas?

Every Mega attacks every other Mega with every usable damaging move, at three
attacker baselines and three defender baselines:

    attacker  max+   32 SP in the attacking stat, boosting nature
              max    32 SP in the attacking stat, neutral nature
              none   no investment, neutral nature
    defender  none   no investment, neutral nature
              hp     32 HP
              bulk   32 HP and 32 in the defensive stat the move hits

Each Mega holds its Mega Stone. The field follows from the two abilities:
weather and terrain setters apply (when both set weather, the slower Mega's
wins because abilities activate in speed order), Intimidate lowers a physical
attacker's Attack (with the usual immunities and Contrary/Defiant), and Trace
copies the opponent's ability. Moves that cannot OHKO on the first turn of a
fresh matchup are left out (see EXCLUDED_MOVES).

Each matchup records the strongest move overall and the strongest "reliable"
move, one with at least 90% accuracy after weather and No Guard. Crits,
accuracy-lowering effects and speed order are otherwise ignored.

Usage:
    python analysis/mega_ohko.py                 # singles
    python analysis/mega_ohko.py --doubles       # spread moves take 0.75x
    python analysis/mega_ohko.py --exclude recharge   # also drop Hyper Beam & co.
    python analysis/mega_ohko.py --out analysis/output
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from champcalc import Calculator, Field, PokemonSet

ATTACKER_TIERS = ("max+", "max", "none")
DEFENDER_TIERS = ("none", "hp", "bulk")

# Moves that cannot one-hit KO on turn one of a fresh 1v1, with the reason.
EXCLUDED_MOVES = {
    **dict.fromkeys(["Fissure", "Guillotine", "Horn Drill", "Sheer Cold"], "OHKO move"),
    **dict.fromkeys(
        ["Counter", "Mirror Coat", "Metal Burst", "Comeuppance", "Endeavor", "Final Gambit",
         "Super Fang", "Night Shade", "Seismic Toss"],
        "fixed or reactive damage",
    ),
    **dict.fromkeys(["Explosion", "Self-Destruct", "Misty Explosion"], "user faints"),
    **dict.fromkeys(
        ["Bounce", "Dig", "Dive", "Fly", "Phantom Force", "Sky Attack", "Meteor Beam"],
        "two-turn move",
    ),
    "Focus Punch": "fails if the user is hit first",
    "Future Sight": "delayed damage",
    "Upper Hand": "fails unless the target uses a priority move",
    "Belch": "needs a consumed Berry",
    "Last Resort": "needs the user's other moves used first",
    "Spit Up": "needs Stockpile",
    "Snore": "needs the user asleep",
    "Beat Up": "depends on the party",
    "Fling": "Mega Stones cannot be flung",
}
# Moves that force the user to skip its next turn; drop them with --exclude recharge.
RECHARGE_MOVES = {"Hyper Beam", "Giga Impact", "Blast Burn", "Frenzy Plant", "Hydro Cannon"}
# Charge for a turn unless sun is up (Mega Sol counts as sun for its user).
SOLAR_MOVES = {"Solar Beam", "Solar Blade"}
# Fail without terrain.
TERRAIN_MOVES = {"Steel Roller"}

WEATHER_ABILITIES = {"Drought": "Sun", "Drizzle": "Rain", "Sand Stream": "Sand", "Snow Warning": "Snow"}
TERRAIN_ABILITIES = {
    "Electric Surge": "Electric", "Grassy Surge": "Grassy", "Psychic Surge": "Psychic", "Misty Surge": "Misty",
}
INTIMIDATE_IMMUNE = {
    "Inner Focus", "Scrappy", "Own Tempo", "Oblivious", "Clear Body", "White Smoke",
    "Full Metal Body", "Hyper Cutter", "Mirror Armor",
}
INTIMIDATE_RAISES = {"Contrary", "Defiant", "Guard Dog"}  # net +1 Attack
UNTRACEABLE = {"Trace"}

RELIABLE_ACCURACY = 90
# Weather that changes accuracy: (move, weather) -> accuracy.
WEATHER_ACCURACY = {
    ("Blizzard", "Snow"): 100,
    ("Thunder", "Rain"): 100, ("Hurricane", "Rain"): 100,
    ("Thunder", "Sun"): 50, ("Hurricane", "Sun"): 50,
}

BOOSTING_NATURE = {"atk": "Adamant", "spa": "Modest", "def": "Bold"}
NEUTRAL_NATURE = "Serious"


@dataclass(frozen=True)
class Mega:
    name: str
    ability: str
    stone: str
    speed: int  # with no Speed investment, used for weather order
    moves: tuple[str, ...]


def intimidate_stage(target_ability: str) -> int:
    if target_ability in INTIMIDATE_IMMUNE:
        return 0
    if target_ability in INTIMIDATE_RAISES:
        return 1
    return -1


def accuracy(move: str, base: int | bool, weather: str | None, no_guard: bool) -> int:
    if no_guard or base is True:
        return 100
    return WEATHER_ACCURACY.get((move, weather), base)


def effective_abilities(a: Mega, d: Mega) -> tuple[str, str]:
    """Abilities after Trace copies the opponent's."""
    ability_a = d.ability if a.ability == "Trace" and d.ability not in UNTRACEABLE else a.ability
    ability_d = a.ability if d.ability == "Trace" and a.ability not in UNTRACEABLE else d.ability
    return ability_a, ability_d


def resolve_condition(a: Mega, d: Mega, ability_a: str, ability_d: str, table: dict[str, str]):
    """The weather (or terrain) in play and whether it was a speed contest.

    Both Megas' abilities activate in speed order, so the slower one's setting
    is the one left standing. A speed tie is a coin flip; the attacker's is used.
    """
    set_a, set_d = table.get(ability_a), table.get(ability_d)
    if set_a and set_d and set_a != set_d:
        if d.speed < a.speed:
            return set_d, "contested"
        return set_a, "contested" if a.speed < d.speed else "contested (speed tie)"
    return set_a or set_d, ""


def load_megas(calc: Calculator, excluded: set[str] = frozenset()) -> list[Mega]:
    move_info = {}
    megas = []
    for name in calc.all_species:
        if "-Mega" not in name:
            continue
        info = calc.species(name)
        moves = []
        for move in calc.learnset(name):
            if move not in move_info:
                move_info[move] = calc.move(move)
            if move_info[move]["category"] != "Status" and move not in EXCLUDED_MOVES and move not in excluded:
                moves.append(move)
        megas.append(Mega(
            name=name,
            ability=info["abilities"][0],
            stone=info["mega_stone"],
            speed=info["base_stats"]["spe"] + 20,
            moves=tuple(moves),
        ))
    return megas


_calc: Calculator | None = None
_move_info: dict[str, dict] = {}


def _worker_init() -> None:
    global _calc
    _calc = Calculator()


def _info(move: str) -> dict:
    if move not in _move_info:
        _move_info[move] = _calc.move(move)
    return _move_info[move]


def attacker_rows(args: tuple[Mega, list[Mega], bool]) -> list[dict]:
    """Best move and OHKO status for one attacker against every other Mega, at every baseline."""
    a, megas, doubles = args
    rows = []
    for d in megas:
        if d.name == a.name:
            continue
        ability_a, ability_d = effective_abilities(a, d)
        weather, weather_note = resolve_condition(a, d, ability_a, ability_d, WEATHER_ABILITIES)
        terrain, _ = resolve_condition(a, d, ability_a, ability_d, TERRAIN_ABILITIES)
        field = Field(weather=weather, terrain=terrain, doubles=doubles)

        sun = weather == "Sun" or ability_a == "Mega Sol"
        moves = [
            m for m in a.moves
            if (m not in SOLAR_MOVES or sun) and (m not in TERRAIN_MOVES or terrain)
        ]
        # Group moves by the stats they use, so each group gets matching investment.
        groups: dict[tuple, list[str]] = defaultdict(list)
        for m in moves:
            info = _info(m)
            groups[(info["offensive_stat"], info["defensive_stat"], info["uses_target_attack"])].append(m)

        notes = [n for n in (f"{weather} ({weather_note})" if weather_note else "",) if n]
        if ability_a != a.ability:
            notes.append(f"{a.name} traces {ability_a}")
        if ability_d != d.ability:
            notes.append(f"{d.name} traces {ability_d}")
        attacker_atk = intimidate_stage(ability_a) if ability_d == "Intimidate" else 0
        defender_atk = intimidate_stage(ability_d) if ability_a == "Intimidate" else 0
        if attacker_atk:
            notes.append(f"Intimidate: attacker Atk {attacker_atk:+d}")

        no_guard = "No Guard" in (ability_a, ability_d)
        acc = {m: accuracy(m, _info(m)["accuracy"], weather, no_guard) for m in moves}

        for a_tier in ATTACKER_TIERS:
            for d_tier in DEFENDER_TIERS:
                results = []
                hp = None
                for (off_stat, def_stat, uses_target), group in groups.items():
                    sp_a = {} if a_tier == "none" or uses_target else {off_stat: 32}
                    nature_a = BOOSTING_NATURE[off_stat] if a_tier == "max+" and not uses_target else NEUTRAL_NATURE
                    sp_d = {"none": {}, "hp": {"hp": 32}, "bulk": {"hp": 32, def_stat: 32}}[d_tier]
                    attacker = PokemonSet(
                        a.name, nature=nature_a, ability=ability_a, item=a.stone, sp=sp_a,
                        boosts={"atk": attacker_atk} if attacker_atk else {},
                    )
                    defender = PokemonSet(
                        d.name, nature=NEUTRAL_NATURE, ability=ability_d, item=d.stone, sp=sp_d,
                        boosts={"atk": defender_atk} if defender_atk else {},
                    )
                    _, d_info, group_results = _calc.calculate_many(attacker, defender, group, field)
                    hp = d_info["max_hp"]
                    results.extend(group_results)
                if not results:
                    continue
                strongest = lambda rs: max(rs, key=lambda r: (r.min, r.max), default=None)  # noqa: E731
                best = strongest(results)
                reliable = strongest([r for r in results if acc[r.move] >= RELIABLE_ACCURACY])
                rows.append({
                    "attacker": a.name,
                    "defender": d.name,
                    "attacker_baseline": a_tier,
                    "defender_baseline": d_tier,
                    "best_move": best.move,
                    "move_type": best.type,
                    "accuracy": acc[best.move],
                    "min_percent": best.min_percent,
                    "max_percent": best.max_percent,
                    "guaranteed_ohko": best.min >= hp,
                    "possible_ohko": best.max >= hp,
                    "reliable_move": reliable.move if reliable else "",
                    "reliable_min_percent": reliable.min_percent if reliable else 0,
                    "reliable_max_percent": reliable.max_percent if reliable else 0,
                    "reliable_guaranteed_ohko": bool(reliable) and reliable.min >= hp,
                    "weather": weather or "",
                    "terrain": terrain or "",
                    "notes": "; ".join(notes),
                })
    return rows


def rank(rows: list[dict], megas: list[Mega]) -> list[dict]:
    counts: dict[tuple, dict] = defaultdict(lambda: {"guaranteed": 0, "possible": 0, "reliable": 0})
    for r in rows:
        key = (r["attacker"], r["attacker_baseline"], r["defender_baseline"])
        counts[key]["guaranteed"] += r["guaranteed_ohko"]
        counts[key]["possible"] += r["possible_ohko"]
        counts[key]["reliable"] += r["reliable_guaranteed_ohko"]
    ranking = []
    for m in megas:
        row = {"mega": m.name, "ability": m.ability}
        for a_tier in ATTACKER_TIERS:
            for d_tier in DEFENDER_TIERS:
                c = counts[(m.name, a_tier, d_tier)]
                row[f"{a_tier} vs {d_tier}: guaranteed"] = c["guaranteed"]
                row[f"{a_tier} vs {d_tier}: possible"] = c["possible"]
                row[f"{a_tier} vs {d_tier}: reliable"] = c["reliable"]
        ranking.append(row)
    key = "max+ vs hp: guaranteed"
    ranking.sort(key=lambda r: (-r[key], -r["max+ vs hp: possible"], r["mega"]))
    return ranking


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doubles", action="store_true", help="doubles: spread moves take 0.75x damage")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument(
        "--exclude", default="",
        help="comma-separated moves to leave out; 'recharge' means Hyper Beam, Giga Impact and the like",
    )
    args = parser.parse_args()

    start = time.time()
    excluded = set()
    for name in filter(None, (m.strip() for m in args.exclude.split(","))):
        excluded |= RECHARGE_MOVES if name.lower() == "recharge" else {name}
    megas = load_megas(Calculator(), excluded)
    with Pool(args.processes, initializer=_worker_init) as pool:
        chunks = pool.map(attacker_rows, [(a, megas, args.doubles) for a in megas], chunksize=1)
    rows = [r for chunk in chunks for r in chunk]

    mode = ("doubles" if args.doubles else "singles") + ("_no-recharge" if excluded >= RECHARGE_MOVES else "")
    args.out.mkdir(parents=True, exist_ok=True)
    matchups = args.out / f"mega_ohko_{mode}_matchups.csv"
    ranking_path = args.out / f"mega_ohko_{mode}_ranking.csv"
    write_csv(matchups, rows)
    ranking = rank(rows, megas)
    write_csv(ranking_path, ranking)

    print(f"{len(megas)} Megas, {len(rows)} matchup rows ({mode}) in {time.time() - start:.0f}s")
    print(f"Wrote {matchups} and {ranking_path}\n")
    print("Guaranteed OHKOs, max+ attacker vs 32 HP defender (out of 81):")
    print(f"  {'':20} {'any':>4} {'90%+':>5} {'possible':>9}")
    for r in ranking[:20]:
        print(f"  {r['mega']:20} {r['max+ vs hp: guaranteed']:4} {r['max+ vs hp: reliable']:5}"
              f" {r['max+ vs hp: possible']:9}")


if __name__ == "__main__":
    main()
