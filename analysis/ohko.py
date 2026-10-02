"""Which Pokemon one-hit KO the most of the field?

Every Pokemon on the roster attacks every other one with every usable damaging
move, at three attacker and three defender baselines:

    attacker  max+   32 SP in the attacking stat, boosting nature
              max    32 SP in the attacking stat, neutral nature
              none   no investment, neutral nature
    defender  none   no investment, neutral nature
              hp     32 HP
              bulk   32 HP and 32 in the defensive stat the move hits

Items: Megas hold their Mega Stone. Other Pokemon attack holding Life Orb (the
strongest damage item in Champions, which has no Choice Band or Specs) and
defend holding nothing.

Abilities: every attacker/defender ability pairing is calculated. An OHKO only
counts if it works against every ability the defender could have, and each
attacker is ranked with whichever single ability OHKOs the most.

The field follows from the abilities: weather and terrain setters apply (when
both set weather, the slower Pokemon's wins because abilities activate in
speed order; Cloud Nine cancels it), Intimidate lowers a physical attacker's
Attack (with the usual immunities, Contrary/Defiant, and Competitive's +2 Sp.
Atk), Trace copies the opponent's ability, Queenly Majesty, Armor Tail and
Psychic Terrain block priority moves, Rivalry assumes the worse gender pairing,
and Sturdy and Disguise stop OHKOs from full HP (the calc ignores both). Ditto
is skipped: Imposter turns it into its opponent.
Moves that cannot OHKO on the first turn of a fresh matchup are left out (see
EXCLUDED_MOVES).

Each matchup records the strongest move, the strongest "reliable" move (at least
90% accuracy after weather and abilities) and the strongest reliable move that
does not need a recharge turn. Crits and speed order are otherwise ignored.

Usage:
    python analysis/ohko.py                      # every legal Pokemon, singles
    python analysis/ohko.py --roster megas       # Megas only
    python analysis/ohko.py --doubles            # spread moves take 0.75x
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from champcalc import Calculator, Field, PokemonSet

ATTACKER_TIERS = ("max+", "max", "none")
DEFENDER_TIERS = ("none", "hp", "bulk")
TIER_PAIRS = [(a, d) for a in ATTACKER_TIERS for d in DEFENDER_TIERS]

ATTACKER_ITEM = "Life Orb"

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
    "Fling": "depends on the held item",
}
# Moves that force the user to skip its next turn.
RECHARGE_MOVES = {
    "Hyper Beam", "Giga Impact", "Blast Burn", "Frenzy Plant", "Hydro Cannon", "Rock Wrecker",
    "Meteor Assault",
}
# Charge for a turn unless the weather is right (Mega Sol counts as sun for its user).
SOLAR_MOVES = {"Solar Beam", "Solar Blade"}
RAIN_CHARGE_MOVES = {"Electro Shot"}
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
UNTRACEABLE = {"Trace", "Stance Change", "Disguise", "Zero to Hero", "Forecast", "Ice Face"}
MOLD_BREAKERS = {"Mold Breaker", "Teravolt", "Turboblaze"}
WEATHER_NEGATORS = {"Cloud Nine", "Air Lock"}
# Block priority moves aimed at the holder (Psychic Terrain does for grounded targets).
PRIORITY_BLOCKERS = {"Queenly Majesty", "Armor Tail", "Dazzling"}
NOT_GROUNDED = {"Levitate", "Eelevate"}
# Imposter copies its opponent, which the calc cannot model.
SKIPPED_SPECIES = {"Ditto": "Imposter transforms it into its opponent"}

RELIABLE_ACCURACY = 90
# Weather that changes accuracy: (move, weather) -> accuracy.
WEATHER_ACCURACY = {
    ("Blizzard", "Snow"): 100,
    ("Thunder", "Rain"): 100, ("Hurricane", "Rain"): 100,
    ("Thunder", "Sun"): 50, ("Hurricane", "Sun"): 50,
}
# The calc assumes every hit lands, but these roll accuracy per hit.
PER_HIT_ACCURACY = {"Population Bomb": 10, "Triple Axel": 3, "Triple Kick": 3}

BOOSTING_NATURE = {"atk": "Adamant", "spa": "Modest", "def": "Bold"}
NEUTRAL_NATURE = "Serious"


@dataclass(frozen=True)
class Entry:
    name: str
    abilities: tuple[str, ...]
    mega: bool
    stone: str | None
    speed: int  # with no Speed investment, used for weather order
    moves: tuple[str, ...]
    types: tuple[str, ...] = ()
    gender: str | None = None  # fixed gender ('M', 'F', 'N'), or None when it can be either

    @property
    def attack_item(self) -> str | None:
        return self.stone if self.mega else ATTACKER_ITEM

    @property
    def defense_item(self) -> str | None:
        return self.stone if self.mega else None


def intimidate_stage(target_ability: str) -> int:
    if target_ability in INTIMIDATE_IMMUNE:
        return 0
    if target_ability in INTIMIDATE_RAISES:
        return 1
    return -1


def rivalry_genders(attacker: str | None, defender: str | None) -> tuple[str | None, str | None]:
    """Genders for a Rivalry attacker: it matches a single-gender target, but a
    target that can be either gender is assumed to be the opposite one (0.75x)."""
    if defender in ("M", "F"):
        return attacker or defender, defender
    if defender == "N":
        return attacker, "N"
    mine = attacker if attacker in ("M", "F") else "M"
    return mine, {"M": "F", "F": "M"}[mine]


def effective_abilities(ability_a: str, ability_d: str) -> tuple[str, str]:
    """Abilities after Trace copies the opponent's."""
    traced_a = ability_d if ability_a == "Trace" and ability_d not in UNTRACEABLE else ability_a
    traced_d = ability_a if ability_d == "Trace" and ability_a not in UNTRACEABLE else ability_d
    return traced_a, traced_d


def resolve_condition(speed_a: int, speed_d: int, ability_a: str, ability_d: str, table: dict[str, str]):
    """The weather (or terrain) in play and whether it was a speed contest.

    Both abilities activate in speed order, so the slower Pokemon's setting is
    the one left standing. A speed tie is a coin flip; the attacker's is used.
    """
    set_a, set_d = table.get(ability_a), table.get(ability_d)
    if set_a and set_d and set_a != set_d:
        if speed_d < speed_a:
            return set_d, "contested"
        return set_a, "contested" if speed_a < speed_d else "contested (speed tie)"
    return set_a or set_d, ""


def accuracy(move: str, base: int | bool, category: str, weather: str | None,
             ability_a: str, ability_d: str) -> float:
    """Chance the move fully lands, as a percentage."""
    if base is True or "No Guard" in (ability_a, ability_d):
        return 100
    acc = WEATHER_ACCURACY.get((move, weather), base)
    if ability_a == "Hustle" and category == "Physical":
        acc *= 0.8
    if ability_a == "Compound Eyes":
        acc *= 1.3
    acc = min(acc, 100)
    hits = 1 if ability_a == "Skill Link" else PER_HIT_ACCURACY.get(move, 1)
    return round(100 * (acc / 100) ** hits, 1)


def load_roster(calc: Calculator, roster: str, excluded: set[str]) -> list[Entry]:
    names = [n for n in calc.legal_species if n not in SKIPPED_SPECIES]
    if roster == "megas":
        names = [n for n in names if "-Mega" in n]
    move_info: dict[str, dict] = {}
    entries = []
    for name in names:
        info = calc.species(name)
        moves = []
        for move in calc.learnset(name):
            if move not in move_info:
                move_info[move] = calc.move(move)
            if move_info[move]["category"] != "Status" and move not in EXCLUDED_MOVES and move not in excluded:
                moves.append(move)
        entries.append(Entry(
            name=name,
            abilities=tuple(info["abilities"]) or ("",),
            mega="-Mega" in name,
            stone=info["mega_stone"],
            speed=info["base_stats"]["spe"] + 20,
            moves=tuple(moves),
            types=tuple(info["types"]),
            gender=info["gender"],
        ))
    return entries


@dataclass
class Setup:
    """One ability pairing of a matchup and the field it produces."""

    ability_a: str
    ability_d: str
    effective_a: str
    effective_d: str
    field: Field
    weather: str | None
    terrain: str | None
    moves: list[int]  # indices into the attacker's move list
    accuracy: dict[int, float]
    attacker_boosts: dict[str, int]
    defender_atk: int
    genders: tuple[str | None, str | None]
    block: str  # "", "Sturdy" or "Disguise": stops single-hit OHKOs from full HP
    notes: list[str]


_calc: Calculator | None = None
_info: dict[str, dict] = {}
_roster: list[Entry] = []
_doubles = False


def _worker_init(roster: list[Entry], doubles: bool) -> None:
    global _calc, _roster, _doubles
    _calc, _roster, _doubles = Calculator(), roster, doubles


def _move(move: str) -> dict:
    if move not in _info:
        _info[move] = _calc.move(move)
    return _info[move]


def setups_for(a: Entry, d: Entry) -> list[Setup]:
    out = []
    for ability_a in a.abilities:
        for ability_d in d.abilities:
            eff_a, eff_d = effective_abilities(ability_a, ability_d)
            notes = []
            if eff_a != ability_a:
                notes.append(f"{a.name} traces {eff_a}")
            if eff_d != ability_d:
                notes.append(f"{d.name} traces {eff_d}")
            weather, weather_note = resolve_condition(a.speed, d.speed, eff_a, eff_d, WEATHER_ABILITIES)
            if weather_note:
                notes.append(f"{weather} ({weather_note})")
            if weather and {eff_a, eff_d} & WEATHER_NEGATORS:
                notes.append(f"{weather} cancelled by {', '.join(sorted({eff_a, eff_d} & WEATHER_NEGATORS))}")
                weather = None
            terrain, _ = resolve_condition(a.speed, d.speed, eff_a, eff_d, TERRAIN_ABILITIES)
            sun = weather == "Sun" or eff_a == "Mega Sol"
            grounded = "Flying" not in d.types and eff_d not in NOT_GROUNDED
            no_priority = eff_d in PRIORITY_BLOCKERS or (terrain == "Psychic" and grounded)
            if no_priority:
                notes.append("priority moves blocked")
            moves = [
                i for i, m in enumerate(a.moves)
                if (m not in SOLAR_MOVES or sun)
                and (m not in RAIN_CHARGE_MOVES or weather == "Rain")
                and (m not in TERRAIN_MOVES or terrain)
                and not (no_priority and _move(m)["priority"] > 0)
            ]
            attacker_boosts = {}
            if eff_d == "Intimidate":
                attacker_boosts["atk"] = intimidate_stage(eff_a)
                if eff_a == "Competitive":
                    attacker_boosts["spa"] = 2
                attacker_boosts = {k: v for k, v in attacker_boosts.items() if v}
                if attacker_boosts:
                    notes.append("Intimidate: attacker " + ", ".join(f"{k} {v:+d}" for k, v in attacker_boosts.items()))
            defender_atk = intimidate_stage(eff_d) if eff_a == "Intimidate" else 0
            genders = rivalry_genders(a.gender, d.gender) if eff_a == "Rivalry" else (None, None)
            if eff_a == "Rivalry":
                notes.append(f"Rivalry: {genders[0]} vs {genders[1]}")
            block = eff_d if eff_d in ("Sturdy", "Disguise") and eff_a not in MOLD_BREAKERS else ""
            if block:
                notes.append(f"{block} blocks a single-hit OHKO")
            out.append(Setup(
                ability_a, ability_d, eff_a, eff_d,
                Field(weather=weather, terrain=terrain, doubles=_doubles),
                weather, terrain, moves,
                {i: accuracy(a.moves[i], _move(a.moves[i])["accuracy"], _move(a.moves[i])["category"],
                             weather, eff_a, eff_d) for i in moves},
                attacker_boosts, defender_atk, genders, block, notes,
            ))
    return out


def evaluate(a: Entry, d: Entry) -> dict[tuple, dict]:
    """Best moves for every (attacker ability, defender ability, tier pair)."""
    setups = setups_for(a, d)
    jobs, keys = [], []
    for s_index, s in enumerate(setups):
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i in s.moves:
            info = _move(a.moves[i])
            groups[(info["offensive_stat"], info["defensive_stat"], info["uses_target_attack"])].append(i)
        for a_tier, d_tier in TIER_PAIRS:
            for (off_stat, def_stat, uses_target), idx in groups.items():
                attacker = PokemonSet(
                    a.name, ability=s.effective_a or None, item=a.attack_item,
                    nature=BOOSTING_NATURE[off_stat] if a_tier == "max+" and not uses_target else NEUTRAL_NATURE,
                    sp={} if a_tier == "none" or uses_target else {off_stat: 32},
                    boosts=s.attacker_boosts, gender=s.genders[0],
                )
                defender = PokemonSet(
                    d.name, ability=s.effective_d or None, item=d.defense_item, nature=NEUTRAL_NATURE,
                    sp={"none": {}, "hp": {"hp": 32}, "bulk": {"hp": 32, def_stat: 32}}[d_tier],
                    boosts={"atk": s.defender_atk} if s.defender_atk else {}, gender=s.genders[1],
                )
                jobs.append((attacker, defender, s.field, idx))
                keys.append((s_index, a_tier, d_tier, idx))
    results: dict[tuple, list] = defaultdict(list)
    hps: dict[tuple, int] = {}
    if jobs:
        for (s_index, a_tier, d_tier, idx), (hp, damage) in zip(keys, _calc.calculate_batch(list(a.moves), jobs)):
            results[(s_index, a_tier, d_tier)].extend(zip(idx, damage))
            hps[(s_index, a_tier, d_tier)] = hp

    out = {}
    for s_index, s in enumerate(setups):
        for a_tier, d_tier in TIER_PAIRS:
            hp = hps.get((s_index, a_tier, d_tier))
            candidates = []
            for i, (lo, hi) in results.get((s_index, a_tier, d_tier), []):
                move = a.moves[i]
                multi = _move(move)["multi_hit"] or s.effective_a == "Parental Bond"
                can_ohko = not s.block or (s.block == "Sturdy" and multi)
                candidates.append({
                    "move": move,
                    "min": lo, "max": hi,
                    "accuracy": s.accuracy[i],
                    "guaranteed": can_ohko and lo >= hp,
                    "possible": can_ohko and hi >= hp,
                    "recharge": move in RECHARGE_MOVES,
                })
            rank_key = lambda c: (c["guaranteed"], c["possible"], c["min"], c["max"])  # noqa: E731
            best = max(candidates, key=rank_key, default=None)
            reliable = [c for c in candidates if c["accuracy"] >= RELIABLE_ACCURACY]
            best_reliable = max(reliable, key=rank_key, default=None)
            best_no_recharge = max((c for c in reliable if not c["recharge"]), key=rank_key, default=None)
            out[(s.ability_a, s.ability_d, a_tier, d_tier)] = {
                "setup": s, "hp": hp, "best": best, "reliable": best_reliable, "no_recharge": best_no_recharge,
            }
    return out


def percent(damage: int, hp: int | None) -> float:
    return round(100 * damage / hp, 1) if hp else 0.0


def attacker_task(index: int) -> tuple[list[dict], dict]:
    """All matchups for one attacker; picks the ability that OHKOs the most."""
    a = _roster[index]
    flag = lambda r, key: bool(r[key]) and r[key]["guaranteed"]  # noqa: E731
    # per (attacker ability, tier pair): defender -> worst-case summary
    summaries: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for d in _roster:
        if d.name == a.name:
            continue
        evaluated = evaluate(a, d)
        for ability_a in a.abilities:
            for a_tier, d_tier in TIER_PAIRS:
                per_d = [evaluated[(ability_a, ability_d, a_tier, d_tier)] for ability_d in d.abilities]
                per_d = [r for r in per_d if r["best"]]
                if not per_d:
                    continue
                worst = min(per_d, key=lambda r: (r["best"]["guaranteed"], r["best"]["possible"],
                                                   r["best"]["min"], r["best"]["max"]))
                summaries[(ability_a, a_tier, d_tier)][d.name] = {
                    "worst": worst,
                    "guaranteed": all(r["best"]["guaranteed"] for r in per_d),
                    "possible": all(r["best"]["possible"] for r in per_d),
                    "reliable": all(flag(r, "reliable") for r in per_d),
                    "no_recharge": all(flag(r, "no_recharge") for r in per_d),
                }

    rows = []
    ranking = {"pokemon": a.name, "mega": a.mega}
    for a_tier, d_tier in TIER_PAIRS:
        def score(ability_a):
            s = summaries.get((ability_a, a_tier, d_tier), {})
            return tuple(sum(v[k] for v in s.values()) for k in ("reliable", "guaranteed", "possible"))
        ability_a = max(a.abilities, key=score)
        summary = summaries.get((ability_a, a_tier, d_tier), {})
        prefix = f"{a_tier} vs {d_tier}"
        ranking[f"{prefix}: ability"] = ability_a
        for key in ("reliable", "no_recharge", "guaranteed", "possible"):
            ranking[f"{prefix}: {key}"] = sum(v[key] for v in summary.values())
        for d_name, v in summary.items():
            w = v["worst"]
            s, hp, best, rel, nr = w["setup"], w["hp"], w["best"], w["reliable"], w["no_recharge"]
            rows.append({
                "attacker": a.name,
                "attacker_ability": ability_a,
                "attacker_item": a.attack_item or "",
                "defender": d_name,
                "defender_ability": s.ability_d,
                "attacker_baseline": a_tier,
                "defender_baseline": d_tier,
                "best_move": best["move"],
                "accuracy": best["accuracy"],
                "min_percent": percent(best["min"], hp),
                "max_percent": percent(best["max"], hp),
                "guaranteed_ohko": v["guaranteed"],
                "possible_ohko": v["possible"],
                "reliable_move": rel["move"] if rel else "",
                "reliable_min_percent": percent(rel["min"], hp) if rel else 0,
                "reliable_ohko": v["reliable"],
                "no_recharge_move": nr["move"] if nr else "",
                "no_recharge_min_percent": percent(nr["min"], hp) if nr else 0,
                "no_recharge_ohko": v["no_recharge"],
                "weather": s.weather or "",
                "terrain": s.terrain or "",
                "notes": "; ".join(s.notes),
            })
    return rows, ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roster", choices=("all", "megas"), default="all")
    parser.add_argument("--doubles", action="store_true", help="doubles: spread moves take 0.75x damage")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, help="only run the first N attackers (for testing)")
    parser.add_argument(
        "--exclude", default="",
        help="comma-separated moves to leave out; 'recharge' means Hyper Beam, Giga Impact and the like",
    )
    args = parser.parse_args()

    excluded = set()
    for name in filter(None, (m.strip() for m in args.exclude.split(","))):
        excluded |= RECHARGE_MOVES if name.lower() == "recharge" else {name}
    roster = load_roster(Calculator(), args.roster, excluded)
    attackers = range(len(roster))[: args.limit]

    start = time.time()
    rows, ranking = [], []
    with Pool(args.processes, initializer=_worker_init, initargs=(roster, args.doubles)) as pool:
        for done, (attacker_rows, entry) in enumerate(pool.imap_unordered(attacker_task, attackers), 1):
            rows.extend(attacker_rows)
            ranking.append(entry)
            elapsed = time.time() - start
            print(f"\r{done}/{len(attackers)} attackers, {elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / done * (len(attackers) - done) / 60:.0f} min left", end="", file=sys.stderr)
    print(file=sys.stderr)

    name = f"ohko_{args.roster}_{'doubles' if args.doubles else 'singles'}"
    if excluded:
        name += "_excluding-" + ("recharge" if excluded == RECHARGE_MOVES else str(len(excluded)))
    args.out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["attacker"], r["attacker_baseline"], r["defender_baseline"], r["defender"]))
    with gzip.open(args.out / f"{name}_matchups.csv.gz", "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranking.sort(key=lambda r: (-r["max+ vs hp: reliable"], -r["max+ vs hp: guaranteed"], r["pokemon"]))
    with (args.out / f"{name}_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)

    print(f"{len(roster)} Pokemon, {len(rows)} matchup rows in {(time.time() - start) / 60:.1f} min")
    print(f"Wrote {args.out / name}_matchups.csv.gz and _ranking.csv\n")
    others = len(roster) - 1
    print(f"Reliable OHKOs, max+ attacker vs 32 HP defender (out of {others}):")
    print(f"  {'':24} {'ability':16} {'90%+':>5} {'no-rchg':>8} {'any':>5}")
    for r in ranking[:25]:
        print(f"  {r['pokemon']:24} {r['max+ vs hp: ability']:16} {r['max+ vs hp: reliable']:5}"
              f" {r['max+ vs hp: no_recharge']:8} {r['max+ vs hp: guaranteed']:5}")


if __name__ == "__main__":
    main()
