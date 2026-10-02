"""Meta-weighted matchup features for every legal Pokemon (doubles).

The meta is every Pokemon with at least META_MIN_USAGE usage on the Showdown
best-of-1 ladder at 1760+, weighted by that usage, each with its most common
set (ability, item, nature + Stat Points, moves) from all-rating ladder data.
Every legal Pokemon is then scored against it using its own most common set,
or a synthesized one when it has no ladder data:

    offense   share of the meta it OHKOs / 2HKOs, and average damage dealt,
              with its common moves ("set") and with any legal move ("potential")
    defense   share of the meta that OHKOs / 2HKOs it, and average damage taken
    speed     share of the meta it outspeeds, under Tailwind, and under Trick Room
    utility   access to support tools (Fake Out, Tailwind, Trick Room,
              redirection, Intimidate, weather/terrain, ...)

KO shares are expected values: each KO counts with its move's accuracy. Calcs
are doubles (spread moves take 0.75x); weather/terrain, Intimidate, Trace,
Sturdy and Disguise follow the same rules as analysis/ohko.py.

Usage:
    python analysis/meta_matchups.py                 # current regulation
    python analysis/meta_matchups.py --regulation M-B
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import ohko
from champcalc import CalcError, Calculator, Field, PokemonSet

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"

META_MIN_USAGE = 0.01
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
# Halve damage only at full HP; the calc applies them to every hit.
FIRST_HIT_HALVED = {"Multiscale", "Shadow Shield"}
# Showdown ladder names the calc spells differently.
CALC_NAMES = {"Aegislash": "Aegislash-Both"}

# Support tools: tag -> moves or abilities that grant it.
UTILITY_MOVES = {
    "fake_out": {"Fake Out"},
    "tailwind": {"Tailwind"},
    "trick_room": {"Trick Room"},
    "redirection": {"Follow Me", "Rage Powder"},
    "helping_hand": {"Helping Hand"},
    "wide_guard": {"Wide Guard", "Quick Guard"},
    "speed_drop": {"Icy Wind", "Electroweb", "Rock Tomb", "Bulldoze", "Scary Face"},
    "pivot": {"U-turn", "Volt Switch", "Flip Turn", "Parting Shot", "Teleport"},
    "screens": {"Reflect", "Light Screen", "Aurora Veil"},
    "disruption": {"Taunt", "Encore", "Disable", "Imprison", "Quash"},
    "sleep": {"Spore", "Sleep Powder", "Hypnosis", "Yawn", "Dark Void"},
    "attack_drop": {"Snarl", "Parting Shot", "Noble Roar", "Memento", "Charm", "Feather Dance"},
    "healing": {"Pollen Puff", "Life Dew", "Heal Pulse", "Floral Healing", "Wish"},
    "setup": {"Swords Dance", "Nasty Plot", "Dragon Dance", "Calm Mind", "Bulk Up", "Quiver Dance",
              "Shell Smash", "Belly Drum", "Coil", "Shift Gear", "Victory Dance", "Tidy Up"},
}
UTILITY_ABILITIES = {
    "intimidate": {"Intimidate"},
    "weather": set(ohko.WEATHER_ABILITIES),
    "terrain": set(ohko.TERRAIN_ABILITIES),
    "redirection": {"Lightning Rod", "Storm Drain"},
    "prankster": {"Prankster"},
}


@dataclass
class MonSet:
    name: str
    ability: str
    item: str | None
    nature: str
    sp: dict[str, int]
    moves: list[str]           # damaging moves of its common set
    potential: list[str]       # every usable damaging move it can learn
    types: tuple[str, ...]
    speed: int = 0
    observed: bool = True
    gender: str | None = None
    utility: dict[str, bool] = field(default_factory=dict)

    def to_set(self, boosts: dict[str, int] | None = None, gender: str | None = None) -> PokemonSet:
        return PokemonSet(self.name, ability=self.ability or None, item=self.item, nature=self.nature,
                          sp=self.sp, boosts=boosts or {}, gender=gender)


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_spread(spread: str) -> tuple[str, dict[str, int]] | None:
    """'Adamant:32/32/0/0/2/0' -> ('Adamant', {...}); None if not a legal SP spread."""
    nature, _, values = spread.partition(":")
    try:
        sp = dict(zip(STATS, map(int, values.split("/"))))
    except ValueError:
        return None
    if len(sp) != 6 or max(sp.values()) > 32 or sum(sp.values()) > 66:
        return None
    return nature, {k: v for k, v in sp.items() if v}


class SetBuilder:
    def __init__(self, calc: Calculator, db: sqlite3.Connection, month: str):
        self.calc, self.db, self.month = calc, db, month
        self.move_names = {to_id(m): m for m in calc.all_moves}
        self.item_names = {to_id(i): i for i in calc.all_items}
        self.ability_names = {to_id(a): a for a in calc.all_abilities}
        self.move_info: dict[str, dict] = {}

    def info(self, move: str) -> dict:
        if move not in self.move_info:
            self.move_info[move] = self.calc.move(move)
        return self.move_info[move]

    def details(self, ladder_name: str, kind: str) -> list[tuple[str, float]]:
        return self.db.execute(
            """SELECT value, share FROM showdown_details WHERE regulation=? AND month=? AND ladder='bo1'
               AND rating=0 AND pokemon=? AND kind=? ORDER BY share DESC""",
            (self.calc.regulation, self.month, ladder_name, kind),
        ).fetchall()

    def build(self, name: str, ladder_name: str | None) -> MonSet:
        info = self.calc.species(name)
        learnset = self.calc.learnset(name)
        potential = [m for m in learnset
                     if self.info(m)["category"] != "Status" and m not in ohko.EXCLUDED_MOVES]
        legal_abilities = info["abilities"] or [""]
        utility = {tag: bool(moves & set(learnset)) for tag, moves in UTILITY_MOVES.items()}
        for tag, abilities in UTILITY_ABILITIES.items():
            utility[tag] = utility.get(tag, False) or bool(abilities & set(legal_abilities))

        abilities = items = spreads = moves = []
        if ladder_name:
            abilities = self.details(ladder_name, "ability")
            items = self.details(ladder_name, "item")
            spreads = self.details(ladder_name, "spread")
            moves = self.details(ladder_name, "move")
        ability = next((self.ability_names[a] for a, _ in abilities if self.ability_names.get(a) in legal_abilities),
                       legal_abilities[0])
        item = info["mega_stone"] or next((self.item_names[i] for i, _ in items if i in self.item_names), None)
        parsed = next(filter(None, (parse_spread(s) for s, _ in spreads)), None)
        common = [self.move_names[m] for m, share in moves if m in self.move_names and share >= 0.2]
        common = [m for m in common if m in potential][:4]
        observed = bool(parsed and common)
        if not observed:
            # Synthesized set: invest in the better attacking stat and Speed.
            stats = info["base_stats"]
            off = "atk" if stats["atk"] >= stats["spa"] else "spa"
            parsed = ({"atk": "Adamant", "spa": "Modest"}[off], {off: 32, "spe": 32})
            common = sorted(potential, key=lambda m: -self.info(m)["base_power"])[:4]
        nature, sp = parsed
        return MonSet(name, ability, item, nature, sp, common, potential, tuple(info["types"]),
                      observed=observed, gender=info["gender"], utility=utility)


def speed_of(calc: Calculator, mon: MonSet) -> int:
    # Damage calcs report stats; use an attack into a neutral dummy target.
    attacker, _, _ = calc.calculate_many(mon.to_set(), PokemonSet("Snorlax"), ["Earthquake"])
    return attacker["stats"]["spe"]


# --- Matchups -------------------------------------------------------------------------------------

def field_for(a: MonSet, d: MonSet, doubles: bool = True):
    """Field and stat changes when `a` attacks `d`, from their abilities."""
    eff_a, eff_d = ohko.effective_abilities(a.ability, d.ability)
    weather, _ = ohko.resolve_condition(a.speed, d.speed, eff_a, eff_d, ohko.WEATHER_ABILITIES)
    if {eff_a, eff_d} & ohko.WEATHER_NEGATORS:
        weather = None
    terrain, _ = ohko.resolve_condition(a.speed, d.speed, eff_a, eff_d, ohko.TERRAIN_ABILITIES)
    a_boosts = {}
    if eff_d == "Intimidate":
        a_boosts["atk"] = ohko.intimidate_stage(eff_a)
        if eff_a == "Competitive":
            a_boosts["spa"] = 2
    d_boosts = {"atk": ohko.intimidate_stage(eff_d)} if eff_a == "Intimidate" else {}
    genders = ohko.rivalry_genders(a.gender, d.gender) if eff_a == "Rivalry" else (None, None)
    block = eff_d if eff_d in ("Sturdy", "Disguise") and eff_a not in ohko.MOLD_BREAKERS else ""
    attacker = PokemonSet(a.name, ability=eff_a or None, item=a.item, nature=a.nature, sp=a.sp,
                          boosts={k: v for k, v in a_boosts.items() if v}, gender=genders[0])
    defender = PokemonSet(d.name, ability=eff_d or None, item=d.item, nature=d.nature, sp=d.sp,
                          boosts={k: v for k, v in d_boosts.items() if v}, gender=genders[1])
    return attacker, defender, Field(weather=weather, terrain=terrain, doubles=doubles), weather, terrain, eff_a, block


def usable(moves: list[str], weather: str | None, terrain: str | None, eff_a: str) -> list[str]:
    sun = weather == "Sun" or eff_a == "Mega Sol"
    return [m for m in moves
            if (m not in ohko.SOLAR_MOVES or sun)
            and (m not in ohko.RAIN_CHARGE_MOVES or weather == "Rain")
            and (m not in ohko.TERRAIN_MOVES or terrain)]


_calc: Calculator | None = None
_builder_info: dict[str, dict] = {}
_meta: list[tuple[MonSet, float]] = []


def _init(regulation: str, meta: list[tuple[MonSet, float]]) -> None:
    global _calc, _meta
    _calc, _meta = Calculator(regulation), meta


def _move(move: str) -> dict:
    if move not in _builder_info:
        _builder_info[move] = _calc.move(move)
    return _builder_info[move]


def ko_scores(results, moves, hp, eff_a, weather, eff_d, block) -> tuple[float, float, float]:
    """(expected OHKO, expected 2HKO, best guaranteed damage %) over the given moves."""
    best_ohko = best_2hko = best_damage = 0.0
    for move, (lo, hi) in zip(moves, results):
        info = _move(move)
        acc = ohko.accuracy(move, info["accuracy"], info["category"], weather, eff_a, eff_d) / 100
        multi = info["multi_hit"] or eff_a == "Parental Bond"
        can_ohko = not block or (block == "Sturdy" and multi)
        if lo >= hp and can_ohko:
            best_ohko = max(best_ohko, acc)
        # Two hits: Disguise eats the first, so the second must KO alone;
        # Multiscale only halves the first, so the second does double.
        if block == "Disguise":
            two_hits = lo
        elif eff_d in FIRST_HIT_HALVED:
            two_hits = 3 * lo
        else:
            two_hits = 2 * lo
        if two_hits >= hp:
            best_2hko = max(best_2hko, acc * acc)
        best_damage = max(best_damage, min(100.0, 100 * lo / hp))
    return best_ohko, best_2hko, best_damage


def evaluate(mon: MonSet) -> dict:
    jobs, plan = [], []
    move_list = list(dict.fromkeys(mon.potential + [m for meta, _ in _meta for m in meta.moves]))
    index = {m: i for i, m in enumerate(move_list)}
    for meta, weight in _meta:
        if meta.name == mon.name:
            continue
        # Offense: mon attacks meta with every usable move it learns.
        attacker, defender, fld, weather, terrain, eff_a, block = field_for(mon, meta)
        moves = usable(mon.potential, weather, terrain, eff_a)
        if moves:
            jobs.append((attacker, defender, fld, [index[m] for m in moves]))
            plan.append(("off", meta, weight, moves, weather, terrain, eff_a, defender.ability, block))
        # Defense: meta attacks mon with its common moves.
        attacker, defender, fld, weather, terrain, eff_a, block = field_for(meta, mon)
        moves = usable(meta.moves, weather, terrain, eff_a)
        if moves:
            jobs.append((attacker, defender, fld, [index[m] for m in moves]))
            plan.append(("def", meta, weight, moves, weather, terrain, eff_a, defender.ability, block))
    results = _calc.calculate_batch(move_list, jobs) if jobs else []

    sums = dict.fromkeys(["off_ohko_set", "off_2hko_set", "off_damage_set", "off_ohko_potential",
                          "off_2hko_potential", "off_damage_potential", "def_ohkod", "def_2hkod",
                          "def_damage_taken"], 0.0)
    total_weight = sum(w for meta, w in _meta if meta.name != mon.name) or 1
    for (side, meta, weight, moves, weather, terrain, eff_a, eff_d, block), (hp, damage) in zip(plan, results):
        if side == "off":
            o, t, dmg = ko_scores(damage, moves, hp, eff_a, weather, eff_d, block)
            sums["off_ohko_potential"] += weight * o
            sums["off_2hko_potential"] += weight * t
            sums["off_damage_potential"] += weight * dmg
            set_pairs = [(m, r) for m, r in zip(moves, damage) if m in mon.moves]
            if set_pairs:
                o, t, dmg = ko_scores([r for _, r in set_pairs], [m for m, _ in set_pairs], hp,
                                      eff_a, weather, eff_d, block)
                sums["off_ohko_set"] += weight * o
                sums["off_2hko_set"] += weight * t
                sums["off_damage_set"] += weight * dmg
        else:
            o, t, dmg = ko_scores(damage, moves, hp, eff_a, weather, eff_d, block)
            sums["def_ohkod"] += weight * o
            sums["def_2hkod"] += weight * t
            sums["def_damage_taken"] += weight * dmg
    row = {"pokemon": mon.name, "observed_set": mon.observed, "ability": mon.ability, "item": mon.item or "",
           "spread": f"{mon.nature} " + "/".join(str(mon.sp.get(s, 0)) for s in STATS),
           "set_moves": ", ".join(mon.moves), "speed": mon.speed}
    row.update({k: round(v / total_weight, 4) for k, v in sums.items()})
    others = [(m, w) for m, w in _meta if m.name != mon.name]
    row["speed_outspeeds"] = round(sum(w * ((mon.speed > m.speed) + 0.5 * (mon.speed == m.speed))
                                       for m, w in others) / total_weight, 4)
    row["speed_tailwind"] = round(sum(w * ((2 * mon.speed > m.speed) + 0.5 * (2 * mon.speed == m.speed))
                                      for m, w in others) / total_weight, 4)
    row["speed_trick_room"] = round(sum(w * ((mon.speed < m.speed) + 0.5 * (mon.speed == m.speed))
                                        for m, w in others) / total_weight, 4)
    row.update({f"util_{k}": v for k, v in mon.utility.items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regulation", help="default: the current one")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()

    calc = Calculator(args.regulation)
    db = sqlite3.connect(DB)
    month = db.execute("SELECT MAX(month) FROM showdown_usage WHERE regulation=?", (calc.regulation,)).fetchone()[0]
    if not month:
        raise SystemExit(f"No ladder data for {calc.regulation}; run analysis/ingest.py first")
    usage = dict(db.execute(
        """SELECT pokemon, usage FROM showdown_usage WHERE regulation=? AND month=? AND ladder='bo1'
           AND rating=1760""", (calc.regulation, month)).fetchall())
    # Keyed by id, so "Sirfetch'd" and "Sirfetch’d" match.
    usage_by_id = {to_id(CALC_NAMES.get(n, n)): u for n, u in usage.items()}
    ladder_names = {to_id(CALC_NAMES.get(n, n)): n for n in usage}
    builder = SetBuilder(calc, db, month)

    mons = []
    for name in calc.legal_species:
        if name in ohko.SKIPPED_SPECIES:
            continue
        try:
            mon = builder.build(name, ladder_names.get(to_id(name)))
            mon.speed = speed_of(calc, mon)
        except CalcError as e:
            print(f"skipping {name}: {e}", file=sys.stderr)
            continue
        mons.append(mon)
    by_id = {to_id(m.name): m for m in mons}
    meta = [(by_id[i], u) for i, u in usage_by_id.items() if u >= META_MIN_USAGE and i in by_id]
    print(f"{calc.regulation} {month}: {len(mons)} Pokemon ({sum(m.observed for m in mons)} with ladder sets), "
          f"meta of {len(meta)} covering {sum(u for _, u in meta) / 6:.0%} of team slots", file=sys.stderr)

    start = time.time()
    with Pool(args.processes, initializer=_init, initargs=(calc.regulation, meta)) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(evaluate, mons, chunksize=4), 1):
            rows.append(row)
            if i % 20 == 0:
                print(f"\r{i}/{len(mons)} Pokemon, {time.time() - start:.0f}s", end="", file=sys.stderr)
    print(file=sys.stderr)

    df = pd.DataFrame(rows).set_index("pokemon").sort_index()
    df["ladder_usage_1760"] = [usage_by_id.get(to_id(p), 0.0) for p in df.index]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"meta_matchups_{calc.regulation}.csv"
    df.to_csv(path)
    print(f"Wrote {path} ({len(df)} Pokemon) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
