"""Interaction graph: which Pokemon create conditions (weather, terrain, speed control, setup cover, ...) that other
Pokemon's sets benefit from, how much each benefit is worth by damage calcs, and whether teams that pair them
actually did better in tournaments.

For each common tournament set and each condition a partner can provide, the calcs compare the set with and without
the condition against the most used Pokemon (usage-weighted):
    ohko     share of opponents its best attack OHKOs (accuracy included)
    ko2      share it 2HKOs
    speed    share of opponents it outspeeds (under Trick Room: share it moves before)
    survive  share of the opponents' attacks it survives from full HP
Then the real-results side: the set's wins above player skill on teams with a provider vs without one.

    python analysis/interactions.py                 # validation + strongest under-played interactions
    python analysis/interactions.py --min-sets 15

Results go to analysis/output/interactions.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import doubles_sim as ds
from champcalc import CalcError, Field, PokemonSet

OUT = Path(__file__).parent / "output"
RECENT = ds.RECENT

# ---------------------------------------------------------------------------------------------------- conditions

WEATHER = {"Drought": "Sun", "Drizzle": "Rain", "Sand Stream": "Sand", "Snow Warning": "Snow",
           "Sunny Day": "Sun", "Rain Dance": "Rain", "Sandstorm": "Sand", "Snowscape": "Snow"}
TERRAIN = {"Electric Surge": "Electric", "Grassy Surge": "Grassy", "Psychic Surge": "Psychic", "Misty Surge": "Misty",
           "Seed Sower": "Grassy", "Electric Terrain": "Electric", "Grassy Terrain": "Grassy",
           "Psychic Terrain": "Psychic", "Misty Terrain": "Misty"}
MOVE_CONDITIONS = {"Tailwind": "Tailwind", "Trick Room": "Trick Room", "Follow Me": "redirection",
                   "Rage Powder": "redirection", "Fake Out": "Fake Out", "Helping Hand": "Helping Hand",
                   "Reflect": "screens", "Light Screen": "screens", "Aurora Veil": "screens"}
ABILITY_CONDITIONS = {"Friend Guard": "Friend Guard", "Armor Tail": "priority block",
                      "Queenly Majesty": "priority block", "Dazzling": "priority block"}
CONDITIONS = (["Sun", "Rain", "Sand", "Snow", "Electric Terrain", "Grassy Terrain", "Psychic Terrain", "Misty Terrain"]
              + sorted(set(MOVE_CONDITIONS.values()) | set(ABILITY_CONDITIONS.values())))
SPEED_ABILITY = {"Swift Swim": "Rain", "Chlorophyll": "Sun", "Sand Rush": "Sand", "Slush Rush": "Snow",
                 "Surge Surfer": "Electric"}
SEEDS = {"Electric Seed": ("Electric", "def"), "Grassy Seed": ("Grassy", "def"), "Psychic Seed": ("Psychic", "spd"),
         "Misty Seed": ("Misty", "spd")}
COVER = {"redirection", "Fake Out", "priority block"}  # conditions that buy a free setup turn


def provides(names: set[str]) -> set[str]:
    """Conditions a set provides to its partner, from its ability (as entered, or after Mega Evolution) and moves."""
    out = set()
    for x in names:
        if x in WEATHER:
            out.add(WEATHER[x])
        if x in TERRAIN:
            out.add(TERRAIN[x] + " Terrain")
        if x in MOVE_CONDITIONS:
            out.add(MOVE_CONDITIONS[x])
        if x in ABILITY_CONDITIONS:
            out.add(ABILITY_CONDITIONS[x])
    if "Psychic Terrain" in out:
        out.add("priority block")
    return out


def set_names(spec: ds.Spec) -> set[str]:
    return {spec.ability, spec.pre_ability, *spec.moves} - {None}


# ---------------------------------------------------------------------------------------------------- calcs

class Bench:
    """The most used Pokemon (recent events) with usage weights, to measure sets against."""

    def __init__(self, n=35):
        use = Counter(m for (m,) in ds.db().execute(
            "SELECT m.pokemon FROM team_members m JOIN tournaments t ON t.id = m.tournament "
            "WHERE t.regulation = ? AND t.date >= ?", (ds.REGULATION, RECENT)))
        self.opp = [(ds.real_spec(n), c) for n, c in use.most_common(n + 15) if ds.real_spec(n)][:n]
        self.total = sum(c for _, c in self.opp)


def pset(spec, boosts=None):
    return PokemonSet(spec.name, ability=spec.ability, item=spec.item, nature=spec.nature, sp=spec.sp,
                      boosts={k: v for k, v in (boosts or {}).items() if v})


def measure(spec: ds.Spec, bench: Bench, weather=None, terrain=None, boosts=None, def_boosts=None, speed_mult=1.0,
            helping=False, screens=False, friend_guard=False, trick_room=False) -> dict:
    """Usage-weighted ohko / speed / survive shares for `spec` under the given field (see module docstring)."""
    attacks = [m for m in spec.moves if ds.move_info(m)["category"] != "Status" and m != "Fake Out"]
    me_att = pset(spec, boosts)
    me_def = pset(spec, def_boosts)
    stats = ds.calc.calculate(me_att, PokemonSet("Snorlax"), attacks[0] if attacks else "Hyper Voice").attacker["stats"]
    my_speed = stats["spe"] * speed_mult * (1.5 if spec.item == "Choice Scarf" else 1)
    ohko = ko2 = speed = survive = 0.0
    surv_w = 0.0
    for o, c in bench.opp:
        field = Field(doubles=True, weather=weather, terrain=terrain, helping_hand=helping)
        best = best2 = 0.0
        for mv in attacks:
            boost = dict(boosts or {})
            if mv == "Electro Shot" and weather == "Rain":  # fires at once, after its +1 Special Attack
                boost["spa"] = boost.get("spa", 0) + 1
            try:
                r = ds.calc.calculate(pset(spec, boost), pset(o), mv, field)
            except CalcError:
                continue
            acc = ds.accuracy(mv, weather)
            best = max(best, sum(x >= r.defender["max_hp"] for x in r.rolls) / 16 * acc)
            best2 = max(best2, sum(2 * x >= r.defender["max_hp"] for x in r.rolls) / 16 * acc ** 2)
        ohko += c * best
        ko2 += c * best2
        ospe = o.speed * (1.5 if o.item == "Choice Scarf" else 1)
        speed += c * ((my_speed < ospe) if trick_room else (my_speed > ospe))
        dfield = Field(doubles=True, weather=weather, terrain=terrain, reflect=screens, light_screen=screens,
                       friend_guard=friend_guard)
        for mv in o.moves:
            if ds.move_info(mv)["category"] == "Status" or mv == "Fake Out":
                continue
            try:
                r = ds.calc.calculate(pset(o), me_def, mv, dfield)
            except CalcError:
                continue
            survive += c * sum(x < r.defender["max_hp"] for x in r.rolls) / 16
            surv_w += c
    return {"ohko": ohko / bench.total, "ko2": ko2 / bench.total, "speed": speed / bench.total,
            "survive": survive / max(surv_w, 1e-9)}


def condition_effect(spec: ds.Spec, cond: str, bench: Bench, base: dict) -> dict | None:
    """What a partner's `cond` does for `spec`: the change in each measure, or None if it has no mechanical link."""
    names = set_names(spec)
    kw: dict = {}
    if cond in ("Sun", "Rain", "Sand", "Snow"):
        kw["weather"] = cond
        if SPEED_ABILITY.get(spec.ability) == cond:
            kw["speed_mult"] = 2.0
    elif cond.endswith("Terrain"):
        terrain = cond.split()[0]
        kw["terrain"] = terrain
        seed = SEEDS.get(spec.item)
        if seed and seed[0] == terrain:
            kw["def_boosts"] = {seed[1]: 1}
            if spec.ability == "Unburden":
                kw["speed_mult"] = 2.0
        if SPEED_ABILITY.get(spec.ability) == terrain:
            kw["speed_mult"] = 2.0
    elif cond == "Tailwind":
        kw["speed_mult"] = 2.0
    elif cond == "Trick Room":
        kw["trick_room"] = True
    elif cond == "Helping Hand":
        kw["helping"] = True
    elif cond == "screens":
        kw["screens"] = True
    elif cond == "Friend Guard":
        kw["friend_guard"] = True
    elif cond in COVER:
        setup = next((ds.BOOST[m] for m in spec.moves if m in ds.BOOST), None)
        if not setup:
            return None
        kw["boosts"] = setup  # the value a free setup turn unlocks
    after = measure(spec, bench, **kw)
    eff = {k: after[k] - base[k] for k in ("ohko", "ko2", "speed", "survive")}
    if all(abs(v) < 1e-9 for v in eff.values()):
        return None
    return eff


def score(eff: dict) -> float:
    return eff["ohko"] + 0.5 * eff["ko2"] + 0.5 * eff["speed"] + 0.5 * eff["survive"]


# ---------------------------------------------------------------------------------------------------- real data

def load_sets(min_sets: int):
    """Common complete tournament sets: (species, item, ability, nature, moves) -> list of (tournament, player)."""
    rows = ds.db().execute(
        "SELECT m.tournament, m.player, m.pokemon, m.item, m.ability, m.nature, m.moves FROM team_members m "
        "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (ds.REGULATION,)).fetchall()
    team_provides, sets, members = defaultdict(lambda: defaultdict(set)), defaultdict(list), defaultdict(set)
    for t, p, name, item, ability, nature, moves in rows:
        mv = tuple(sorted(json.loads(moves))) if moves else ()
        members[(t, p)].add(name)
        for c in provides({ability or "", *mv}):
            team_provides[(t, p)][c].add(name)
        if nature and mv:
            sets[(name, item or "", ability or "", nature, mv)].append((t, p))
    common = {k: v for k, v in sets.items() if len(v) >= min_sets}
    return common, team_provides


def results():
    from signals import player_skill
    skill = player_skill(ds.db()).set_index(["tournament", "player"])["skill"].to_dict()
    rec = {}
    for t, p, w, l in ds.db().execute(
            "SELECT DISTINCT m.tournament, m.player, m.wins, m.losses FROM team_members m "
            "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (ds.REGULATION,)):
        if (t, p) in skill and (w or 0) + (l or 0):
            rec[(t, p)] = (w, w + l, skill[(t, p)] * (w + l))
    return rec


def above(keys, rec):
    ks = [k for k in keys if k in rec]
    g = sum(rec[k][1] for k in ks)
    return (100 * sum(rec[k][0] - rec[k][2] for k in ks) / g, g, len(ks)) if g else (0.0, 0, 0)


# ---------------------------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--min-sets", type=int, default=25, help="tournament entries a set needs to be analysed")
    p.add_argument("--min-games", type=int, default=150, help="games each arm needs for the validation")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args(argv)
    bench = Bench()
    sets, team_provides = load_sets(args.min_sets)
    rec = results()
    print(f"{len(sets)} sets with {args.min_sets}+ entries; benchmark: {len(bench.opp)} most used Pokemon", flush=True)
    rows = []
    for (name, item, ability, nature, moves), keys in sets.items():
        spec = ds.build_spec(name, item, ability, nature, moves)
        if spec is None:
            continue
        base = measure(spec, bench)
        for cond in CONDITIONS:
            eff = condition_effect(spec, cond, bench, base)
            if eff is None:
                continue
            # real data: this set on teams where a teammate provides the condition vs where none does
            with_c = [k for k in keys if any(x != name for x in team_provides[k].get(cond, ()))]
            without = [k for k in keys if not any(x != name for x in team_provides[k].get(cond, ()))]
            (yw, gw, nw), (yo, go, no) = above(with_c, rec), above(without, rec)
            providers = Counter(x for k in with_c for x in team_provides[k][cond] if x != name).most_common(3)
            rows.append({"set": f"{name} @ {item} / {ability} / {nature}: {', '.join(moves)}", "species": name,
                         "condition": cond, "score": score(eff), **{f"d_{k}": v for k, v in eff.items()},
                         "teams_with": nw, "games_with": gw, "above_with": yw, "teams_without": no,
                         "games_without": go, "above_without": yo,
                         "real_diff": yw - yo if gw and go else None,
                         "real_se": 100 * (0.25 / gw + 0.25 / go) ** 0.5 if gw and go else None,
                         "providers": "; ".join(f"{x} ({c})" for x, c in providers)})
    OUT.mkdir(exist_ok=True)
    with open(OUT / "interactions.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    report(rows, args)


def report(rows, args):
    import numpy as np
    tested = [r for r in rows if r["games_with"] >= args.min_games and r["games_without"] >= args.min_games]
    if len(tested) >= 5:
        x = np.array([r["score"] for r in tested])
        y = np.array([r["real_diff"] for r in tested])
        w = 1 / np.array([r["real_se"] for r in tested]) ** 2
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        rw = np.average((x - mx) * (y - my), weights=w) / np.sqrt(
            np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
        print(f"\nValidation: {len(tested)} set/condition pairs with {args.min_games}+ games with and without a "
              f"provider.\n  correlation of the calc benefit with the real difference (with minus without a "
              f"provider): r = {np.corrcoef(x, y)[0, 1]:+.2f}, weighted by precision {rw:+.2f}")
        for lo, hi, label in ((-9, 0, "calc says it hurts"), (0, .05, "small benefit"), (.05, 9, "big benefit")):
            part = [r for r in tested if lo <= r["score"] < hi]
            if part:
                ww = np.array([1 / r["real_se"] ** 2 for r in part])
                diff = np.average([r["real_diff"] for r in part], weights=ww)
                print(f"  {label:20s} {len(part):3d} pairs: real difference {diff:+.1f} ± {1 / np.sqrt(ww.sum()):.1f}")
    fresh = sorted((r for r in rows if r["teams_with"] < 10 and r["score"] > 0), key=lambda r: -r["score"])
    print(f"\nStrongest interactions that are rarely played (fewer than 10 teams pair the set with a provider):")
    for r in fresh[:args.top]:
        print(f"  {r['score']:+.3f}  {r['condition']:16s} ohko {100*r['d_ohko']:+5.1f}  2hko {100*r['d_ko2']:+5.1f}  "
              f"speed {100*r['d_speed']:+5.1f}  survive {100*r['d_survive']:+5.1f}  ({r['teams_with']} teams)  "
              f"{r['set']}")
    played = sorted((r for r in rows if r["games_with"] >= args.min_games and r["real_diff"] is not None),
                    key=lambda r: -r["real_diff"])
    print("\nInteractions that worked best in real tournaments (with vs without a provider):")
    for r in played[:12]:
        print(f"  {r['real_diff']:+5.1f} ± {r['real_se']:.1f}  calc {r['score']:+.3f}  {r['condition']:16s} "
              f"{r['teams_with']:4d} vs {r['teams_without']:4d} teams  {r['set']}  [{r['providers']}]")


if __name__ == "__main__":
    main()
