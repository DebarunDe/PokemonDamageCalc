"""Lead model: turn 1 of a doubles game as a simultaneous-choice game, solved for both sides' best mixed strategies.

Each lead picks an action at the same time as the opponent's: a damaging move and its target, Protect, or Fake Out.
Every combination is resolved with real damage calcs and turn-1 mechanics: priority and Speed order (Swift Swim and
friends in their weather, Unburden after a terrain seed), weather and terrain from lead and Mega abilities (the slower
setter's wins, Megas after leads), Intimidate (Defiant / Competitive answer it), Fake Out flinch, Protect, Sucker
Punch needing an attacking target, Psychic Terrain blocking priority into grounded targets, single-target moves
retargeting when their target faints. Payoff = damage dealt minus damage taken, + 0.5 per KO. The game's value under
optimal mixed play is the lead matchup's predicted edge. Status moves other than Protect / Fake Out are not modelled.

Validation against ladder replays (python analysis/leads.py validate):
    1. turn-1 faints: do the model's expected KOs match the faints logged before turn 2?
    2. results: does the lead value predict who wins, beyond the Elo expectation and the Pokemon's individual strength?

    python analysis/leads.py validate --games 3000
    python analysis/leads.py matchup Raichu-Mega-X,Sneasler Archaludon,Pelipper
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sqlite3
from collections import Counter
from functools import lru_cache

import numpy as np

import doubles_sim as ds
from champcalc import CalcError, Field, PokemonSet

WEATHER = {"Drizzle": "Rain", "Drought": "Sun", "Sand Stream": "Sand", "Snow Warning": "Snow"}
TERRAIN = {"Electric Surge": "Electric", "Grassy Surge": "Grassy", "Psychic Surge": "Psychic", "Misty Surge": "Misty"}
SPEED_WEATHER = {"Swift Swim": "Rain", "Chlorophyll": "Sun", "Sand Rush": "Sand", "Slush Rush": "Snow"}
SEEDS = {"Electric Seed": ("Electric", "def"), "Grassy Seed": ("Grassy", "def"), "Psychic Seed": ("Psychic", "spd"),
         "Misty Seed": ("Misty", "spd")}
PROTECT = {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Burning Bulwark", "Silk Trap"}
KO_BONUS = 0.5
TOP_MOVES = 3  # damaging actions kept per Pokemon (by damage), besides Protect and Fake Out


@lru_cache(maxsize=None)
def move_info(name):
    try:
        return ds.calc.move(name)
    except CalcError:
        return None


def grounded(spec):
    return "Flying" not in spec.types and spec.ability != "Levitate"


class Side:
    """Two leads with the boosts, Speed and field effects they have on turn 1."""

    def __init__(self, specs):
        self.specs = specs
        self.boosts = [{} for _ in specs]
        self.speed = [s.speed for s in specs]


def field_of(a: Side, b: Side):
    """Weather and terrain after leads' abilities (slower setter wins) and then Mega abilities (slower Mega wins)."""
    weather = terrain = None
    mons = a.specs + b.specs
    for stage in ("pre", "mega"):
        for s in sorted(mons, key=lambda s: -s.speed):  # fastest activates first, so the slowest one sticks
            ab = s.pre_ability if stage == "pre" else (s.ability if "-Mega" in s.name else None)
            if ab in WEATHER:
                weather = WEATHER[ab]
            if ab in TERRAIN:
                terrain = TERRAIN[ab]
    return weather, terrain


def setup(a_specs, b_specs):
    a, b = Side(a_specs), Side(b_specs)
    weather, terrain = field_of(a, b)
    for side, other in ((a, b), (b, a)):
        for i, s in enumerate(side.specs):
            if s.pre_ability == "Intimidate":
                for j, t in enumerate(other.specs):
                    if t.ability == "Defiant":
                        other.boosts[j]["atk"] = other.boosts[j].get("atk", 0) + 2
                    elif t.ability == "Competitive":
                        other.boosts[j]["spa"] = other.boosts[j].get("spa", 0) + 2
                    elif t.ability not in ("Clear Body", "Inner Focus", "Oblivious", "Own Tempo", "Scrappy"):
                        other.boosts[j]["atk"] = other.boosts[j].get("atk", 0) - 1
            if s.item in SEEDS and SEEDS[s.item][0] == terrain:
                stat = SEEDS[s.item][1]
                side.boosts[i][stat] = side.boosts[i].get(stat, 0) + 1
                if s.ability == "Unburden":
                    side.speed[i] *= 2
            if SPEED_WEATHER.get(s.ability) == weather:
                side.speed[i] *= 2
    return a, b, weather, terrain


def pset(spec, boosts):
    return PokemonSet(spec.name, ability=spec.ability, item=spec.item, nature=spec.nature, sp=spec.sp,
                      boosts={k: max(-6, min(6, v)) for k, v in boosts.items() if v})


@lru_cache(maxsize=200000)
def _damage(att_key, def_key, weather, terrain, moves):
    att, ab = att_key
    dfn, db = def_key
    try:
        _, _, res = ds.calc.calculate_many(pset(att, dict(ab)), pset(dfn, dict(db)), list(moves),
                                           Field(weather=weather, terrain=terrain, doubles=True))
    except CalcError:
        return tuple(0.0 for _ in moves)
    out = []
    for m, r in zip(moves, res):
        acc = move_info(m)["accuracy"]
        acc = 1.0 if acc is True else acc / 100
        if m == "Hurricane" and weather == "Rain" or m == "Blizzard" and weather == "Snow":
            acc = 1.0
        out.append(acc * (r.min_percent + r.max_percent) / 2)
    return tuple(out)


def damage(side, i, other, j, moves, weather, terrain):
    key_a = (side.specs[i], tuple(sorted(side.boosts[i].items())))
    key_d = (other.specs[j], tuple(sorted(other.boosts[j].items())))
    return _damage(key_a, key_d, weather, terrain, tuple(moves))


def actions(side, i, other, weather, terrain):
    """A lead's candidate actions: ('protect',), ('fake', target), ('move', name, target or None for spread)."""
    s = side.specs[i]
    dmg_moves = [m for m in s.moves if move_info(m) and move_info(m)["category"] != "Status" and m != "Fake Out"]
    cands = []
    if dmg_moves:
        d0 = damage(side, i, other, 0, dmg_moves, weather, terrain)
        d1 = damage(side, i, other, 1, dmg_moves, weather, terrain)
        for k, m in enumerate(dmg_moves):
            if move_info(m)["spread"]:
                cands.append((d0[k] + d1[k], ("move", m, None)))
            else:
                cands.append((d0[k], ("move", m, 0)))
                cands.append((d1[k], ("move", m, 1)))
    cands.sort(key=lambda x: -x[0])
    out = [c for _, c in cands[:TOP_MOVES]]
    if "Fake Out" in s.moves:
        out += [("fake", 0), ("fake", 1)]
    if PROTECT & set(s.moves):
        out.append(("protect",))
    return out or [("pass",)]


def resolve(a, b, act_a, act_b, weather, terrain):
    """One turn. Returns (payoff to side a, expected KOs by a, expected KOs by b)."""
    sides = (a, b)
    hp = [[100.0, 100.0], [100.0, 100.0]]
    protected, flinched, moved = set(), set(), set()
    order = []
    for si, acts in enumerate((act_a, act_b)):
        for i, act in enumerate(acts):
            if act[0] == "protect":
                pri = 4
            elif act[0] == "fake":
                pri = 3
            elif act[0] == "move":
                pri = move_info(act[1])["priority"]
                if act[1] == "Grassy Glide" and terrain == "Grassy":
                    pri = 1
            else:
                pri = -9
            order.append((pri, sides[si].speed[i], si, i, act))
    order.sort(key=lambda x: (-x[0], -x[1]))
    acts_of = {(0, i): act_a[i] for i in range(2)} | {(1, i): act_b[i] for i in range(2)}
    for pri, _, si, i, act in order:
        moved.add((si, i))
        if hp[si][i] <= 0 or (si, i) in flinched or act[0] == "pass":
            continue
        if act[0] == "protect":
            protected.add((si, i))
            continue
        oi = 1 - si
        me, other = sides[si], sides[oi]
        if act[0] == "fake" or (act[0] == "move" and act[2] is not None):
            t = act[1] if act[0] == "fake" else act[2]
            if hp[oi][t] <= 0:
                t = 1 - t
            targets = [t] if hp[oi][t] > 0 else []
        else:
            targets = [t for t in (0, 1) if hp[oi][t] > 0]
        for t in targets:
            if (oi, t) in protected:
                continue
            if pri > 0 and terrain == "Psychic" and grounded(other.specs[t]):
                continue
            if act[0] == "fake":
                dmg = damage(me, i, other, t, ["Fake Out"], weather, terrain)[0]
                if (oi, t) not in moved:
                    flinched.add((oi, t))
            else:
                if act[1] == "Sucker Punch" and ((oi, t) in moved or acts_of[(oi, t)][0] != "move"):
                    continue
                dmg = damage(me, i, other, t, [act[1]], weather, terrain)[0]
            hp[oi][t] -= dmg
    lost = [[max(0.0, 100 - max(h, 0.0)) / 100 + (KO_BONUS if h <= 0 else 0.0) for h in hp[s]] for s in (0, 1)]
    kos = [sum(1 for h in hp[s] if h <= 0) for s in (0, 1)]
    return sum(lost[1]) - sum(lost[0]), kos[1], kos[0]


def solve(M):
    """Value and optimal mixed strategies of the zero-sum game with payoff matrix M (rows maximise)."""
    from scipy.optimize import linprog
    m, n = M.shape
    shift = M.min() - 1.0
    A = M - shift  # all positive: value > 0
    # rows: minimise sum(u) s.t. A^T u >= 1, u >= 0; value = 1 / sum(u), x = u * value
    r = linprog(np.ones(m), A_ub=-A.T, b_ub=-np.ones(n), bounds=[(0, None)] * m, method="highs")
    c = linprog(-np.ones(n), A_ub=A, b_ub=np.ones(m), bounds=[(0, None)] * n, method="highs")
    v = 1.0 / r.x.sum()
    return v + shift, r.x * v, c.x / c.x.sum()


def matchup(a_specs, b_specs):
    """(value to side a, expected KOs by a, expected KOs by b, details)."""
    a, b, weather, terrain = setup(a_specs, b_specs)
    acts_a = list(itertools.product(actions(a, 0, b, weather, terrain), actions(a, 1, b, weather, terrain)))
    acts_b = list(itertools.product(actions(b, 0, a, weather, terrain), actions(b, 1, a, weather, terrain)))
    M = np.zeros((len(acts_a), len(acts_b)))
    KA = np.zeros_like(M)
    KB = np.zeros_like(M)
    for r, x in enumerate(acts_a):
        for c, y in enumerate(acts_b):
            M[r, c], KA[r, c], KB[r, c] = resolve(a, b, x, y, weather, terrain)
    v, px, py = solve(M)
    return v, float(px @ KA @ py), float(px @ KB @ py), {"weather": weather, "terrain": terrain, "acts_a": acts_a,
                                                          "acts_b": acts_b, "px": px, "py": py}


# ------------------------------------------------------------------------------------------------- validation

def lead_forms():
    """Ladder (preview) name -> most common tournament form, and the Megas of each base species."""
    from replays import base
    names = Counter(n for (n,) in ds.db().execute(
        "SELECT m.pokemon FROM team_members m JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?",
        (ds.REGULATION,)))
    best = {}
    for n, c in names.most_common():
        best.setdefault(base(n), n)
    return best


def turn1_faints(log):
    """Faints per side ('p1', 'p2') between turn 1 and turn 2."""
    out = {"p1": 0, "p2": 0}
    on = False
    for line in log.split("\n"):
        if line.startswith("|turn|1"):
            on = True
        elif line.startswith("|turn|2"):
            break
        elif on and line.startswith("|faint|"):
            out[line.split("|")[2][:2]] += 1
    return out


def validate(n_games=3000, seed=0):
    from replays import DB, base, expected
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    sides = {}
    for rid, side, r, o, won, leads, megas in db.execute(
            "SELECT id, side, rating, opp_rating, won, leads, mega FROM sides"):
        sides[(rid, side)] = (r, o, won, json.loads(leads), json.loads(megas))
    games = [rid for (rid, side) in sides if side == "p1" and (rid, "p2") in sides
             and len(sides[(rid, "p1")][3]) == 2 and len(sides[(rid, "p2")][3]) == 2 and sides[(rid, "p1")][0]
             and sides[(rid, "p1")][1]]
    random.Random(seed).shuffle(games)
    forms = lead_forms()

    def specs(leads, megas):
        out = []
        for n in leads:
            mega = next((m for m in megas if base(m) == n), None)
            name = mega or (n if megas else forms.get(n, n))
            s = ds.real_spec(name) or ds.real_spec(n)
            if s is None:
                return None
            out.append(s)
        return out

    rows = []
    for rid in games[:n_games]:
        p1, p2 = sides[(rid, "p1")], sides[(rid, "p2")]
        a, b = specs(p1[3], p1[4]), specs(p2[3], p2[4])
        if not a or not b:
            continue
        try:
            v, ka, kb, _ = matchup(a, b)
        except Exception:
            continue
        log = db.execute("SELECT log FROM replays WHERE id = ?", (rid,)).fetchone()[0]
        f = turn1_faints(log)
        rows.append((v, ka, kb, f["p2"], f["p1"], p1[2] - expected(p1[0], p1[1]), p1[3], p2[3]))
        if len(rows) % 250 == 0:
            print(f"  {len(rows)} games", flush=True)
    R = np.array([r[:6] for r in rows], dtype=float)
    v, ka, kb, fa, fb, res = R.T
    print(f"\n{len(rows)} ladder games (random sample)")
    print("1. Turn-1 KOs, predicted (equilibrium play) vs logged:")
    print(f"   KOs by side 1: predicted mean {ka.mean():.2f}, actual {fa.mean():.2f}; "
          f"correlation {np.corrcoef(ka, fa)[0, 1]:+.3f}")
    print(f"   KO difference (1 - 2): correlation {np.corrcoef(ka - kb, fa - fb)[0, 1]:+.3f}")
    for lo, hi in ((-9, -0.5), (-0.5, -0.1), (-0.1, 0.1), (0.1, 0.5), (0.5, 9)):
        m = (ka - kb >= lo) & (ka - kb < hi)
        if m.sum() >= 30:
            print(f"     predicted diff in [{lo:+}, {hi:+}): {m.sum():5d} games, "
                  f"logged diff {(fa - fb)[m].mean():+.2f}")
    print("2. Game results (won minus Elo expectation):")
    print(f"   correlation with lead value {np.corrcoef(v, res)[0, 1]:+.3f}; with actual turn-1 KO difference "
          f"{np.corrcoef(fa - fb, res)[0, 1]:+.3f} (how much turn 1 matters at all)")
    for lo, hi in ((-9, -0.5), (-0.5, -0.15), (-0.15, 0.15), (0.15, 0.5), (0.5, 9)):
        m = (v >= lo) & (v < hi)
        if m.sum() >= 30:
            print(f"     lead value in [{lo:+}, {hi:+}): {m.sum():5d} games, result {100 * res[m].mean():+5.1f} "
                  f"± {100 * (0.25 / m.sum()) ** 0.5:.1f}")
    # beyond the Pokemon's individual strength: regress result on lead value and on the leads' strength difference
    try:
        import potential as pt
        f = pt.fit(pt.ladder(), lam_p=1e12)["species"]
        strength = np.array([sum(f.get(x, 0.0) for x in r[6]) - sum(f.get(x, 0.0) for x in r[7]) for r in rows])
        X = np.column_stack([np.ones(len(v)), v, strength])
        beta, *_ = np.linalg.lstsq(X, res, rcond=None)
        resid = res - X @ beta
        cov = np.linalg.inv(X.T @ X) * resid.var()
        print(f"   controlling for the leads' individual strength: lead value coefficient {100 * beta[1]:+.2f} "
              f"± {100 * cov[1, 1] ** 0.5:.2f} points per game per unit (strength coefficient {beta[2]:+.2f})")
    except Exception as e:  # potential needs pandas; skip the control without it
        print(f"   (strength control skipped: {e})")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--games", type=int, default=3000)
    m = sub.add_parser("matchup")
    m.add_argument("a", help="comma-separated lead names (tournament names, e.g. Raichu-Mega-X,Sneasler)")
    m.add_argument("b")
    args = p.parse_args(argv)
    if args.cmd == "validate":
        validate(args.games)
    else:
        a = [ds.real_spec(x) for x in args.a.split(",")]
        b = [ds.real_spec(x) for x in args.b.split(",")]
        v, ka, kb, d = matchup(a, b)
        print(f"{args.a} vs {args.b}: value {v:+.2f} (damage + KO units for the first side), expected turn-1 KOs "
              f"{ka:.2f} vs {kb:.2f}; weather {d['weather']}, terrain {d['terrain']}")
        for label, acts, p in (("first side", d["acts_a"], d["px"]), ("second side", d["acts_b"], d["py"])):
            print(f"  {label} plays:")
            for k in np.argsort(-p)[:5]:
                if p[k] > 0.01:
                    print(f"    {100 * p[k]:4.0f}%  {acts[k]}")


if __name__ == "__main__":
    main()
