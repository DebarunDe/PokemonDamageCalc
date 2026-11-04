"""Best movesets from the whole learnset, searched exhaustively, against every legal Pokemon weighted equally.

For each legal Pokemon, each ability, each attacking category (both when Attack and Sp. Atk are within 15) and each
item, every combination of 3 and of 4 damaging moves from its learnset is scored against the whole roster with the
theorycraft bench (no usage data):

    power   share of the roster its best move OHKOs (1) or 2HKOs (0.5), accuracy included
    first   the same, counting only Pokemon it moves before
    score   power + first

Items: none, Life Orb, Expert Belt, Muscle Band / Wise Glasses, the type item of each STAB type, Choice Scarf
(Champions has no Choice Band / Specs). Expert Belt is calculated; the others are flat damage multipliers (or Speed,
for Scarf) applied to the no-item damage. Megas hold their stone. Life Orb recoil and Choice lock are not costed.

Moves: every damaging move except recharge, two-turn, self-KO, lock-in, first-turn-only, fixed-damage and other
conditional moves (Sucker Punch counts as landing; Eruption / Water Spout at full HP). Moves no better than another
move against every Pokemon are pruned before the search, so the search is exact over the rest.

    python analysis/movesets.py [--workers 4] [--only Medicham-Mega Lucario-Mega]
Output: analysis/output/theory_movesets.csv (best set per Pokemon, with runner-up sets).
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations

import numpy as np

import theorycraft as tc
from champcalc import CalcError, Field

EXCLUDE = {
    # recharge / two-turn / charge
    "Hyper Beam", "Giga Impact", "Blast Burn", "Hydro Cannon", "Frenzy Plant", "Prismatic Laser", "Roar of Time",
    "Rock Wrecker", "Eternabeam", "Meteor Beam", "Sky Attack", "Skull Bash", "Solar Beam", "Solar Blade",
    "Electro Shot", "Phantom Force", "Shadow Force", "Bounce", "Dig", "Dive", "Fly", "Sky Drop", "Geomancy",
    "Freeze Shock", "Ice Burn", "Razor Wind",
    # self-KO / half HP / lock-in / first turn
    "Explosion", "Self-Destruct", "Misty Explosion", "Final Gambit", "Mind Blown", "Steel Beam", "Chloroblast",
    "Outrage", "Petal Dance", "Thrash", "Raging Fury", "Uproar", "Rollout", "Ice Ball", "Glaive Rush",
    "Fake Out", "First Impression", "Meteor Assault",
    # fixed damage / conditional / party-dependent
    "Fling", "Natural Gift", "Last Resort", "Dream Eater", "Belch", "Focus Punch", "Future Sight", "Doom Desire",
    "Synchronoise", "Steel Roller", "Shell Trap", "Beak Blast", "Counter", "Mirror Coat", "Metal Burst", "Bide",
    "Endeavor", "Super Fang", "Ruination", "Nature's Madness", "Night Shade", "Seismic Toss", "Psywave",
    "Snore", "Spit Up", "Beat Up", "Upper Hand", "Burn Up", "Double Shock",
}
TYPE_ITEM = {"Fire": "Charcoal", "Water": "Mystic Water", "Grass": "Miracle Seed", "Electric": "Magnet",
             "Ice": "Never-Melt Ice", "Fighting": "Black Belt", "Poison": "Poison Barb", "Ground": "Soft Sand",
             "Flying": "Sharp Beak", "Psychic": "Twisted Spoon", "Bug": "Silver Powder", "Rock": "Hard Stone",
             "Ghost": "Spell Tag", "Dragon": "Dragon Fang", "Dark": "Black Glasses", "Steel": "Metal Coat",
             "Normal": "Silk Scarf", "Fairy": "Fairy Feather"}
MAX_COMBOS = 400_000
# expected share of full damage when each hit rolls accuracy a and the move stops at the first miss
MULTI_ACC = {"Population Bomb": lambda a: sum(a ** i for i in range(1, 11)) / 10,
             "Triple Axel": lambda a: (a * 20 + a ** 2 * 40 + a ** 3 * 60) / 120,
             "Triple Kick": lambda a: (a * 10 + a ** 2 * 20 + a ** 3 * 30) / 60}


def moves_for(s, cat):
    out = []
    for m in sorted(tc.learn(s)):
        i = tc.minfo(m)
        if i and i["category"] == cat and m not in EXCLUDE and i["priority"] >= 0 and (
                i["base_power"] or m in ("Low Kick", "Grass Knot", "Heavy Slam", "Heat Crash", "Crush Grip",
                                         "Hard Press", "Gyro Ball", "Electro Ball", "Eruption", "Water Spout",
                                         "Dragon Energy", "Wring Out", "Punishment", "Last Respects", "Rage Fist")):
            out.append(m)
    return out


def pct_matrix(t, moves, item):
    """[bench Pokemon x move] average % damage, holding `item` (None keeps the Theory's own item)."""
    defs, _ = tc.bench()
    f = Field(doubles=True)
    t_item = t.item
    if item is not None:
        t.item = item
    att = t.pset()
    t.item = t_item
    good = []
    for m in moves:
        try:
            tc.calc.calculate_batch([m], [(att, defs[0], f, [0])])
            good.append(m)
        except CalcError:
            pass
    if not good:
        return [], None
    res = tc.calc.calculate_batch(good, [(att, d, f, list(range(len(good)))) for d in defs])
    P = np.array([[100 * (lo + hi) / 2 / hp for lo, hi in dmg] for hp, dmg in res])
    return good, P


def credit(P, acc):
    return acc[None, :] * np.where(P >= 100, 1.0, np.where(P >= 50, 0.5, 0.0))


def prune(C):
    """Indices of columns not dominated by another column (ties keep the first)."""
    keep = []
    for j in range(C.shape[1]):
        dominated = False
        for k in range(C.shape[1]):
            if k != j and np.all(C[:, k] >= C[:, j]) and (np.any(C[:, k] > C[:, j]) or k < j):
                dominated = True
                break
        if not dominated:
            keep.append(j)
    return keep


def search(C, w, k):
    """Best k columns of C (max over chosen columns, weighted by w): exact when the combinations are few enough,
    else greedy plus one-swap improvement. Returns [(score, column tuple)] best first (top 5)."""
    cols = prune(C)
    k = min(k, len(cols))
    if k == 0:
        return []
    from math import comb
    if comb(len(cols), k) <= MAX_COMBOS:
        best = []
        Cc = C[:, cols]
        for combo in combinations(range(len(cols)), k):
            v = float(w @ Cc[:, combo].max(axis=1))
            best.append((v, tuple(cols[i] for i in combo)))
        best.sort(reverse=True)
        return best[:5]
    chosen = []
    for _ in range(k):
        chosen.append(max((c for c in cols if c not in chosen),
                          key=lambda c: float(w @ C[:, chosen + [c]].max(axis=1))))
    improved = True
    while improved:
        improved = False
        cur = float(w @ C[:, chosen].max(axis=1))
        for i in range(k):
            for c in cols:
                if c in chosen:
                    continue
                trial = chosen[:i] + [c] + chosen[i + 1:]
                v = float(w @ C[:, trial].max(axis=1))
                if v > cur + 1e-12:
                    chosen, cur, improved = trial, v, True
    return [(cur, tuple(chosen))]


def best_for(s):
    """All abilities / categories / items for one Pokemon: rows for its best 4-move and 3-move sets."""
    _, speeds = tc.bench()
    speeds = np.array(speeds)
    n = len(speeds)
    b = tc.info(s)["base_stats"]
    cats = ["Physical", "Special"] if abs(b["atk"] - b["spa"]) <= 15 else [tc.category(s)]
    mega = "-Mega" in s
    results = []
    for ab in tc.abilities(s):
        for cat in cats:
            moves = moves_for(s, cat)
            if not moves:
                continue
            t = tc.Theory(s, ab, cat=cat)
            good, P0 = pct_matrix(t, moves, None)
            if not good:
                continue
            acc = np.array([1.0 if ab == "No Guard" else tc.acc_of(m) for m in good])
            if ab == "Hustle":  # Attack x1.5 (in the calc), physical moves 20% less accurate
                acc = acc * np.array([0.8 if tc.minfo(m)["category"] == "Physical" else 1.0 for m in good])
            # multi-hit moves that roll accuracy on every hit and stop at the first miss: the calc assumes every hit
            # lands, so scale damage by the expected share of it and drop the separate accuracy factor
            # (Skill Link / No Guard check once)
            scale = np.ones(len(good))
            for j, m in enumerate(good):
                if m in MULTI_ACC and ab not in ("Skill Link", "No Guard"):
                    scale[j], acc[j] = MULTI_ACC[m](acc[j]), 1.0
            P0 = P0 * scale[None, :]
            mtype = [tc.minfo(m)["type"] for m in good]
            items = {"(none)" if not mega else t.item: (P0, 1.0)}
            if not mega:
                items["Life Orb"] = (P0 * 1.3, 1.0)
                items["Muscle Band" if cat == "Physical" else "Wise Glasses"] = (P0 * 1.1, 1.0)
                for typ in set(tc.info(s)["types"]):
                    if typ in TYPE_ITEM:
                        mult = np.array([1.2 if mt == typ else 1.0 for mt in mtype])
                        items[TYPE_ITEM[typ]] = (P0 * mult[None, :], 1.0)
                items["Choice Scarf"] = (P0, 1.5)
                _, PE = pct_matrix(t, good, "Expert Belt")
                if PE is not None:
                    items["Expert Belt"] = (PE * scale[None, :], 1.0)
            itemless = np.array([0.5 if m == "Acrobatics" else 1.0 for m in good])[None, :]
            for item, (P, spe_mult) in items.items():
                if not mega and item not in ("(none)", "Expert Belt"):
                    P = P * itemless  # Acrobatics is double power only without an item
                C = credit(P, acc)
                mask = (t.speed * spe_mult > speeds).astype(float)
                w = (1 + mask) / n
                for k in (4, 3):
                    for score, cols in search(C, w, k):
                        vals = C[:, list(cols)].max(axis=1)
                        results.append({
                            "pokemon": s, "ability": ab, "category": cat, "item": item, "k": k,
                            "moves": ", ".join(good[c] for c in cols), "score": round(score, 4),
                            "power": round(float(vals.mean()), 4), "first": round(float((vals * mask).mean()), 4),
                            "ohko": round(float((vals > 0.5).mean()), 4),  # OHKO credit at any accuracy above 50%
                            "no_2hko": round(float((vals == 0).mean()), 4), "speed": round(t.speed * spe_mult),
                            "walled_by": ", ".join(d for d, v in zip(tc.ROSTER, vals) if v == 0)})
    return results


def _job(s):
    try:
        return best_for(s)
    except Exception as e:  # keep the run going; report the failure
        return [{"pokemon": s, "error": repr(e)}]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", nargs="*")
    args = p.parse_args(argv)
    import multiprocessing as mp
    tc.bench()  # writes the bench Speed cache before the workers start
    names = args.only or tc.ROSTER
    rows, errors = [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, names)):
            for r in res:
                (errors if "error" in r else rows).append(r)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1} / {len(names)}", flush=True)
    for e in errors:
        print("  failed:", e["pokemon"], e["error"][:150])
    keys = ["pokemon", "ability", "category", "item", "k", "moves", "score", "power", "first", "ohko", "no_2hko",
            "speed", "walled_by"]
    rows.sort(key=lambda r: (r["pokemon"], r["k"], -r["score"]))
    tc.OUT.mkdir(exist_ok=True)
    with open(tc.OUT / "theory_movesets.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    best = {}
    for r in rows:
        if r["k"] == 4 and (r["pokemon"] not in best or r["score"] > best[r["pokemon"]]["score"]):
            best[r["pokemon"]] = r
    print(f"\n{len(rows)} sets for {len(best)} Pokemon -> {tc.OUT / 'theory_movesets.csv'}")
    for label, keep in (("Megas", lambda s: "-Mega" in s), ("non-Megas", lambda s: "-Mega" not in s)):
        print(f"\nBest 4-attack sets, {label}:")
        for r in sorted((r for r in best.values() if keep(r["pokemon"])), key=lambda r: -r["score"])[:25]:
            print(f"  {r['score']:.2f}  {r['pokemon']:20s} {r['ability']:15s} @ {r['item']:14s} power {r['power']:.2f} "
                  f"first {r['first']:.2f} OHKO {r['ohko']:.2f} walled {r['no_2hko']:.2f}  {r['moves']}")


if __name__ == "__main__":
    main()
