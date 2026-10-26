"""Do the idea generators predict anything on independent data? Scores synergy.py suggest (role synergy per pair)
and the interaction graph (calc benefit of a condition, paired with species that provide it) against ladder pair
effects beyond species (potential.py on replays). Needs output/synergy_pairs.csv and output/interactions.csv.

    python analysis/idea_check.py
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict

import numpy as np

import interactions as ix
import potential as pt


def wcorr(xs, ys, ws):
    """Weighted correlation, slope and slope standard error (binomial noise in y)."""
    x, y, w = (np.array(v, dtype=float) for v in (xs, ys, ws))
    xc, yc = x - np.average(x, weights=w), y - np.average(y, weights=w)
    sxx = np.sum(w * xc ** 2)
    return np.sum(w * xc * yc) / np.sqrt(sxx * np.sum(w * yc ** 2)), np.sum(w * xc * yc) / sxx, \
        np.sqrt(np.sum(w * xc ** 2 * 0.25)) / sxx


def pooled(items):
    """Games-weighted ladder mean of (mean, games) items, in points per game, with its standard error."""
    g = sum(x[1] for x in items)
    return 100 * sum(x[0] * x[1] for x in items) / g, 100 * (0.25 / g) ** 0.5, int(g), len(items)


def report(label, rows):
    """rows: (prediction, ladder mean, ladder games); correlation plus top vs bottom fifth by prediction."""
    r, b, se = wcorr(*zip(*[(x[0], x[1], x[2]) for x in rows]))
    print(f"  {label}: r {r:+.3f}  ({len(rows)} pairs)")
    rows = sorted(rows, key=lambda x: -x[0])
    q = len(rows) // 5
    for name, part in (("top fifth", rows[:q]), ("bottom fifth", rows[-q:])):
        m, s, g, n = pooled([(x[1], x[2]) for x in part])
        print(f"      {name:12s} ladder {m:+.2f} ± {s:.2f} points per game ({g} games, {n} pairs)")


def providers(min_teams=30):
    """Condition -> species whose tournament sets provide it at least half the time."""
    db = sqlite3.connect(f"file:{pt.META}?mode=ro", uri=True)
    prov, n = defaultdict(Counter), Counter()
    for name, ability, moves in db.execute(
            "SELECT m.pokemon, m.ability, m.moves FROM team_members m JOIN tournaments t ON t.id = m.tournament "
            "WHERE t.regulation = ?", (pt.REGULATION,)):
        n[name] += 1
        for c in ix.provides({ability or "", *(json.loads(moves) if moves else [])}):
            prov[name][c] += 1
    out = defaultdict(set)
    for s, cs in prov.items():
        for c, k in cs.items():
            if n[s] >= min_teams and k >= 0.5 * n[s]:
                out[c].add(s)
    return out


def main():
    ft, fl = pt.fit(pt.tournaments()), pt.fit(pt.ladder(), lam_p=0.0)  # raw ladder pair means beyond species
    names = pt.ladder_names(ft)

    def ladder_pair(a, b, min_games=30):
        la, lb = names.get(a), names.get(b)
        k = tuple(sorted((la, lb))) if la and lb and la != lb else None
        g = fl["pair_games"].get(k, 0)
        return (fl["pair"][k], g) if g >= min_games else None

    print("Idea generators scored on ladder pair effects beyond species (independent of the tournament data):")
    syn, games = defaultdict(list), Counter()
    for r in csv.DictReader(open(pt.OUT / "synergy_pairs.csv")):
        a, b = (re.sub(r" \[.*\]$", "", x) for x in r["pair"].split(" / "))
        if a != b:
            syn[tuple(sorted((a, b)))].append(float(r["synergy"]))
            games[tuple(sorted((a, b)))] += int(r["games"])
    rows = [(np.mean(v), *lp, games[k]) for k, v in syn.items() if (lp := ladder_pair(*k))]
    report("synergy.py predicted synergy", rows)
    report("  same, pairs rarely played in tournaments (< 150 games)", [x for x in rows if x[3] < 150])
    base_rows = [(ft["pair"][k], *lp) for k in syn if k in ft["pair"] and (lp := ladder_pair(*k))]
    report("baseline: plain tournament pair effect", base_rows)

    best = {}
    for r in csv.DictReader(open(pt.OUT / "interactions.csv")):
        k = (r["species"], r["condition"])
        best[k] = max(best.get(k, -9.0), float(r["score"]))
    prov, rows = providers(), []
    for (user, cond), score in best.items():
        for p in prov.get(cond, ()):
            if p != user and (lp := ladder_pair(user, p)):
                rows.append((score, *lp, cond))
    report("interaction graph calc benefit (user + provider)", rows)
    by_cond = defaultdict(list)
    for x in rows:
        by_cond[x[3]].append(x)
    print("  by condition:")
    for c, xs in sorted(by_cond.items(), key=lambda kv: -len(kv[1])):
        if len(xs) >= 15:
            r, b, se = wcorr(*zip(*[(x[0], x[1], x[2]) for x in xs]))
            print(f"      {c:18s} r {r:+.2f}  slope {b:+.3f} ± {se:.3f}  ({len(xs)} pairs)")


if __name__ == "__main__":
    main()
