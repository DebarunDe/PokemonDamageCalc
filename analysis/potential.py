"""Search for under-used Pokemon, pairs and cores with potential, from two independent sources of results.

Tournament team sheets (meta.sqlite): each team's wins above its player's skill (skill from their other events).
Ladder replays (replays.sqlite): each side's result above the Elo expectation from the two ladder ratings.

In each source, a team's result is explained by its species (a ridge regression), then a pair's effect is what its
teams do beyond their species (a shrunk mean): what the pair adds beyond its two members. Nothing is trusted from one
source alone:

    1. reliability, forward in time: fitted on earlier events, how well do species, and species + pairs, predict
       later events, and the ladder (whose replays all come after the last event)? Random halves of the same weeks
       are not used: they share that period's meta and overstate the pair terms several times over.
    2. species: rare in tournaments, estimated positive in both sources
    3. pairs: a positive pair effect in both sources, and a strong total (both species + the pair)
    4. cores: the best thirds for the top pairs, from the same terms

    python analysis/potential.py [--min-rating 1100] [--max-share 0.05]

Results go to analysis/output/potential_*.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
META, REPLAYS, OUT = HERE / "data" / "meta.sqlite", HERE / "data" / "replays.sqlite", HERE / "output"
REGULATION = "M-C"
# games; tuned on earlier -> later events (lighter pair shrinkage, e.g. 300, overfits and predicts worse than none)
LAM_SPECIES = 1000.0
LAM_PAIR = 3000.0
CUT = "2026-10-08"  # earlier / later events for the forward check


class Rows:
    """One source's teams: species lists, result above expectation per game, games, and date (ISO)."""

    def __init__(self, name):
        self.name, self.teams, self.y, self.g, self.when = name, [], [], [], []

    def add(self, species, y, g, when):
        self.teams.append(sorted(set(species)))
        self.y.append(y)
        self.g.append(g)
        self.when.append(when)

    def subset(self, keep, name=None):
        out = Rows(name or self.name)
        for i in range(len(self.teams)):
            if keep(self.when[i]):
                out.add(self.teams[i], self.y[i], self.g[i], self.when[i])
        return out


def tournaments():
    from signals import player_skill
    db = sqlite3.connect(f"file:{META}?mode=ro", uri=True)
    skill = player_skill(db).set_index(["tournament", "player"])["skill"].to_dict()
    teams, record = defaultdict(list), {}
    for t, p, mon, w, l, day in db.execute(
            "SELECT m.tournament, m.player, m.pokemon, m.wins, m.losses, t.date FROM team_members m "
            "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)):
        teams[(t, p)].append(mon)
        record[(t, p)] = (w or 0, l or 0, day[:10])
    rows = Rows("tournaments")
    for key, mons in teams.items():
        w, l, day = record[key]
        if w + l and key in skill and len(mons) >= 4:
            rows.add(mons, (w - skill[key] * (w + l)) / (w + l), w + l, day)
    return rows


def ladder(min_rating=0):
    """Team preview, with a species named by its Mega form when that Mega was used in the game."""
    from replays import expected
    db = sqlite3.connect(f"file:{REPLAYS}?mode=ro", uri=True, timeout=60)
    rows = Rows("ladder")
    for when, rating, opp, won, preview, megas in db.execute(
            "SELECT r.uploadtime, s.rating, s.opp_rating, s.won, s.preview, s.mega FROM sides s "
            "JOIN replays r ON r.id = s.id"):
        if not rating or not opp or min(rating, opp) < min_rating:
            continue
        mega = {m.split("-Mega")[0]: m for m in json.loads(megas)}
        mons = [mega.get(p, p) for p in json.loads(preview)]
        if len(mons) >= 4:
            day = datetime.fromtimestamp(when, timezone.utc).date().isoformat()
            rows.add(mons, won - expected(rating, opp), 1, day)
    return rows


def fit(rows, lam_s=LAM_SPECIES, lam_p=LAM_PAIR):
    """Species effects first (ridge on team results, with an intercept), then each pair's effect as the shrunk mean
    of what is left on its teams. Fitting both at once lets a strong species' effect spread over its many pairs
    (cheaper under the penalty), so 'synergy' would partly be species strength. In two stages a pair's effect is
    understated only when the pair is a large share of a species' teams, which an under-used pair is not."""
    names = sorted({s for t in rows.teams for s in t})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    A, b = np.zeros((k + 1, k + 1)), np.zeros(k + 1)
    for t, y, g in zip(rows.teams, rows.y, rows.g):
        ii = [idx[s] for s in t] + [k]
        A[np.ix_(ii, ii)] += g
        b[ii] += g * y
    x = np.linalg.solve(A + np.diag(np.r_[np.full(k, lam_s), 1e-9]), b)
    games, teams = np.diag(A)[:k], Counter(s for t in rows.teams for s in t)
    # a species' effect is relative to the average team slot; the shift goes into the intercept
    w = np.array([teams[n] for n in names], dtype=float)
    shift = float(np.average(x[:k], weights=w))
    species = {n: float(x[idx[n]]) - shift for n in names}
    intercept = float(x[k]) + shift * w.sum() / len(rows.teams)
    num, den, pair_teams = defaultdict(float), defaultdict(float), Counter()
    for t, y, g in zip(rows.teams, rows.y, rows.g):
        r = y - intercept - sum(species[s] for s in t)
        for p in itertools.combinations(t, 2):
            num[p] += g * r
            den[p] += g
            pair_teams[p] += 1
    return {"species": species, "intercept": intercept, "species_games": dict(zip(names, map(float, games))),
            "species_teams": dict(teams), "pair": {p: num[p] / (den[p] + lam_p) for p in num},
            "pair_games": dict(den), "pair_teams": dict(pair_teams), "n": len(rows.teams)}


def se(games, lam):
    """Posterior spread of a shrunk mean of per-game results (binomial variance 0.25)."""
    return float(np.sqrt(0.25 / (games + lam)))


def corr(a, b, weight=None):
    keys = [k for k in a if k in b]
    if len(keys) < 5:
        return float("nan"), len(keys)
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    w = np.ones(len(keys)) if weight is None else np.array([weight(k) for k in keys], dtype=float)
    x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
    return float(np.sum(w * x * y) / np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))), len(keys)


def carry(train, test, kind, min_train, min_test):
    """How much of each estimate from `train` shows up again in `test`: the games-weighted slope of the test's own
    (unshrunk for pairs) estimate on the train estimate, its standard error (binomial noise only, so a lower
    bound), the correlation and the number compared. Slope 1 = fully persistent, 0 = noise."""
    a, b = fit(train), fit(test, lam_p=0.0)
    games_a, games_b = a[kind + "_games"], b[kind + "_games"]
    keys = [k for k in a[kind] if games_a[k] >= min_train and games_b.get(k, 0) >= min_test]
    if len(keys) < 5:
        return None
    x, y = np.array([a[kind][k] for k in keys]), np.array([b[kind][k] for k in keys])
    w = np.array([games_b[k] for k in keys])
    xc, yc = x - np.average(x, weights=w), y - np.average(y, weights=w)
    sxx = np.sum(w * xc ** 2)
    return (float(np.sum(w * xc * yc) / sxx), float(np.sqrt(np.sum(w * xc ** 2 * 0.25)) / sxx),
            float(np.sum(w * xc * yc) / np.sqrt(sxx * np.sum(w * yc ** 2))), len(keys))


def reliability(tour, lad, cut=CUT):
    """Forward checks: (kind, label, carry result)."""
    early, late = tour.subset(lambda d: d < cut), tour.subset(lambda d: d >= cut)
    out = []
    for kind, mt, ms in (("species", 300, 300), ("pair", 100, 60)):
        out.append((kind, f"tournaments before {cut} -> after", carry(early, late, kind, mt, ms)))
        if lad.teams:
            out.append((kind, "all tournaments -> ladder", carry(tour, lad, kind, mt, ms)))
    return out


def combine(estimates):
    """Precision-weighted mean of (value, se) pairs that exist."""
    est = [(v, s) for v, s in estimates if v is not None]
    w = [1 / s ** 2 for _, s in est]
    return sum(v * x for (v, _), x in zip(est, w)) / sum(w), sum(w) ** -0.5


def species_table(ft, fl, min_tour_teams, min_ladder_games):
    rows = []
    n_t = ft["n"]
    for s, v in ft["species"].items():
        tt = ft["species_teams"][s]
        if tt < min_tour_teams:
            continue
        gt = ft["species_games"][s]
        t = (v, se(gt, LAM_SPECIES))
        gl = fl["species_games"].get(s, 0)
        l = (fl["species"][s], se(gl, LAM_SPECIES)) if gl >= min_ladder_games else (None, None)
        c, cs = combine([t, l])
        rows.append({"species": s, "tour_share": tt / n_t, "tour_teams": tt, "tour": 100 * t[0], "tour_se": 100 * t[1],
                     "ladder_share": fl["species_teams"].get(s, 0) / max(1, fl["n"]), "ladder_games": int(gl),
                     "ladder": None if l[0] is None else 100 * l[0], "ladder_se": None if l[1] is None else 100 * l[1],
                     "combined": 100 * c, "combined_se": 100 * cs,
                     "agree": l[0] is not None and t[0] > 0 and l[0] > 0})
    rows.sort(key=lambda r: -(r["combined"] - r["combined_se"]))
    return rows


def pair_table(ft, fl, min_tour_teams, min_ladder_games):
    rows = []
    for p, v in ft["pair"].items():
        tt = ft["pair_teams"][p]
        if tt < min_tour_teams or ("-Mega" in p[0] and "-Mega" in p[1]):
            continue
        t = (v, se(ft["pair_games"][p], LAM_PAIR))
        gl = fl["pair_games"].get(p, 0)
        l = (fl["pair"][p], se(gl, LAM_PAIR)) if gl >= min_ladder_games else (None, None)
        c, cs = combine([t, l])
        sums = [ft["species"][p[0]] + ft["species"][p[1]]]
        if l[0] is not None:
            sums.append(fl["species"].get(p[0], 0.0) + fl["species"].get(p[1], 0.0))
        alone = sum(sums) / len(sums)
        rows.append({"pair": f"{p[0]} / {p[1]}", "a": p[0], "b": p[1], "tour_teams": tt,
                     "tour_share": tt / ft["n"], "tour": 100 * t[0], "tour_se": 100 * t[1], "ladder_games": int(gl),
                     "ladder": None if l[0] is None else 100 * l[0], "ladder_se": None if l[1] is None else 100 * l[1],
                     "synergy": 100 * c, "synergy_se": 100 * cs, "species_sum": 100 * alone,
                     "total": 100 * (alone + c), "agree": l[0] is not None and t[0] > 0 and l[0] > 0})
    rows.sort(key=lambda r: -(r["synergy"] - r["synergy_se"]))
    return rows


def cores(pairs, ft, fl, top_pairs, n_third=3):
    """For the best pairs, thirds that add the most: the third's species effect plus its pair terms with both."""
    syn = {(r["a"], r["b"]): r["synergy"] / 100 for r in pairs}
    sp = {s: (ft["species"][s] + fl["species"].get(s, ft["species"][s])) / 2 for s in ft["species"]}
    out = []
    for r in top_pairs:
        a, b = r["a"], r["b"]
        cands = []
        for c in sp:
            if c in (a, b) or ft["species_teams"][c] < 30 or sum("-Mega" in x for x in (a, b, c)) > 1:
                continue
            ac, bc = syn.get(tuple(sorted((a, c)))), syn.get(tuple(sorted((b, c))))
            if ac is None or bc is None:
                continue
            cands.append((sp[c] + ac + bc, c, ac, bc))
        cands.sort(reverse=True)
        for gain, c, ac, bc in cands[:n_third]:
            out.append({"pair": r["pair"], "third": c, "adds": 100 * gain, "third_alone": 100 * sp[c],
                        "with_a": 100 * ac, "with_b": 100 * bc,
                        "core_total": r["total"] + 100 * gain,
                        "third_share": ft["species_teams"][c] / ft["n"]})
    return out


def write(name, rows):
    if rows:
        OUT.mkdir(exist_ok=True)
        with open(OUT / f"potential_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def fmt(v, s):
    return "    n/a     " if v is None else f"{v:+5.1f} ± {s:3.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--min-rating", type=int, default=0, help="ladder games where both sides are rated this or more")
    p.add_argument("--max-share", type=float, default=0.05, help="'under-used': on at most this share of teams")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)
    tour, lad = tournaments(), ladder(args.min_rating)
    print(f"{len(tour.teams)} tournament teams ({sum(tour.g):.0f} games); {len(lad.teams)} rated ladder sides")
    print("\n1. Forward checks: how much of an earlier estimate shows up again later (slope 1 = all of it, 0 = noise):")
    for kind, label, c in reliability(tour, lad):
        if c:
            print(f"  {kind:8s} {label:36s} slope {c[0]:+.2f} ± {c[1]:.2f}  r {c[2]:+.2f}  ({c[3]} compared)")
    ft, fl = fit(tour), fit(lad)

    sp = species_table(ft, fl, min_tour_teams=30, min_ladder_games=100)
    write("species", sp)
    rare = [r for r in sp if r["tour_share"] <= args.max_share]
    print(f"\n2. Under-used species (on <= {100 * args.max_share:.0f}% of tournament teams), points per game above "
          "expectation:")
    print(f"  {'tournaments':>12} {'ladder':>12} {'combined':>12}  share t/l   both+  species")
    for r in rare[:args.top]:
        print(f"  {fmt(r['tour'], r['tour_se'])} {fmt(r['ladder'], r['ladder_se'])} "
              f"{fmt(r['combined'], r['combined_se'])}  {100 * r['tour_share']:4.1f}/{100 * r['ladder_share']:4.1f}%  "
              f"{'yes' if r['agree'] else ' - ':>5}  {r['species']}")

    pr = pair_table(ft, fl, min_tour_teams=20, min_ladder_games=40)
    write("pairs", pr)
    rare_pairs = [r for r in pr if r["tour_share"] <= args.max_share / 5 and r["total"] > 0]
    print(f"\n3. Under-used pairs (on <= {20 * args.max_share:.0f}% of tournament teams) by surest synergy "
          "(beyond both species), with total = both species + synergy:")
    print(f"  {'tournaments':>12} {'ladder':>12} {'synergy':>12} {'total':>6} teams  both+  pair")
    for r in rare_pairs[:args.top]:
        print(f"  {fmt(r['tour'], r['tour_se'])} {fmt(r['ladder'], r['ladder_se'])} "
              f"{fmt(r['synergy'], r['synergy_se'])} {r['total']:+6.1f} {r['tour_teams']:5d}  "
              f"{'yes' if r['agree'] else ' - ':>5}  {r['pair']}")

    best = [r for r in rare_pairs if r["agree"]][:8]
    co = cores(pr, ft, fl, best)
    write("cores", co)
    print("\n4. Cores: best thirds for the pairs above that both sources back (points per game; adds = third alone + "
          "its synergy with each):")
    for r in co:
        print(f"  {r['core_total']:+5.1f}  {r['pair']} + {r['third']}  (adds {r['adds']:+.1f}: alone "
              f"{r['third_alone']:+.1f}, with each {r['with_a']:+.1f} / {r['with_b']:+.1f}; "
              f"on {100 * r['third_share']:.1f}% of teams)")


if __name__ == "__main__":
    main()
