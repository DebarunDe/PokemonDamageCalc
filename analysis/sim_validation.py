"""Check the doubles simulator (analysis/doubles_sim.py) against real tournament results.

Real side: the most played pairs (or trios) in Regulation M-C tournaments, scored by how the teams carrying them did
beyond what their players' skill predicts (wins above skill per game, as in analysis/signals.py).

Simulated side, two modes:
    team      (default) the real teams carrying the core: a sample of them, each bringing the core plus others from
              its own six, plays 4v4 against real opponents. This measures the same thing as the real number.
              Both sides pick their four at team preview (doubles_sim.preview), or at random with --preview random.
    isolated  the core alone (2v2 or 3v3) against as many of each opponent's Pokemon.

The report gives the correlation between the two, the best correlation the noise on both sides allows, and the
cores the simulator over- and underrates most. Use it as the scorecard when changing the simulator.

Usage:
    python analysis/sim_validation.py                         # pairs, team mode (~15 min on 4 cores)
    python analysis/sim_validation.py --size 3 --cores 40     # trios
    python analysis/sim_validation.py --mode isolated

Results go to analysis/output/sim_validation_<mode>_<size>.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import random
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"

_sim = None
_opponents: list = []


def _init(opponents):
    global _sim, _opponents
    import doubles_sim
    _sim, _opponents = doubles_sim, opponents


def _specs(names):
    specs = [_sim.real_spec(n) for n in names]
    return [s for s in specs if s is not None]


def _play(job):
    core_names, mode, teams, seed, smart = job
    core = _specs(core_names)
    results = []
    for k, team_names in enumerate(teams):
        team = _specs(team_names)
        for j, opp_names in enumerate(_opponents):
            opp = _specs(opp_names)
            rng = random.Random(hash((seed, k, j)) & 0xFFFFFFFF)
            if mode == "isolated":
                if len(opp) < len(core):
                    continue
                ours, theirs = core, _sim.bring(opp, len(core), rng)
            else:
                if len(team) < 4 or len(opp) < 4:
                    continue
                if smart:
                    ours = _sim.preview(team, opp, 4, rng, fixed=core)
                    theirs = _sim.preview(opp, team, 4, rng)
                else:
                    ours, theirs = _sim.bring(team, 4, rng, fixed=core), _sim.bring(opp, 4, rng)
            results.append(_sim.Battle(ours, theirs, random.Random(rng.randrange(10**9))).run())
    return core_names, results


def real_cores(size: int, n: int, max_teams: int, seed: int = 3):
    """The n most played cores of this size: (core, wins above skill per game in %, its standard error, games,
    up to max_teams of the real teams carrying it)."""
    import doubles_sim
    import signals
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    skill = signals.player_skill(db).set_index(["tournament", "player"])["skill"].to_dict()
    teams, record = defaultdict(list), {}
    for t, p, mon, w, l in db.execute(
            "SELECT m.tournament, m.player, m.pokemon, m.wins, m.losses FROM team_members m "
            "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ? ORDER BY m.slot",
            (doubles_sim.REGULATION,)):
        teams[(t, p)].append(mon)
        record[(t, p)] = (w or 0, l or 0)
    agg = defaultdict(lambda: [0, 0, 0.0, []])
    for key, mons in teams.items():
        w, l = record[key]
        if not w + l or key not in skill:
            continue
        for core in itertools.combinations(sorted(set(mons)), size):
            if sum("-Mega" in m for m in core) > 1:
                continue
            a = agg[core]
            a[0] += w
            a[1] += w + l
            a[2] += skill[key] * (w + l)
            a[3].append(mons)
    use = Counter(m for mons in teams.values() for m in mons)
    out = []
    for core, (w, g, expected, carriers) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        if not all(doubles_sim.real_spec(m) for m in core):
            continue
        order = tuple(sorted(core, key=lambda m: -use[m]))  # the most used lead
        sample = random.Random(seed).sample(carriers, min(max_teams, len(carriers)))
        out.append((order, 100 * (w - expected) / g, 100 * (0.25 / g) ** 0.5, g, sample))
        if len(out) == n:
            break
    return out


def weighted_corr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--size", type=int, default=2)
    p.add_argument("--cores", type=int, default=70)
    p.add_argument("--mode", choices=("team", "isolated"), default="team")
    p.add_argument("--teams-per-core", type=int, default=8, help="real teams carrying each core (team mode)")
    p.add_argument("--opponents", type=int, default=40)
    p.add_argument("--preview", choices=("smart", "random"), default="smart",
                   help="how both sides pick their four in team mode")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    import doubles_sim
    cores = real_cores(args.size, args.cores, args.teams_per_core if args.mode == "team" else 1)
    opponents = [names for _, names in doubles_sim.opponent_teams(300)]
    opponents = random.Random(5).sample(opponents, min(args.opponents, len(opponents)))
    jobs = [(c[0], args.mode, c[4] if args.mode == "team" else [[]], i, args.preview == "smart")
            for i, c in enumerate(cores)]
    print(f"{len(cores)} real {args.size}-cores, {args.mode} mode, {len(opponents)} opponents", flush=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(opponents,)) as pool:
        simulated = dict(pool.imap_unordered(_play, jobs))

    rows = []
    for core, real, real_se, games, _ in cores:
        r = simulated[core]
        rows.append({"core": " / ".join(core), "sim": 100 * statistics.mean(r),
                     "sim_se": 100 * statistics.pstdev(r) / len(r) ** 0.5, "real": real, "real_se": real_se,
                     "games": games, "battles": len(r)})
    x, y = np.array([r["sim"] for r in rows]), np.array([r["real"] for r in rows])
    w = np.array([r["games"] for r in rows], float)
    xe, ye = np.array([r["sim_se"] for r in rows]), np.array([r["real_se"] for r in rows])
    rel_x = max(0.0, 1 - np.mean(xe ** 2) / np.var(x))
    rel_y = max(0.0, 1 - np.mean(ye ** 2) / np.var(y))
    rank = lambda v: np.argsort(np.argsort(v))
    slope, intercept = np.polyfit(x, y, 1)
    for r in rows:
        r["residual"] = r["real"] - (slope * r["sim"] + intercept)
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"sim_validation_{args.mode}_{args.size}.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    r_all, r_w = np.corrcoef(x, y)[0, 1], weighted_corr(x, y, w)
    print(f"\ncorrelation r = {r_all:+.2f}   weighted by games {r_w:+.2f}   "
          f"rank {np.corrcoef(rank(x), rank(y))[0, 1]:+.2f}")
    print(f"reliability: real {rel_y:.2f}, simulated {rel_x:.2f} -> best attainable r ~ {np.sqrt(rel_x * rel_y):.2f}")
    by_res = sorted(rows, key=lambda r: r["residual"])
    for label, part in (("overrates", by_res[:6]), ("underrates", by_res[::-1][:6])):
        print(f"most {label}: " + "; ".join(f"{r['core']} (sim {r['sim']:.0f}%, real {r['real']:+.1f})" for r in part))


if __name__ == "__main__":
    main()
