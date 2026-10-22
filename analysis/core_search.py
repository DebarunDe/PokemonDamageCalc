"""Search for the best 2-, 3- and 4-Pokemon cores with the doubles battle simulator (analysis/doubles_sim.py).

Candidates are the most used Pokemon in recent tournaments, each with its most common real set; a core holds at
most one Mega. Each stage screens cores on a few opponent teams, then plays the best again on many more:

    pairs    every pair, 2v2 against two of an opponent's six
    trios    the best pairs plus each other candidate, 3v3 against three
    quads    the best trios plus each other candidate, 4v4 against four (a full bring)

In a trio or quad the core's best pair (from the pair stage) leads and the rest come in, best first. Every core
plays the same opponents with the same random seeds, so differences between cores aren't luck of the draw.

Usage:
    python analysis/core_search.py                       # ~2 hours on 4 cores
    python analysis/core_search.py --candidates 30 --screen 20 --final 200
    python analysis/core_search.py --include Metagross-Mega Annihilape    # add Pokemon outside the top N

Results go to analysis/output/cores_<size>.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import random
import sqlite3
import statistics
import time
from collections import Counter
from pathlib import Path

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"

_sim = None
_teams: list = []


def _init(teams):
    global _sim, _teams
    import doubles_sim
    _sim, _teams = doubles_sim, teams


def _play(job):
    """Results of one core (names, in order) against the given opponent team indices."""
    names, indices = job
    core = [_sim.real_spec(n) for n in names]
    results = {"official": [], "online": []}
    for i in indices:
        pool, team_names = _teams[i]
        team = [s for s in (_sim.real_spec(n) for n in team_names) if s is not None]
        if len(team) >= len(core):
            results[pool] += _sim.play_core(core, team, seed=i)
    return names, results


def candidates(n: int) -> list[str]:
    import doubles_sim as sim
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    counts = Counter(name for (name,) in db.execute(
        "SELECT m.pokemon FROM team_members m JOIN tournaments t ON t.id = m.tournament "
        "WHERE t.regulation = ? AND t.date >= ?", (sim.REGULATION, sim.RECENT)))
    return [name for name, _ in counts.most_common(n * 2) if sim.real_spec(name) is not None][:n]


def is_mega(name: str) -> bool:
    return "-Mega" in name


class Search:
    def __init__(self, pool, teams, screen, final):
        self.pool, self.teams = pool, teams
        order = list(range(len(teams)))
        random.Random(11).shuffle(order)
        self.screen_idx, self.final_idx = order[:screen], list(range(len(teams)))
        self.screen_scores: dict[tuple, float] = {}
        self.final_rows: dict[tuple, dict] = {}

    def run(self, cores, indices, label):
        t0 = time.time()
        out = {}
        for i, (names, res) in enumerate(self.pool.imap_unordered(_play, [(c, indices) for c in cores], chunksize=2)):
            out[tuple(names)] = res
            if (i + 1) % 100 == 0:
                print(f"    {label}: {i + 1}/{len(cores)} ({time.time() - t0:.0f}s)", flush=True)
        return out

    def screen(self, cores, label):
        for names, res in self.run(cores, self.screen_idx, f"screen {label}").items():
            every = res["official"] + res["online"]
            self.screen_scores[names] = sum(every) / len(every) if every else 0.0

    def final(self, cores, label):
        for names, res in self.run(cores, self.final_idx, f"final {label}").items():
            every = res["official"] + res["online"]
            mean = sum(every) / len(every)
            se = statistics.pstdev(every) / len(every) ** 0.5
            self.final_rows[names] = {
                "core": " / ".join(names), "size": len(names), "win": round(100 * mean, 1), "se": round(100 * se, 1),
                "official": round(100 * statistics.mean(res["official"]), 1) if res["official"] else "",
                "online": round(100 * statistics.mean(res["online"]), 1) if res["online"] else "",
                "battles": len(every), "screen": round(100 * self.screen_scores.get(names, float("nan")), 1)}


def write(rows, size):
    OUT.mkdir(exist_ok=True)
    rows = sorted((r for r in rows if r["size"] == size), key=lambda r: -r["win"])
    with open(OUT / f"cores_{size}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nTop {size}-Pokemon cores (lead first):")
    for r in rows[:15]:
        print(f"  {r['win']:5.1f}% (+/- {r['se']})  official {r['official']}%  online {r['online']}%   {r['core']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--candidates", type=int, default=40)
    p.add_argument("--screen", type=int, default=30, help="opponent teams for the quick screen")
    p.add_argument("--final", type=int, default=300, help="opponent teams for the full test")
    p.add_argument("--keep", type=int, nargs=3, default=(60, 40, 30), help="cores re-tested in full per size")
    p.add_argument("--extend", type=int, nargs=2, default=(20, 15), help="pairs / trios extended to the next size")
    p.add_argument("--include", nargs="*", default=[], help="extra candidates, e.g. Metagross-Mega")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    import doubles_sim as sim
    names = candidates(args.candidates)
    names += [n for n in args.include if n not in names and sim.real_spec(n) is not None]
    teams = [(pool, t) for pool, t in sim.opponent_teams(args.final)]
    print(f"{len(names)} candidates: {', '.join(names)}\n{len(teams)} opponent teams", flush=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(teams,)) as pool:
        s = Search(pool, teams, args.screen, args.final)

        pairs = [c for c in itertools.combinations(names, 2) if sum(map(is_mega, c)) <= 1]
        print(f"\npairs: screening {len(pairs)}", flush=True)
        s.screen(pairs, "pairs")
        top = sorted(pairs, key=lambda c: -s.screen_scores[c])[:args.keep[0]]
        s.final(top, "pairs")
        write(s.final_rows.values(), 2)

        def pair_score(a, b):
            return s.screen_scores.get((a, b), s.screen_scores.get((b, a), 0.0))
        solo = {}
        for n in names:
            scores = [pair_score(n, m) for m in names if (n, m) in s.screen_scores or (m, n) in s.screen_scores]
            solo[n] = statistics.mean(scores) if scores else 0.0

        def ordered(members):
            """Best pair leads; the rest follow, strongest first."""
            lead = max(itertools.combinations(members, 2), key=lambda c: pair_score(*c))
            back = sorted((m for m in members if m not in lead), key=lambda m: -solo[m])
            return tuple(lead) + tuple(back)

        def extend(bases, size):
            seen, out = set(), []
            for base in bases:
                for n in names:
                    members = frozenset(base) | {n}
                    if n in base or len(members) != size or sum(map(is_mega, members)) > 1 or members in seen:
                        continue
                    seen.add(members)
                    out.append(ordered(members))
            return out

        for size, keep, ext in ((3, args.keep[1], args.extend[0]), (4, args.keep[2], args.extend[1])):
            prev = sorted((r for r in s.final_rows.values() if r["size"] == size - 1), key=lambda r: -r["win"])
            bases = [tuple(r["core"].split(" / ")) for r in prev[:ext]]
            cores = extend(bases, size)
            print(f"\n{size}-cores: screening {len(cores)}", flush=True)
            s.screen(cores, f"{size}-cores")
            top = sorted(cores, key=lambda c: -s.screen_scores[c])[:keep]
            s.final(top, f"{size}-cores")
            write(s.final_rows.values(), size)


if __name__ == "__main__":
    main()
