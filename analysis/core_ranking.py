"""Rank cores (2, 3 or 4 Pokemon on the same team) by how the teams carrying them did in real tournaments.

Score: wins per game above what the players' skill predicts (skill from analysis/signals.py, leave-one-out), shrunk
toward the average by sample size with the empirical-Bayes prior from analysis/backtest.py, so a core can't top
the list on a handful of lucky games. For each core it also reports what the ranking should be checked against:

    players    how many different players used it, and the biggest single player's share of its games
    official   the same score at official Regionals only, and at online events only
    recent     the same score since the current meta settled (RECENT), and before it

Usage:
    python analysis/core_ranking.py                 # pairs, trios and fours
    python analysis/core_ranking.py --size 3 --top 30 --min-games 300

Results go to analysis/output/core_ranking_<size>.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import shrink
from signals import player_skill

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"
REGULATION = "M-C"
RECENT = "2026-09-26"


def load(db):
    skill = player_skill(db).set_index(["tournament", "player"])["skill"].to_dict()
    teams, info = defaultdict(set), {}
    for t, p, mon, w, l, org, date in db.execute(
            "SELECT m.tournament, m.player, m.pokemon, m.wins, m.losses, t.organizer, t.date FROM team_members m "
            "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)):
        teams[(t, p)].add(mon)
        kind = "official" if org == "rk9" else "online"
        info[(t, p)] = (w or 0, l or 0, kind, "recent" if date >= RECENT else "early")
    return teams, info, skill


def rank(size, teams, info, skill, min_games):
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))  # core -> slice -> [wins, games, expected wins]
    players = defaultdict(Counter)
    for key, mons in teams.items():
        w, l, kind, era = info[key]
        if not w + l or key not in skill:
            continue
        expected = skill[key] * (w + l)
        for core in itertools.combinations(sorted(mons), size):
            for part in ("all", kind, era):
                a = agg[core][part]
                a[0] += w
                a[1] += w + l
                a[2] += expected
            players[core][key[1]] += w + l
    rows = []
    for core, parts in agg.items():
        w, g, e = parts["all"]
        if g < min_games:
            continue
        row = {"core": " / ".join(core), "megas": sum("-Mega" in m for m in core), "games": g,
               "raw": 100 * (w - e) / g}
        for part in ("official", "online", "recent", "early"):
            pw, pg, pe = parts.get(part, (0, 0, 0.0))
            row[part] = round(100 * (pw - pe) / pg, 2) if pg else None
            row[f"{part}_games"] = pg
        row["players"] = len(players[core])
        row["top_player_share"] = round(max(players[core].values()) / g, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"], prior = shrink(df["raw"] / 100, df["games"])
    df["score"] *= 100
    df = df.sort_values("score", ascending=False)
    df.attrs["prior_games"] = prior
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--size", type=int, nargs="*", default=[2, 3, 4])
    p.add_argument("--min-games", type=int, default=200, help="games a core needs to be ranked")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    teams, info, skill = load(db)
    OUT.mkdir(exist_ok=True)
    for size in args.size:
        df = rank(size, teams, info, skill, args.min_games)
        df.to_csv(OUT / f"core_ranking_{size}.csv", index=False, float_format="%.2f", quoting=csv.QUOTE_MINIMAL)
        print(f"\n=== {size}-Pokemon cores: {len(df)} with {args.min_games}+ games "
              f"(shrinkage prior worth {df.attrs['prior_games']:.0f} games) ===")
        print(f"  {'score':>6} {'raw':>6} {'games':>6} {'players':>7} {'top%':>5} {'official':>8} {'online':>7} "
              f"{'recent':>7} {'early':>6}  core")
        for _, r in df.head(args.top).iterrows():
            fmt = lambda v: f"{v:+.1f}" if v is not None and not pd.isna(v) else "-"
            print(f"  {r['score']:+6.2f} {r['raw']:+6.2f} {r['games']:6d} {r['players']:7d} "
                  f"{100 * r['top_player_share']:4.0f}% {fmt(r['official']):>8} {fmt(r['online']):>7} "
                  f"{fmt(r['recent']):>7} {fmt(r['early']):>6}  {r['core']}")


if __name__ == "__main__":
    main()
