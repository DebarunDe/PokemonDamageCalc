"""Emerging-set detector: set choices (a species' item, moves or ability, alone or with a given teammate) that are
still rare but have started winning, flagged early and checked for robustness.

For each choice the signal is its lift: wins per game above player skill for teams making the choice, minus the same
for teams with that species (or, for a choice with a teammate, that set choice) that don't. A choice is flagged in
the recent window (default 14 days) when all of these hold:
    lift z-score >= 3 (the lift over its standard error)
    12+ players and 5+ events, no single player behind more than 30% of its teams
    the lift is positive in both random halves of the events
    it is still rare: at most 30% of its species' (or set choice's) teams

    python analysis/emerging.py                    # flags as of the latest event
    python analysis/emerging.py --as-of 2026-10-08
    python analysis/emerging.py --backtest         # flags at past dates and how they did over the next 7 days

Results go to analysis/output/emerging.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import zlib
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"
REGULATION = "M-C"


def member_features(name: str, item: str | None, ability: str | None, moves: list[str]) -> list[str]:
    """A member's set choices as feature keys ('Sneasler|item:Electric Seed', 'Sneasler|move:Swords Dance', ...)."""
    out = [f"{name}|item:{item}"] if item else []
    if ability:
        out.append(f"{name}|ability:{ability}")
    out += [f"{name}|move:{m}" for m in moves]
    return out


def team_features(members: list[tuple[str, str | None, str | None, list[str]]], universal=frozenset()) \
        -> tuple[set, dict]:
    """Every feature key on a team, and each key's parent: a set choice's parent is its species; a species with a
    teammate ('Sneasler + Raichu-Mega-X') has the species as its parent; a set choice with a teammate
    ('Sneasler|item:Electric Seed + Raichu-Mega-X') has the set choice as its parent. Choices in `universal`
    (made by nearly every copy of the species) are left out: they would only repeat the species."""
    keys, parent = set(), {}
    species = {m[0] for m in members}
    for name, item, ability, moves in members:
        keys.add(name)
        for mate in species - {name}:
            keys.add(f"{name} + {mate}")
            parent[f"{name} + {mate}"] = name
        for f in member_features(name, item, ability, moves):
            if f in universal:
                continue
            keys.add(f)
            parent[f] = name
            for mate in species - {name}:
                pair = f"{f} + {mate}"
                keys.add(pair)
                parent[pair] = f
    return keys, parent


class Agg:
    """Running totals for one feature: games, wins above skill, players, events, the two random halves."""
    __slots__ = ("g", "d", "players", "events", "half_g", "half_d", "teams")

    def __init__(self):
        self.g = self.d = 0.0
        self.players, self.events, self.teams = Counter(), set(), 0
        self.half_g, self.half_d = [0.0, 0.0], [0.0, 0.0]

    def add(self, g, d, player, event):
        self.g += g
        self.d += d
        self.players[player] += 1
        self.events.add(event)
        self.teams += 1
        h = zlib.crc32(event.encode()) % 2
        self.half_g[h] += g
        self.half_d[h] += d


def lift(a: Agg, p: Agg) -> dict | None:
    """The feature's lift over the rest of its parent, with standard error and the lift in each half."""
    rest_g, rest_d = p.g - a.g, p.d - a.d
    if a.g <= 0 or rest_g <= 0:
        return None
    value = 100 * (a.d / a.g - rest_d / rest_g)
    se = 100 * (0.25 / a.g + 0.25 / rest_g) ** 0.5
    halves = []
    for h in (0, 1):
        rg, rd = p.half_g[h] - a.half_g[h], p.half_d[h] - a.half_d[h]
        halves.append(100 * (a.half_d[h] / a.half_g[h] - rd / rg) if a.half_g[h] > 0 and rg > 0 else None)
    return {"lift": value, "se": se, "z": value / se, "halves": halves}


def load(db):
    from signals import player_skill
    skill = player_skill(db).set_index(["tournament", "player"])["skill"].to_dict()
    teams, record = defaultdict(list), {}
    for t, p, name, item, ability, moves, w, l, day in db.execute(
            "SELECT m.tournament, m.player, m.pokemon, m.item, m.ability, m.moves, m.wins, m.losses, t.date "
            "FROM team_members m JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)):
        teams[(t, p)].append((name, item, ability, json.loads(moves) if moves else []))
        record[(t, p)] = (w or 0, l or 0, day[:10])
    # choices made by 80%+ of a species' copies are part of the species, not a choice
    species_n, feature_n = Counter(), Counter()
    for members in teams.values():
        for name, item, ability, moves in members:
            species_n[name] += 1
            feature_n.update(set(member_features(name, item, ability, moves)))
    universal = frozenset(f for f, c in feature_n.items() if c >= 0.8 * species_n[f.split("|")[0]])
    out = []
    for key, members in teams.items():
        w, l, day = record[key]
        if w + l and key in skill:
            keys, parent = team_features(members, universal)
            out.append((key, date.fromisoformat(day), w + l, w - skill[key] * (w + l), keys, parent))
    return out


def scan(teams, start, end, min_z=3.0, min_players=12, min_events=5, max_player_share=0.3, max_share=0.3):
    """Flags among teams played in [start, end)."""
    aggs, parents = defaultdict(Agg), {}
    for (t, p), day, g, d, keys, parent in teams:
        if start <= day < end:
            for k in keys:
                aggs[k].add(g, d, p, t)
            parents.update(parent)
    seen = {}
    for k, a in aggs.items():
        par = parents.get(k)
        if par is None or len(a.players) < min_players or len(a.events) < min_events:
            continue
        p = aggs[par]
        if a.teams > max_share * p.teams or max(a.players.values()) > max_player_share * a.teams:
            continue
        s = lift(a, p)
        if s and s["z"] >= min_z and all(h is not None and h > 0 for h in s["halves"]):
            # choices that pick out exactly the same teams (every Rillaboom with Grassy Glide also has Fake Out) are
            # one finding: keep the shortest name
            same = (a.teams, round(a.g), round(a.d, 6))
            if same in seen and len(seen[same]["feature"]) <= len(k):
                continue
            seen[same] = {"feature": k, "parent": par, "teams": a.teams, "share": a.teams / p.teams,
                          "players": len(a.players), "events": len(a.events), "games": int(a.g), **s}
    flags = list(seen.values())
    return sorted(flags, key=lambda f: -f["z"]), aggs, parents


def backtest(teams, window, horizon):
    """Run the detector at past dates; for each flag, its lift over the next `horizon` days. Near-misses (z between
    1.5 and 3, otherwise the same filters) are the control."""
    first, last = min(t[1] for t in teams), max(t[1] for t in teams)
    as_of = first + timedelta(days=window)
    flagged, control = [], []
    while as_of + timedelta(days=horizon) <= last + timedelta(days=1):
        flags, _, _ = scan(teams, as_of - timedelta(days=window), as_of)
        near, _, _ = scan(teams, as_of - timedelta(days=window), as_of, min_z=1.5)
        near = [f for f in near if f["z"] < 3]
        _, fwd, fparents = scan(teams, as_of, as_of + timedelta(days=horizon), min_players=10**9)
        for group, out in ((flags, flagged), (near, control)):
            for f in group:
                a, p = fwd.get(f["feature"]), fwd.get(f["parent"])
                s = lift(a, p) if a and p and a.g >= 20 else None
                if s:
                    out.append({**f, "as_of": as_of.isoformat(), "fwd_lift": s["lift"], "fwd_se": s["se"],
                                "fwd_games": int(a.g)})
        as_of += timedelta(days=3)
    return flagged, control


def summarise(rows, label):
    if not rows:
        print(f"  {label}: none")
        return
    w = [1 / r["fwd_se"] ** 2 for r in rows]
    mean = sum(r["fwd_lift"] * x for r, x in zip(rows, w)) / sum(w)
    print(f"  {label}: {len(rows)} flags, forward lift {mean:+.2f} ± {sum(w) ** -0.5:.2f} "
          f"({sum(r['fwd_lift'] > 0 for r in rows)}/{len(rows)} positive)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--as-of", help="flag as of this date (default: the day after the latest event)")
    p.add_argument("--window", type=int, default=14, help="days of events to look at")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--horizon", type=int, default=7, help="days after a flag to check it over (backtest)")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args(argv)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    teams = load(db)
    if args.backtest:
        flagged, control = backtest(teams, args.window, args.horizon)
        print(f"Backtest: detector run every 3 days on the previous {args.window} days; each flag checked over the "
              f"next {args.horizon} days (lift = teams with the choice vs the rest of its species / set choice)")
        summarise(flagged, "flagged (z >= 3)")
        summarise(control, "near-misses (1.5 <= z < 3)")
        seen = {}
        for r in flagged:
            seen.setdefault(r["feature"], r)
        print("\nFirst flag date of each flagged choice (earliest 20):")
        for r in sorted(seen.values(), key=lambda r: r["as_of"])[:20]:
            print(f"  {r['as_of']}  lift {r['lift']:+5.1f} (z {r['z']:.1f}) -> next {args.horizon} days "
                  f"{r['fwd_lift']:+5.1f} ± {r['fwd_se']:.1f}  {r['feature']}")
        rows = flagged
    else:
        last = max(t[1] for t in teams) + timedelta(days=1)
        as_of = date.fromisoformat(args.as_of) if args.as_of else last
        rows, _, _ = scan(teams, as_of - timedelta(days=args.window), as_of)
        print(f"Flags as of {as_of} (events in the previous {args.window} days):")
        for r in rows[:args.top]:
            print(f"  lift {r['lift']:+5.1f} ± {r['se']:.1f} (z {r['z']:.1f}; halves {r['halves'][0]:+.1f} / "
                  f"{r['halves'][1]:+.1f})  {r['teams']:3d} teams, {100 * r['share']:.0f}% of parent, "
                  f"{r['players']} players, {r['events']} events  {r['feature']}")
    if rows:
        OUT.mkdir(exist_ok=True)
        with open(OUT / "emerging.csv", "w", newline="") as f:
            out = csv.DictWriter(f, fieldnames=list(rows[0]))
            out.writeheader()
            out.writerows(rows)


if __name__ == "__main__":
    main()
