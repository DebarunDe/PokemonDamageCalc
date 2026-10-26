"""Pokemon Showdown replays for game-level data: who was brought, who led, who Mega Evolved, what was used, who won.

Team sheets (meta.sqlite) only say which six a player registered and their record. Replays say what happened in
each game, so a pairing can be judged within the same teams: games where both led against games where one did.

    python analysis/replays.py fetch --pages 100      # newest public ladder replays of the format
    python analysis/replays.py parse                  # turn logs into one row per side per game
    python analysis/replays.py pair Raichu Sneasler   # within-team test for a pair: led together or apart

Replays are stored in analysis/data/replays.sqlite (not committed).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DB = Path(__file__).parent / "data" / "replays.sqlite"
FORMAT = "gen9championsvgc2026regmc"
SITE = "https://replay.pokemonshowdown.com"


def connect():
    DB.parent.mkdir(exist_ok=True)
    db = sqlite3.connect(DB)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS replays (id TEXT PRIMARY KEY, format TEXT, uploadtime INTEGER, rating INTEGER,
                                            p1 TEXT, p2 TEXT, log TEXT);
        CREATE TABLE IF NOT EXISTS sides (id TEXT, side TEXT, player TEXT, rating INTEGER, opp_rating INTEGER,
                                          won INTEGER, preview TEXT, brought TEXT, leads TEXT, mega TEXT,
                                          moves TEXT, items TEXT, fields TEXT, turns INTEGER,
                                          PRIMARY KEY (id, side));
    """)
    return db


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vgc-analysis"}),
                                        timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def fetch(pages, fmt=FORMAT, workers=4):
    """Page back through the format's newest public replays and download the logs we don't have yet."""
    db = connect()
    have = {r[0] for r in db.execute("SELECT id FROM replays")}
    before, listed = None, []
    for page in range(pages):
        url = f"{SITE}/search.json?format={fmt}" + (f"&before={before}" if before else "")
        batch = get(url)
        if not batch:
            break
        listed += [b for b in batch if b["id"] not in have and not b.get("password")]
        before = batch[-1]["uploadtime"]
        if len(batch) < 51:
            break
        time.sleep(0.3)
    print(f"{len(listed)} new replays listed over {page + 1} pages", flush=True)

    def one(b):
        try:
            r = get(f"{SITE}/{b['id']}.json")
        except Exception:
            return None
        return (b["id"], fmt, b["uploadtime"], b.get("rating"), r["players"][0], r["players"][1], r["log"])

    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for row in pool.map(one, listed):
            if row:
                db.execute("INSERT OR IGNORE INTO replays VALUES (?, ?, ?, ?, ?, ?, ?)", row)
                done += 1
                if done % 500 == 0:
                    db.commit()
                    print(f"  {done} downloaded", flush=True)
    db.commit()
    print(f"{done} replays downloaded; {db.execute('SELECT COUNT(*) FROM replays').fetchone()[0]} stored")


SPECIES = re.compile(r"^([^,|]+)")


def base(species: str) -> str:
    """Team preview names: a Mega appears under its base species."""
    return species.split("-Mega")[0]


def parse_log(log: str) -> dict | None:
    """One game's log -> per side: player, rating, won, preview (6), brought, leads, Megas, moves used per species,
    items revealed, field conditions started (weather / terrain / Tailwind / Trick Room) and the game length."""
    sides = {s: {"player": None, "rating": None, "preview": [], "brought": [], "leads": [], "mega": [],
                 "moves": {}, "items": {}, "fields": []} for s in ("p1", "p2")}
    nick: dict[str, str] = {}
    winner, turn, started = None, 0, False
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag == "player" and len(parts) >= 4 and parts[3]:
            s = sides.get(parts[2])
            if s is not None:
                s["player"] = parts[3]
                if len(parts) >= 6 and parts[5].isdigit():
                    s["rating"] = int(parts[5])
        elif tag == "poke":
            sides[parts[2]]["preview"].append(SPECIES.match(parts[3]).group(1))
        elif tag in ("switch", "drag", "replace"):
            who, species = parts[2], SPECIES.match(parts[3]).group(1)
            s = sides[who[:2]]
            nick[who.split(": ", 1)[0][:2] + ":" + who.split(": ", 1)[1]] = species
            name = base(species)
            if name not in s["brought"]:
                s["brought"].append(name)
            if not started and len(s["leads"]) < 2 and name not in s["leads"]:
                s["leads"].append(name)
        elif tag == "detailschange":
            who, species = parts[2], SPECIES.match(parts[3]).group(1)
            nick[who.split(": ", 1)[0][:2] + ":" + who.split(": ", 1)[1]] = species
            if "-Mega" in species:
                sides[who[:2]]["mega"].append(species)
        elif tag == "move" and len(parts) >= 4:
            who = parts[2]
            species = nick.get(who.split(": ", 1)[0][:2] + ":" + who.split(": ", 1)[-1], who)
            moves = sides[who[:2]]["moves"].setdefault(base(species), [])
            if parts[3] not in moves:
                moves.append(parts[3])
        elif tag in ("-item", "-enditem") and len(parts) >= 4:
            who = parts[2]
            species = nick.get(who.split(": ", 1)[0][:2] + ":" + who.split(": ", 1)[-1], who)
            sides[who[:2]]["items"].setdefault(base(species), parts[3])
        elif tag in ("-fieldstart", "-sidestart", "-weather") and len(parts) >= 3:
            src = next((p for p in parts[3:] if p.startswith("[of] ")), None)
            effect = parts[3] if tag == "-sidestart" and len(parts) >= 4 else parts[2]
            side = (src[5:7] if src else (parts[2][:2] if tag == "-sidestart" else None))
            if side in sides and effect not in ("none", "[upkeep]"):
                sides[side]["fields"].append(effect.replace("move: ", ""))
        elif tag == "turn":
            started, turn = True, int(parts[2])
        elif tag == "win":
            winner = parts[2]
    if not winner or not all(s["player"] for s in sides.values()):
        return None
    for s in sides.values():
        s["won"] = int(s["player"] == winner)
        s["turns"] = turn
    return sides


def parse():
    db = connect()
    rows = 0
    new = "SELECT id, log FROM replays WHERE id NOT IN (SELECT DISTINCT id FROM sides)"
    for rid, log in db.execute(new).fetchall():
        g = parse_log(log)
        if not g:
            continue
        for side, other in (("p1", "p2"), ("p2", "p1")):
            s = g[side]
            db.execute("INSERT OR REPLACE INTO sides VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (rid, side, s["player"], s["rating"], g[other]["rating"], s["won"], json.dumps(s["preview"]),
                        json.dumps(s["brought"]), json.dumps(s["leads"]), json.dumps(s["mega"]),
                        json.dumps(s["moves"]), json.dumps(s["items"]), json.dumps(s["fields"]), s["turns"]))
            rows += 1
    db.commit()
    print(f"{rows} sides parsed; {db.execute('SELECT COUNT(*) FROM sides').fetchone()[0]} stored")


def expected(r, o):
    """Elo win expectation from the two ladder ratings (0.5 when either is missing)."""
    return 0.5 if not r or not o else 1 / (1 + 10 ** ((o - r) / 400))


def pair(a, b, min_rating=0, mega=None):
    """Sides whose team preview has both: wins above their rating expectation when both led, only one, or neither.
    Leads, not brought: the back two only show up if they switch in, and a winner's often never do (sides showing
    2-3 Pokemon win about 76%, sides showing all 4 about 43%), so 'brought' mostly measures losing."""
    db = connect()
    groups = {"both led": [], "only " + a + " led": [], "only " + b + " led": [], "neither led": []}
    for rating, opp, won, preview, leads, megas in db.execute(
            "SELECT rating, opp_rating, won, preview, leads, mega FROM sides"):
        if (rating or 0) < min_rating or not rating or not opp:
            continue
        preview, leads = json.loads(preview), set(json.loads(leads))
        if a not in preview or b not in preview or len(leads) < 2:
            continue
        if mega and mega not in json.loads(megas) and mega.split("-Mega")[0] in leads:
            continue  # the same species led, but not as this Mega
        has_a, has_b = a in leads, b in leads
        key = ("both led" if has_a and has_b else "only " + a + " led" if has_a else
               "only " + b + " led" if has_b else "neither led")
        groups[key].append((won, expected(rating, opp)))
    print(f"Sides with {a} and {b} in team preview" + (f", rating {min_rating}+" if min_rating else "") + ":")
    for k, g in groups.items():
        if g:
            n, w, e = len(g), sum(x[0] for x in g), sum(x[1] for x in g)
            print(f"  {k:28s} {n:5d} games  won {100 * w / n:5.1f}%  expected {100 * e / n:5.1f}%  "
                  f"above {100 * (w - e) / n:+5.1f} ± {100 * math.sqrt(0.25 / n):.1f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--pages", type=int, default=100)
    f.add_argument("--format", default=FORMAT)
    sub.add_parser("parse")
    q = sub.add_parser("pair")
    q.add_argument("a")
    q.add_argument("b")
    q.add_argument("--min-rating", type=int, default=0)
    q.add_argument("--mega", help="only count games where this Mega (e.g. Raichu-Mega-X) was the one used")
    args = p.parse_args(argv)
    if args.cmd == "fetch":
        fetch(args.pages, args.format)
    elif args.cmd == "parse":
        parse()
    else:
        pair(args.a, args.b, args.min_rating, args.mega)


if __name__ == "__main__":
    main()
