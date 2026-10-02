"""Download usage and tournament data into a local SQLite database.

Sources:
- Showdown ladder stats (smogon.com/stats): monthly usage, moves, items,
  abilities, Stat Point spreads, teammates and checks/counters for each
  Champions VGC regulation, best-of-1 and best-of-3, at several rating cutoffs.
- Limitless (play.limitlesstcg.com): tournaments, standings, win/loss records
  and full team lists.

Raw downloads are cached under analysis/data/raw, so re-running only fetches
what is new (finished tournaments and past months never change).

Usage:
    python analysis/ingest.py                 # everything
    python analysis/ingest.py showdown        # ladder stats only
    python analysis/ingest.py limitless       # tournaments only
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

from champcalc import Calculator

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"
DB = DATA / "meta.sqlite"

SHOWDOWN = "https://www.smogon.com/stats"
LIMITLESS = "https://play.limitlesstcg.com/api"
USER_AGENT = "PokemonDamageCalc-research/0.1 (github.com/DebarunDe/PokemonDamageCalc)"

# Showdown format id for each regulation; "bo3" adds the best-of-3 ladder.
SHOWDOWN_FORMATS = {
    "M-A": "gen9championsvgc2026regma",
    "M-B": "gen9championsvgc2026regmb",
    "M-C": "gen9championsvgc2026regmc",
}
RATINGS = (0, 1760)
# Limitless tournaments smaller than this are mostly practice rooms.
MIN_PLAYERS = 8
REQUEST_DELAY = 0.5  # seconds between uncached requests, to be polite

SCHEMA = """
CREATE TABLE IF NOT EXISTS showdown_usage (
    regulation TEXT, month TEXT, ladder TEXT, rating INTEGER, battles INTEGER,
    pokemon TEXT, usage REAL, raw_count INTEGER,
    PRIMARY KEY (regulation, month, ladder, rating, pokemon)
);
CREATE TABLE IF NOT EXISTS showdown_details (
    regulation TEXT, month TEXT, ladder TEXT, rating INTEGER,
    pokemon TEXT, kind TEXT, value TEXT, share REAL,
    PRIMARY KEY (regulation, month, ladder, rating, pokemon, kind, value)
);
CREATE TABLE IF NOT EXISTS showdown_checks (
    regulation TEXT, month TEXT, ladder TEXT, rating INTEGER,
    pokemon TEXT, opponent TEXT, n REAL, p REAL, d REAL,
    PRIMARY KEY (regulation, month, ladder, rating, pokemon, opponent)
);
CREATE TABLE IF NOT EXISTS tournaments (
    id TEXT PRIMARY KEY, regulation TEXT, name TEXT, date TEXT, players INTEGER, organizer INTEGER
);
CREATE TABLE IF NOT EXISTS team_members (
    tournament TEXT, player TEXT, placing INTEGER, wins INTEGER, losses INTEGER, ties INTEGER,
    dropped INTEGER, slot INTEGER, pokemon TEXT, item TEXT, ability TEXT, nature TEXT, moves TEXT,
    PRIMARY KEY (tournament, player, slot)
);
"""


def fetch(url: str, cache: Path | None = None, *, json_response: bool = True):
    """GET a URL, optionally caching the body (gzipped) at `cache`."""
    if cache and cache.exists():
        body = gzip.decompress(cache.read_bytes())
    else:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if attempt == 3:
                    raise
            except urllib.error.URLError:
                if attempt == 3:
                    raise
            time.sleep(2 ** attempt)
        time.sleep(REQUEST_DELAY)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(gzip.compress(body))
    return json.loads(body) if json_response else body.decode("utf-8")


# --- Showdown -------------------------------------------------------------------------------------

def showdown_months() -> list[str]:
    index = fetch(f"{SHOWDOWN}/", json_response=False)
    return sorted(set(re.findall(r'href="(\d{4}-\d{2})/"', index)))


def ingest_showdown(db: sqlite3.Connection, months: list[str] | None = None) -> None:
    months = months or [m for m in showdown_months() if m >= "2026-04"]
    current = max(months)
    for month in months:
        listing = fetch(f"{SHOWDOWN}/{month}/chaos/", json_response=False) or ""
        for regulation, base in SHOWDOWN_FORMATS.items():
            for ladder, fmt in (("bo1", base), ("bo3", base + "bo3")):
                for rating in RATINGS:
                    name = f"{fmt}-{rating}.json"
                    if f'href="{name}"' not in listing:
                        continue
                    # The current month is still being published; don't cache it.
                    cache = None if month == current else RAW / "showdown" / month / (name + ".gz")
                    chaos = fetch(f"{SHOWDOWN}/{month}/chaos/{name}", cache)
                    if chaos:
                        store_chaos(db, regulation, month, ladder, rating, chaos)
                        print(f"  showdown {month} {regulation} {ladder} {rating}: {len(chaos['data'])} Pokemon")
    db.commit()


def store_chaos(db, regulation, month, ladder, rating, chaos) -> None:
    key = (regulation, month, ladder, rating)
    battles = chaos["info"]["number of battles"]
    for table in ("showdown_usage", "showdown_details", "showdown_checks"):
        db.execute(f"DELETE FROM {table} WHERE regulation=? AND month=? AND ladder=? AND rating=?", key)
    usage, details, checks = [], [], []
    for pokemon, d in chaos["data"].items():
        usage.append((*key, battles, pokemon, d["usage"], d["Raw count"]))
        total = sum(d["Abilities"].values()) or 1
        for kind, field in (("ability", "Abilities"), ("item", "Items"), ("move", "Moves"),
                            ("spread", "Spreads"), ("teammate", "Teammates")):
            entries = d.get(field) or {}
            if kind == "spread":  # thousands of near-unique spreads: keep the common ones
                entries = dict(sorted(entries.items(), key=lambda kv: -kv[1])[:50])
            details.extend((*key, pokemon, kind, value, weight / total) for value, weight in entries.items())
        for opponent, c in (d.get("Checks and Counters") or {}).items():
            checks.append((*key, pokemon, opponent, c["n"], c["p"], c["d"]))
    db.executemany("INSERT INTO showdown_usage VALUES (?,?,?,?,?,?,?,?)", usage)
    db.executemany("INSERT INTO showdown_details VALUES (?,?,?,?,?,?,?,?)", details)
    db.executemany("INSERT INTO showdown_checks VALUES (?,?,?,?,?,?,?,?,?)", checks)


# --- Limitless ------------------------------------------------------------------------------------

class SpeciesNames:
    """Maps Limitless team entries (base species id + held item) to calc names,
    turning a base species holding its Mega Stone into the Mega forme."""

    def __init__(self) -> None:
        calc = Calculator()
        self.calc = calc
        self.by_id = {}
        self.mega_by_stone = {}
        for name in calc.all_species:
            self.by_id.setdefault(self._id(name), name)
            stone = calc.species(name)["mega_stone"]
            if stone and "-Mega" in name:
                self.mega_by_stone[(self._id(name.split("-Mega")[0]), stone)] = name
        self.by_id["aegislash"] = "Aegislash-Both"

    @staticmethod
    def _id(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def __call__(self, species_id: str, item: str | None) -> str:
        sid = self._id(species_id)
        mega = self.mega_by_stone.get((sid, item))
        if mega:
            return mega
        return self.by_id.get(sid) or species_id


def ingest_limitless(db: sqlite3.Connection) -> None:
    names = SpeciesNames()
    for regulation in SHOWDOWN_FORMATS:
        tournaments, page = [], 1
        while True:
            batch = fetch(f"{LIMITLESS}/tournaments?game=VGC&format={regulation}&limit=500&page={page}")
            if not batch:
                break
            tournaments.extend(batch)
            page += 1
        tournaments = [t for t in tournaments if t.get("players", 0) >= MIN_PLAYERS]
        print(f"  limitless {regulation}: {len(tournaments)} tournaments with {MIN_PLAYERS}+ players")
        for i, t in enumerate(tournaments, 1):
            db.execute("INSERT OR REPLACE INTO tournaments VALUES (?,?,?,?,?,?)",
                       (t["id"], regulation, t["name"], t["date"], t["players"], t.get("organizerId")))
            done = db.execute("SELECT 1 FROM team_members WHERE tournament=? LIMIT 1", (t["id"],)).fetchone()
            if done:
                continue
            # Recent events may still be running; only cache ones at least two days old.
            finished = t["date"] < time.strftime("%Y-%m-%dT%H:%M", time.gmtime(time.time() - 2 * 86400))
            cache = RAW / "limitless" / f"{t['id']}.json.gz" if finished else None
            standings = fetch(f"{LIMITLESS}/tournaments/{t['id']}/standings", cache) or []
            rows = []
            for s in standings:
                record = s.get("record") or {}
                for slot, mon in enumerate(s.get("decklist") or []):
                    rows.append((
                        t["id"], s.get("player"), s.get("placing"),
                        record.get("wins", 0), record.get("losses", 0), record.get("ties", 0),
                        1 if s.get("drop") else 0, slot,
                        names(mon.get("id") or mon.get("name", ""), mon.get("item")),
                        mon.get("item"), mon.get("ability"), mon.get("nature"),
                        json.dumps(mon.get("attacks") or []),
                    ))
            db.executemany("INSERT OR REPLACE INTO team_members VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            if i % 25 == 0:
                db.commit()
                print(f"    {regulation}: {i}/{len(tournaments)} tournaments", file=sys.stderr)
        db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", choices=("all", "showdown", "limitless"), default="all")
    parser.add_argument("--months", help="comma-separated Showdown months, e.g. 2026-08,2026-09")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.executescript(SCHEMA)
    if args.source in ("all", "showdown"):
        ingest_showdown(db, args.months.split(",") if args.months else None)
    if args.source in ("all", "limitless"):
        ingest_limitless(db)
    for table in ("showdown_usage", "showdown_details", "showdown_checks", "tournaments", "team_members"):
        print(f"{table}: {db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]} rows")
    db.close()


if __name__ == "__main__":
    main()
