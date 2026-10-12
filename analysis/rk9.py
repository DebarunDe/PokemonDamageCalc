"""Official Play! Pokémon events from RK9 Labs (rk9.gg): VGC team lists, final
standings and match records, stored in the same tables as Limitless events.

Regionals, Internationals and Worlds run on RK9, not Limitless. For each
finished event in RK9's event list, this finds the VGC roster (the one with
public team lists), reads every Masters team list, and takes each player's
final win-loss-tie record from the last round they appear in on the pairings
pages. The regulation is the oldest one in which nearly every team is legal:
each regulation's legal list contains the previous one's, so an event fits
its own regulation and every later one, but not the ones before it (M-C
Regionals are only 12-27% legal under M-B). No dates need maintaining.

Stored as tournaments.id = "rk9:<roster code>", organizer = "rk9", players are
keyed "First Last [CC]". Only Masters is stored (Juniors and Seniors are a
different field).
"""

from __future__ import annotations

import html
import json
import re
import sqlite3
import time
from datetime import date, datetime

RK9 = "https://rk9.gg"
DIVISION = "Masters"
MIN_LEGAL_SHARE = 0.95  # share of teams fully legal for a regulation to claim the event

# English team-list names for formes whose names differ from the calc's.
FORMS = {
    "Arcanine [Hisuian Form]": "Arcanine-Hisui", "Basculegion [Female]": "Basculegion-F",
    "Basculegion [Male]": "Basculegion", "Floette [Eternal Flower]": "Floette-Eternal",
    "Goodra [Hisuian Form]": "Goodra-Hisui", "Indeedee [Female]": "Indeedee-F", "Indeedee [Male]": "Indeedee",
    "Lycanroc [Dusk Form]": "Lycanroc-Dusk", "Lycanroc [Midnight Form]": "Lycanroc-Midnight",
    "Lycanroc [Midday Form]": "Lycanroc", "Maushold [Family of Four]": "Maushold",
    "Maushold [Family of Three]": "Maushold", "Meowstic [Female]": "Meowstic-F", "Meowstic [Male]": "Meowstic",
    "Ninetales [Alolan Form]": "Ninetales-Alola", "Rotom [Heat Rotom]": "Rotom-Heat", "Rotom [Wash Rotom]": "Rotom-Wash",
    "Rotom [Mow Rotom]": "Rotom-Mow", "Rotom [Frost Rotom]": "Rotom-Frost", "Rotom [Fan Rotom]": "Rotom-Fan",
    "Samurott [Hisuian Form]": "Samurott-Hisui", "Sinistcha [Masterpiece Form]": "Sinistcha",
    "Sinistcha [Unremarkable Form]": "Sinistcha", "Slowking [Galarian Form]": "Slowking-Galar",
    "Slowbro [Galarian Form]": "Slowbro-Galar", "Tauros [Paldean Form - Combat Breed]": "Tauros-Paldea-Combat",
    "Tauros [Paldean Form - Blaze Breed]": "Tauros-Paldea-Blaze", "Tauros [Paldean Form - Aqua Breed]": "Tauros-Paldea-Aqua",
    "Toxtricity [Amped Form]": "Toxtricity", "Toxtricity [Low Key Form]": "Toxtricity-Low-Key",
    "Typhlosion [Hisuian Form]": "Typhlosion-Hisui", "Zoroark [Hisuian Form]": "Zoroark-Hisui",
    "Decidueye [Hisuian Form]": "Decidueye-Hisui", "Goodra": "Goodra", "Aegislash": "Aegislash-Both",
}

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
_MONTHS = {m: i for i, m in enumerate(("January", "February", "March", "April", "May", "June", "July", "August",
                                       "September", "October", "November", "December"), 1)}


def _text(fragment: str) -> str:
    return _SPACE.sub(" ", html.unescape(_TAGS.sub(" ", fragment))).strip()


def species_name(raw: str) -> str:
    """English team-list species -> calc name (formes in brackets mapped; else the bracket is dropped)."""
    raw = _SPACE.sub(" ", html.unescape(raw)).strip()
    return FORMS.get(raw, raw.split(" [")[0])


def end_date(text: str) -> date | None:
    """'October 9-11, 2026' / 'October 31-November 1, 2026' / 'June 27, 2026' -> the last day."""
    m = re.search(r"(?:([A-Z][a-z]+) )?(\d{1,2}), (\d{4})$", text.strip())
    first_month = re.match(r"([A-Z][a-z]+)", text.strip())
    if not m or not first_month:
        return None
    month = _MONTHS.get(m.group(1) or first_month.group(1))
    return date(int(m.group(3)), month, int(m.group(2))) if month else None


def parse_events(page: str) -> list[dict]:
    """Every row of RK9's Pokémon events page: end date, event slug and name."""
    events = []
    for row in re.findall(r"<tr>(.*?)</tr>", page, re.S):
        link = re.search(r'href="/event/([^"]+)"[^>]*>([^<]+)</a>', row)
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if link and cells:
            events.append({"slug": link.group(1), "name": html.unescape(link.group(2)).strip(),
                           "end": end_date(_text(cells[0]))})
    return events


def parse_roster(page: str) -> list[dict]:
    """Roster table: player id, name, country, division, team-list link and standing."""
    body = page.split("<tbody", 1)[1].split("</tbody>", 1)[0] if "<tbody" in page else page
    players = []
    for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        cells = [_text(c) for c in re.findall(r"<td[^>]*>(.*?)</t[dh]>", row, re.S)]
        link = re.search(r'href="(/teamlist/public/[^"]+)"', row)
        if len(cells) < 6:
            continue
        players.append({"first": cells[1], "last": cells[2], "country": cells[3], "division": cells[4],
                        "standing": int(cells[-1]) if cells[-1].isdigit() else None,
                        "url": link.group(1) if link else None})
    return players


def player_key(first: str, last: str, country: str) -> str:
    return f"{first} {last} [{country}]"


def parse_team(page: str) -> list[dict]:
    """The six Pokémon of a team list. The page repeats the team in every
    language; English comes first."""
    team = []
    for block in page.split('<div class="pokemon')[1:7]:
        block = block.split("</div>", 1)[0]
        name = re.search(r'onerror="imgError\(this\)"[^>]*>\s*(.+?)\s*&nbsp;', block, re.S)
        field = lambda label: (m.group(1).strip() if (m := re.search(rf"<b>{label}:</b>\s*([^<]+?)\s*(?:&nbsp;|<)", block)) else None)
        team.append({"species": species_name(name.group(1)) if name else None, "ability": field("Ability"),
                     "item": field("Held Item"), "nature": field("Stat Alignment"),
                     "moves": [html.unescape(m).strip() for m in re.findall(r'<span class="badge">([^<]+)</span>', block)]})
    return team


def parse_round(page: str) -> dict[str, tuple[int, int, int]]:
    """Each player's running (wins, losses, ties) as printed on one round's pairings."""
    records = {}
    for name, w, l, t in re.findall(r'<span class="name">(.*?)</span>\s*\((\d+)-(\d+)-(\d+)\)', page, re.S):
        records[_text(name.replace("<br>", " "))] = (int(w), int(l), int(t))
    return records


def legal_regulation(teams: list[list[str]], regulations: dict[str, set[str]]) -> str | None:
    """Oldest regulation (in the given order) in which nearly every team is fully legal.
    Later regulations only add Pokémon, so the oldest fit is the event's own."""
    for reg, legal in regulations.items():
        ok = sum(all(p in legal for p in team) for team in teams)
        if teams and ok / len(teams) >= MIN_LEGAL_SHARE:
            return reg
    return None


def ingest_rk9(db: sqlite3.Connection, fetch, raw_dir, regulations: dict[str, set[str]], names,
               since: date | None = None, delay: float = 0.2) -> None:
    """Find finished RK9 events with VGC team lists and store their Masters division.

    fetch(url, cache, json_response=False) returns page text; names(species, item)
    maps a species and held item to the calc name (Mega formes)."""
    page = lambda url, cache=None: fetch(RK9 + url, cache, json_response=False) or ""
    today = date.today()
    events = parse_events(page("/events/pokemon"))
    done = {r[0] for r in db.execute("SELECT DISTINCT tournament FROM team_members WHERE tournament LIKE 'rk9:%'")}
    for event in events:
        if not event["end"] or event["end"] >= today or (since and event["end"] < since):
            continue  # unfinished, or older than we care about
        event_page = page(f"/event/{event['slug']}", raw_dir / "events" / f"{event['slug']}.html.gz")
        for code in dict.fromkeys(re.findall(r'href="/roster/([^"]+)"', event_page)):
            if f"rk9:{code}" in done:
                continue
            roster_page = page(f"/roster/{code}", raw_dir / code / "roster.html.gz")
            if "/teamlist/public/" not in roster_page:
                continue  # not the VGC roster (no team lists)
            players = [p for p in parse_roster(roster_page) if p["division"] == DIVISION and p["url"]]
            if not players:
                continue

            def load(p):
                p["team"] = parse_team(page(p["url"], raw_dir / code / f"{p['url'].rsplit('/', 1)[1]}.html.gz"))
                for mon in p["team"]:
                    mon["pokemon"] = names(mon["species"] or "", mon["item"])
                return [m["pokemon"] for m in p["team"]]

            # A sample decides the regulation before downloading every list.
            if not legal_regulation([load(p) for p in players[:20]], regulations):
                print(f"  rk9 {event['name']}: not a Champions regulation, skipped")
                continue
            for i, p in enumerate(players, 1):
                if "team" not in p:
                    load(p)
                if i % 200 == 0:
                    print(f"    {event['name']}: {i}/{len(players)} team lists", flush=True)
            regulation = legal_regulation([[m["pokemon"] for m in p["team"]] for p in players], regulations)
            if not regulation:
                print(f"  rk9 {event['name']}: teams not legal in any known regulation, skipped")
                continue
            records = _records(page, code, raw_dir, len(players))
            rows = []
            for p in players:
                key = player_key(p["first"], p["last"], p["country"])
                w, l, t = records.get(key, (0, 0, 0))
                for slot, mon in enumerate(p["team"]):
                    rows.append((f"rk9:{code}", key, p["standing"], w, l, t, 0, slot, mon["pokemon"], mon["item"],
                                 mon["ability"], mon["nature"], json.dumps(mon["moves"])))
            db.execute("INSERT OR REPLACE INTO tournaments VALUES (?,?,?,?,?,?)",
                       (f"rk9:{code}", regulation, event["name"], event["end"].isoformat() + "T23:00:00.000Z",
                        len(players), "rk9"))
            db.executemany("INSERT OR REPLACE INTO team_members VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            db.commit()
            matched = sum(player_key(p["first"], p["last"], p["country"]) in records for p in players)
            print(f"  rk9 {event['name']} ({regulation}): {len(players)} {DIVISION} teams, records for {matched}")
            time.sleep(delay)


def _records(page, code: str, raw_dir, n_players: int) -> dict[str, tuple[int, int, int]]:
    """Final records for the division's pod: the pod with the most players, every round."""
    index = page(f"/pairings/{code}", raw_dir / code / "pairings" / "index.html.gz")
    rounds: dict[int, int] = {}
    for pod, rnd in re.findall(rf"/pairings/{re.escape(code)}\?pod=(\d+)&(?:amp;)?rnd=(\d+)", index):
        rounds[int(pod)] = max(rounds.get(int(pod), 0), int(rnd))
    best: dict[str, tuple[int, int, int]] = {}
    for pod in sorted(rounds, key=lambda p: -len(parse_round(page(f"/pairings/{code}?pod={p}&rnd=1",
                                                                  raw_dir / code / "pairings" / f"p{p}r1.html.gz")))):
        for rnd in range(1, rounds[pod] + 1):
            for name, rec in parse_round(page(f"/pairings/{code}?pod={pod}&rnd={rnd}",
                                              raw_dir / code / "pairings" / f"p{pod}r{rnd}.html.gz")).items():
                if sum(rec) >= sum(best.get(name, (0, 0, 0))):
                    best[name] = rec
        break  # the largest pod is the Masters division
    return best
