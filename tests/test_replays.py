"""Replay log parsing in analysis/replays.py (no network, no database)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import replays  # noqa: E402

LOG = "\n".join([
    "|player|p1|Alice|1|1350",
    "|player|p2|Bob|2|1290",
    "|poke|p1|Raichu, L50, F|",
    "|poke|p1|Sneasler, L50, M|",
    "|poke|p1|Pelipper, L50, M|",
    "|poke|p2|Incineroar, L50, M|",
    "|poke|p2|Kingambit, L50, M|",
    "|start",
    "|switch|p1a: Sparky|Raichu, L50, F|100/100",
    "|switch|p1b: Sneasler|Sneasler, L50, M|100/100",
    "|switch|p2a: Incineroar|Incineroar, L50, M|100/100",
    "|switch|p2b: Kingambit|Kingambit, L50, M|100/100",
    "|-fieldstart|move: Electric Terrain|[from] ability: Electric Surge|[of] p1a: Sparky",
    "|-enditem|p1b: Sneasler|Electric Seed",
    "|turn|1",
    "|detailschange|p1a: Sparky|Raichu-Mega-X, L50, F",
    "|-mega|p1a: Sparky|Raichu|Raichunite X",
    "|move|p1a: Sparky|Fake Out|p2b: Kingambit",
    "|move|p1b: Sneasler|Close Combat|p2b: Kingambit",
    "|-sidestart|p2: Bob|move: Tailwind",
    "|switch|p1b: Pelipper|Pelipper, L50, M|100/100",
    "|turn|2",
    "|win|Alice",
])


def test_parse_log():
    g = replays.parse_log(LOG)
    p1, p2 = g["p1"], g["p2"]
    assert (p1["player"], p1["rating"], p1["won"], p2["won"], p1["turns"]) == ("Alice", 1350, 1, 0, 2)
    assert p1["preview"] == ["Raichu", "Sneasler", "Pelipper"]
    assert p1["leads"] == ["Raichu", "Sneasler"] and p1["brought"] == ["Raichu", "Sneasler", "Pelipper"]
    assert p1["mega"] == ["Raichu-Mega-X"]
    assert p1["moves"] == {"Raichu": ["Fake Out"], "Sneasler": ["Close Combat"]}  # nicknames resolved
    assert p1["items"] == {"Sneasler": "Electric Seed"}
    assert p1["fields"] == ["Electric Terrain"] and p2["fields"] == ["Tailwind"]


def test_unfinished_game_is_skipped():
    assert replays.parse_log(LOG.replace("|win|Alice", "")) is None


def test_expected():
    assert replays.expected(None, 1300) == 0.5
    assert abs(replays.expected(1400, 1000) - 1 / (1 + 10 ** -1)) < 1e-12
