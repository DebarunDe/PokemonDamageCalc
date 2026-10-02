import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import model  # noqa: E402
from meta_matchups import parse_spread  # noqa: E402


def test_key_merges_spellings():
    assert model.key("Sirfetch'd") == model.key("Sirfetch’d") == "sirfetchd"
    assert model.key("Aegislash-Both") == model.key("Aegislash") == "aegislash"
    assert model.key("Charizard-Mega-Y") == "charizardmegay"


def test_parse_spread():
    assert parse_spread("Adamant:32/32/0/0/2/0") == ("Adamant", {"hp": 32, "atk": 32, "spd": 2})
    assert parse_spread("Jolly:32/32/32/0/0/0") is None  # 96 SP is over the 66 limit
    assert parse_spread("Jolly:33/0/0/0/0/0") is None
    assert parse_spread("garbage") is None
