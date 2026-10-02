import json

import pytest

from champcalc.cli import main, parse_stat_spread
from champcalc.models import CalcError


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {}),
        ("32 Atk / 32 Spe / 2 HP", {"atk": 32, "spe": 32, "hp": 2}),
        ("atk=32,spe=32", {"atk": 32, "spe": 32}),
        ("+1 Atk / -1 Def", {"atk": 1, "def": -1}),
        ("32 SpA", {"spa": 32}),
        ("Sp. Def=10", {"spd": 10}),
    ],
)
def test_parse_stat_spread(text, expected):
    assert parse_stat_spread(text) == expected


def test_parse_stat_spread_rejects_garbage():
    with pytest.raises(CalcError):
        parse_stat_spread("lots of attack")


def test_calc_command(capsys):
    code = main(["calc", "Garchomp", "Earthquake", "Incineroar", "--a-nature", "Jolly",
                 "--a-sp", "32 Atk / 32 Spe", "--d-sp", "32 HP / 2 Def", "--rolls"])
    out = capsys.readouterr().out
    assert code == 0
    assert "186-218 (92 - 107.9%)" in out
    assert out.splitlines()[1].startswith("Rolls: 186, 186")


def test_calc_command_json(capsys):
    assert main(["calc", "Garchomp", "Earthquake", "Incineroar", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["move"]["name"] == "Earthquake"
    assert len(data["rolls"]) == 16


def test_errors_are_reported_without_traceback(capsys):
    assert main(["calc", "Garchomp", "Earthquake", "Incineroar", "--a-item", "Choice Band"]) == 1
    assert "Unknown item in Pokemon Champions: 'Choice Band'" in capsys.readouterr().err


def test_moves_and_info_commands(capsys):
    assert main(["moves", "garchomp"]) == 0
    assert "  Earthquake" in capsys.readouterr().out
    assert main(["info", "incineroar"]) == 0
    assert "Abilities: Blaze, Intimidate" in capsys.readouterr().out
