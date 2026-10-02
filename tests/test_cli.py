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


def test_doubles_flags(capsys):
    base = ["calc", "Garchomp", "Earthquake", "Incineroar", "--a-nature", "Jolly",
            "--a-sp", "32 Atk", "--d-sp", "32 HP / 2 Def", "--doubles"]
    assert main(base) == 0
    assert "138-164" in capsys.readouterr().out
    assert main([*base, "--single-target"]) == 0
    assert "186-218" in capsys.readouterr().out
    assert main([*base, "--helping-hand", "--friend-guard", "--aurora-veil"]) == 0
    out = capsys.readouterr().out
    assert "Helping Hand" in out and "Friend Guard" in out and "Aurora Veil" in out


def test_helping_hand_requires_doubles(capsys):
    assert main(["calc", "Garchomp", "Earthquake", "Incineroar", "--helping-hand"]) == 1
    assert "only applies in doubles" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags, expected",
    [
        (["--a-atk", "+2"], "+2 0 Atk Garchomp"),
        (["--a-atk", "-1"], "-1 0 Atk Garchomp"),
        (["--d-def", "+6"], "vs. +6 0 HP / 0 Def"),
        (["--a-boosts", "+1 Atk", "--d-def", "-2"], "+1 0 Atk Garchomp Earthquake vs. -2 0 HP"),
    ],
)
def test_physical_stage_flags(capsys, flags, expected):
    assert main(["calc", "Garchomp", "Earthquake", "Incineroar", *flags]) == 0
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("attack, defense", [("--a-spa", "--d-spd"), ("--a-spatk", "--d-spdef")])
def test_special_stage_flags_and_aliases(capsys, attack, defense):
    assert main(["calc", "Charizard", "Flamethrower", "Garchomp", attack, "+1", defense, "-2"]) == 0
    assert "+1 0 SpA Charizard Flamethrower vs. -2 0 HP / 0 SpD Garchomp" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["+7", "-7", "two"])
def test_stage_flags_reject_bad_values(value):
    with pytest.raises(SystemExit):
        main(["calc", "Garchomp", "Earthquake", "Incineroar", "--a-atk", value])


def test_conflicting_stage_flags(capsys):
    assert main(["calc", "Garchomp", "Earthquake", "Incineroar", "--a-boosts", "+1 Atk", "--a-atk", "+2"]) == 1
    assert "different atk stages" in capsys.readouterr().err
