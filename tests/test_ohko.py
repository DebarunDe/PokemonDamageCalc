"""Rules used by analysis/ohko.py to set up each matchup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import ohko  # noqa: E402


def test_trace_copies_the_opponent():
    assert ohko.effective_abilities("Trace", "Drought") == ("Drought", "Drought")
    assert ohko.effective_abilities("Drought", "Trace") == ("Drought", "Drought")
    # Trace cannot copy Trace, Disguise and the like.
    assert ohko.effective_abilities("Trace", "Trace") == ("Trace", "Trace")
    assert ohko.effective_abilities("Trace", "Disguise") == ("Trace", "Disguise")


def test_slower_pokemons_weather_wins():
    weather = ohko.WEATHER_ABILITIES
    # Charizard-Mega-Y (120 Speed) against Tyranitar-Mega (91): sand is set last.
    assert ohko.resolve_condition(120, 91, "Drought", "Sand Stream", weather) == ("Sand", "contested")
    assert ohko.resolve_condition(91, 120, "Sand Stream", "Drought", weather) == ("Sand", "contested")
    assert ohko.resolve_condition(100, 100, "Drought", "Drizzle", weather) == ("Sun", "contested (speed tie)")


def test_single_weather_setter_applies_either_way():
    weather = ohko.WEATHER_ABILITIES
    assert ohko.resolve_condition(100, 50, "Drought", "Sand Force", weather) == ("Sun", "")
    assert ohko.resolve_condition(100, 50, "Sand Force", "Drought", weather) == ("Sun", "")
    assert ohko.resolve_condition(100, 50, "Sand Force", "Levitate", weather) == (None, "")


def test_intimidate_reactions():
    assert ohko.intimidate_stage("Tough Claws") == -1
    assert ohko.intimidate_stage("Inner Focus") == 0
    assert ohko.intimidate_stage("Scrappy") == 0
    assert ohko.intimidate_stage("Contrary") == 1
    assert ohko.intimidate_stage("Defiant") == 1


def test_accuracy_adjustments():
    acc = ohko.accuracy
    assert acc("Zap Cannon", 50, "Special", None, "No Guard", "Levitate") == 100
    assert acc("Zap Cannon", 50, "Special", None, "Levitate", "No Guard") == 100
    assert acc("Aerial Ace", True, "Physical", None, "", "") == 100
    assert acc("Blizzard", 70, "Special", "Snow", "", "") == 100
    assert acc("Hurricane", 70, "Special", "Sun", "", "") == 50
    assert acc("Stone Edge", 80, "Physical", "Sand", "", "") == 80
    # Hustle only lowers physical moves; Compound Eyes caps at 100%.
    assert acc("Earthquake", 100, "Physical", None, "Hustle", "") == 80
    assert acc("Thunderbolt", 100, "Special", None, "Hustle", "") == 100
    assert acc("Stone Edge", 80, "Physical", None, "Compound Eyes", "") == 100
    # Every hit rolls accuracy, unless Skill Link makes it one check.
    assert acc("Triple Axel", 90, "Physical", None, "", "") == 72.9
    assert acc("Population Bomb", 90, "Physical", None, "Skill Link", "") == 90


def test_rivalry_assumes_the_worse_gender_pairing():
    # A target that can be either gender is assumed to be the opposite one.
    assert ohko.rivalry_genders(None, None) == ("M", "F")
    assert ohko.rivalry_genders("F", None) == ("F", "M")
    # A single-gender target is matched.
    assert ohko.rivalry_genders(None, "F") == ("F", "F")
    assert ohko.rivalry_genders("M", "F") == ("M", "F")
    # Genderless targets are unaffected.
    assert ohko.rivalry_genders(None, "N") == (None, "N")
