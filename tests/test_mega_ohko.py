"""Rules used by analysis/mega_ohko.py to set up each Mega matchup."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import mega_ohko as m  # noqa: E402


def mega(name, ability, speed=100):
    return m.Mega(name=name, ability=ability, stone="", speed=speed, moves=())


def test_trace_copies_the_opponent():
    alakazam, zard = mega("Alakazam-Mega", "Trace"), mega("Charizard-Mega-Y", "Drought")
    assert m.effective_abilities(alakazam, zard) == ("Drought", "Drought")
    assert m.effective_abilities(zard, alakazam) == ("Drought", "Drought")
    # Trace cannot copy Trace.
    meowstic = mega("Meowstic-M-Mega", "Trace")
    assert m.effective_abilities(alakazam, meowstic) == ("Trace", "Trace")


def test_slower_megas_weather_wins():
    zard, ttar = mega("Charizard-Mega-Y", "Drought", 120), mega("Tyranitar-Mega", "Sand Stream", 91)
    assert m.resolve_condition(zard, ttar, "Drought", "Sand Stream", m.WEATHER_ABILITIES) == ("Sand", "contested")
    assert m.resolve_condition(ttar, zard, "Sand Stream", "Drought", m.WEATHER_ABILITIES) == ("Sand", "contested")


def test_single_weather_setter_applies_either_way():
    zard, other = mega("Charizard-Mega-Y", "Drought"), mega("Garchomp-Mega", "Sand Force")
    assert m.resolve_condition(zard, other, "Drought", "Sand Force", m.WEATHER_ABILITIES) == ("Sun", "")
    assert m.resolve_condition(other, zard, "Sand Force", "Drought", m.WEATHER_ABILITIES) == ("Sun", "")
    assert m.resolve_condition(other, other, "Sand Force", "Sand Force", m.WEATHER_ABILITIES) == (None, "")


def test_intimidate_reactions():
    assert m.intimidate_stage("Tough Claws") == -1
    assert m.intimidate_stage("Inner Focus") == 0
    assert m.intimidate_stage("Scrappy") == 0
    assert m.intimidate_stage("Contrary") == 1
    assert m.intimidate_stage("Defiant") == 1


def test_accuracy_adjustments():
    assert m.accuracy("Zap Cannon", 50, None, no_guard=True) == 100
    assert m.accuracy("Aerial Ace", True, None, no_guard=False) == 100
    assert m.accuracy("Blizzard", 70, "Snow", no_guard=False) == 100
    assert m.accuracy("Hurricane", 70, "Sun", no_guard=False) == 50
    assert m.accuracy("Stone Edge", 80, "Sand", no_guard=False) == 80
