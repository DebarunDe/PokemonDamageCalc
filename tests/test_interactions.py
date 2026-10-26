"""Interaction graph in analysis/interactions.py: who provides which condition, and what it does for a set."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import doubles_sim as sim  # noqa: E402
import interactions as ix  # noqa: E402


def test_providers_come_from_abilities_and_moves():
    assert ix.provides({"Electric Surge", "Fake Out", "Volt Tackle"}) == {"Electric Terrain", "Fake Out"}
    assert ix.provides({"Drizzle", "Tailwind", "Wide Guard"}) == {"Rain", "Tailwind"}
    assert "priority block" in ix.provides({"Psychic Surge", "Follow Me"})
    assert ix.provides({"Close Combat"}) == set()


class FakeBench:
    """One opponent instead of the real usage table, so the test needs no database."""

    def __init__(self, opp):
        self.opp, self.total = [(opp, 1)], 1


def test_seed_and_unburden_need_the_matching_terrain():
    sneasler = sim.Spec("Sneasler", "Unburden", "Electric Seed", "Adamant", {"atk": 32, "spe": 32},
                        ["Close Combat", "Dire Claw", "Swords Dance", "Protect"])
    foe = sim.Spec("Garchomp", "Rough Skin", None, "Jolly", {"atk": 32, "spe": 32}, ["Earthquake", "Dragon Claw"])
    bench = FakeBench(foe)
    base = ix.measure(sneasler, bench)
    electric = ix.condition_effect(sneasler, "Electric Terrain", bench, base)
    assert electric["speed"] >= 0 and electric["survive"] >= 0
    grassy = ix.condition_effect(sneasler, "Grassy Terrain", bench, base)
    assert grassy is None or grassy["speed"] == 0  # the wrong terrain doesn't trigger the Seed


def test_rain_helps_swift_swim_and_hurts_fire():
    foe = sim.Spec("Garchomp", "Rough Skin", None, "Jolly", {"spe": 32}, ["Dragon Claw"])
    bench = FakeBench(foe)
    swimmer = sim.Spec("Basculegion-F", "Swift Swim", "Focus Sash", "Modest", {"spa": 32, "spe": 4},
                       ["Muddy Water", "Shadow Ball"])
    eff = ix.condition_effect(swimmer, "Rain", bench, ix.measure(swimmer, bench))
    assert eff["speed"] > 0
    fire = sim.Spec("Arcanine-Hisui", "Rock Head", None, "Jolly", {"atk": 32, "spe": 32}, ["Flare Blitz"])
    weak = sim.Spec("Kingambit", "Defiant", None, "Adamant", {"hp": 2}, ["Kowtow Cleave"])
    bench = FakeBench(weak)
    eff = ix.condition_effect(fire, "Rain", bench, ix.measure(fire, bench))
    assert eff is None or eff["ohko"] <= 0


@pytest.mark.parametrize("cond", ["Fake Out", "redirection"])
def test_setup_cover_only_matters_to_setup_sets(cond):
    foe = sim.Spec("Incineroar", "Intimidate", None, "Careful", {"hp": 32}, ["Flare Blitz"])
    bench = FakeBench(foe)
    plain = sim.Spec("Garchomp", "Rough Skin", None, "Jolly", {"atk": 32}, ["Dragon Claw", "Earthquake"])
    assert ix.condition_effect(plain, cond, bench, ix.measure(plain, bench)) is None
    dancer = sim.Spec("Kingambit", "Defiant", "Life Orb", "Adamant", {"atk": 32}, ["Kowtow Cleave", "Swords Dance"])
    eff = ix.condition_effect(dancer, cond, bench, ix.measure(dancer, bench))
    assert eff is not None and eff["ohko"] >= 0
