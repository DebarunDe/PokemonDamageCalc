"""Emerging-set detector in analysis/emerging.py: feature keys, parents and lift."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import emerging as em  # noqa: E402


def test_team_features_and_parents():
    members = [("Sneasler", "Electric Seed", "Unburden", ["Close Combat", "Swords Dance"]),
               ("Raichu-Mega-X", "Raichunite X", "Lightning Rod", ["Fake Out"])]
    keys, parent = em.team_features(members, universal=frozenset({"Sneasler|ability:Unburden"}))
    assert "Sneasler|item:Electric Seed" in keys and parent["Sneasler|item:Electric Seed"] == "Sneasler"
    assert parent["Sneasler + Raichu-Mega-X"] == "Sneasler"
    assert parent["Sneasler|move:Swords Dance + Raichu-Mega-X"] == "Sneasler|move:Swords Dance"
    assert "Sneasler|ability:Unburden" not in keys  # universal choices are part of the species


def test_lift_compares_with_the_rest_of_the_parent():
    feature, parent = em.Agg(), em.Agg()
    for i in range(40):  # the choice: 60% wins against an even skill expectation
        feature.add(10, 1.0, f"p{i}", f"e{i % 8}")
        parent.add(10, 1.0, f"p{i}", f"e{i % 8}")
    for i in range(40, 200):  # the rest of the species: even
        parent.add(10, 0.0, f"p{i}", f"e{i % 8}")
    s = em.lift(feature, parent)
    assert abs(s["lift"] - 10.0) < 1e-9 and s["z"] > 3
    assert all(h is not None and h > 0 for h in s["halves"])
