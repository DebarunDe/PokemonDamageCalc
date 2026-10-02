"""Checks stat stages against an independent, hand-written damage formula.

Every attacker stage from -6 to +6 is paired with every defender stage, with
and without crits and the doubles spread penalty, and all 16 rolls must match.
"""

from math import floor

import pytest

from champcalc import Calculator, Field, PokemonSet

STAGES = range(-6, 7)


@pytest.fixture(scope="module")
def calc() -> Calculator:
    return Calculator()


def apply_stage(stat: int, stage: int) -> int:
    # +n multiplies by (2+n)/2 and -n by 2/(2+n), rounded down.
    return floor(stat * (2 + stage) / 2) if stage >= 0 else floor(stat * 2 / (2 - stage))


def poke_round(x: float) -> int:
    # The games round halves down.
    return floor(x) + 1 if x % 1 > 0.5 else floor(x)


def champions_stat(base: int, sp: int, nature: float = 1.0) -> int:
    return floor((base + sp + 20) * nature)


def expected_rolls(attack, defense, power, effectiveness, crit, spread):
    """Gen 9 damage formula at level 50 for a STAB move with no other modifiers."""
    damage = floor(floor(floor(2 * 50 / 5 + 2) * power * attack / defense) / 50) + 2
    if spread:
        damage = poke_round(damage * 3072 / 4096)
    if crit:
        damage = floor(damage * 1.5)
    rolls = []
    for roll in range(85, 101):
        d = floor(damage * roll / 100)
        d = poke_round(d * 6144 / 4096)
        rolls.append(floor(d * effectiveness))
    return rolls


SCENARIOS = {
    # Physical: Jolly 32 Atk Garchomp Earthquake (100 BP, STAB, 2x) vs 32 HP / 2 Def Incineroar.
    "physical": dict(
        attacker=lambda: PokemonSet("Garchomp", nature="Jolly", sp={"atk": 32}),
        defender=lambda: PokemonSet("Incineroar", sp={"hp": 32, "def": 2}),
        move="Earthquake", attack_stat="atk", defense_stat="def",
        attack=champions_stat(130, 32), defense=champions_stat(90, 2), power=100, effectiveness=2,
    ),
    # Special: Modest 32 SpA Charizard Flamethrower (90 BP, STAB, 2x) vs 20 SpD Metagross.
    "special": dict(
        attacker=lambda: PokemonSet("Charizard", nature="Modest", sp={"spa": 32}),
        defender=lambda: PokemonSet("Metagross", sp={"spd": 20}),
        move="Flamethrower", attack_stat="spa", defense_stat="spd",
        attack=champions_stat(109, 32, 1.1), defense=champions_stat(90, 20), power=90, effectiveness=2,
    ),
}


@pytest.mark.parametrize(
    "scenario, crit, spread",
    [
        ("physical", False, False),
        ("physical", True, False),
        ("physical", False, True),
        ("physical", True, True),
        ("special", False, False),
        ("special", True, False),
    ],
)
def test_every_stage_combination_matches_formula(calc, scenario, crit, spread):
    s = SCENARIOS[scenario]
    mismatches = []
    for a in STAGES:
        for d in STAGES:
            attacker = s["attacker"]()
            defender = s["defender"]()
            attacker.boosts = {s["attack_stat"]: a}
            defender.boosts = {s["defense_stat"]: d}
            # Crits ignore the attacker's drops and the defender's boosts.
            attack = apply_stage(s["attack"], max(a, 0) if crit else a)
            defense = apply_stage(s["defense"], min(d, 0) if crit else d)
            expected = expected_rolls(attack, defense, s["power"], s["effectiveness"], crit, spread)
            result = calc.calculate(attacker, defender, s["move"], Field(doubles=spread), is_crit=crit)
            if result.rolls != expected:
                mismatches.append(f"{a:+d} vs {d:+d}: got {result.rolls}, expected {expected}")
    assert not mismatches, "\n".join(mismatches)
