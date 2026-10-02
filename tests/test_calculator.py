import pytest

from champcalc import CalcError, Calculator, Field, PokemonSet


@pytest.fixture(scope="session")
def calc() -> Calculator:
    return Calculator()


def test_damage_matches_hand_calculation(calc):
    # Garchomp: Atk 130 + 32 SP + 20 = 182. Incineroar: HP 95 + 32 + 75 = 202, Def 90 + 2 + 20 = 112.
    # Base damage 73, STAB 1.5, Ground vs Fire/Dark 2x -> 186 (85% roll) to 218 (100% roll).
    result = calc.calculate(
        PokemonSet("Garchomp", nature="Jolly", sp={"atk": 32, "spe": 32}),
        PokemonSet("Incineroar", sp={"hp": 32, "def": 2}),
        "Earthquake",
    )
    assert result.attacker["stats"]["atk"] == 182
    assert result.defender["max_hp"] == 202
    assert (result.min, result.max) == (186, 218)
    assert len(result.rolls) == 16
    assert result.rolls[0] == 186 and result.rolls[-1] == 218
    assert result.max_percent == 107.9
    assert "OHKO" in result.ko.text
    assert result.warnings == []


def test_stat_formula_uses_stat_points_and_nature(calc):
    result = calc.calculate(
        PokemonSet("Garchomp", nature="Jolly", sp={"hp": 2, "spe": 32}),
        PokemonSet("Incineroar"),
        "Earthquake",
    )
    stats = result.attacker["stats"]
    assert stats["hp"] == 108 + 2 + 75
    assert stats["spe"] == int((102 + 32 + 20) * 1.1)
    assert stats["spa"] == int((80 + 20) * 0.9)


def test_field_and_modifiers_change_damage(calc):
    attacker = PokemonSet("Charizard", nature="Modest", sp={"spa": 32})
    defender = PokemonSet("Garchomp")
    plain = calc.calculate(attacker, defender, "Flamethrower")
    assert calc.calculate(attacker, defender, "Flamethrower", Field(weather="sun")).max > plain.max
    assert calc.calculate(attacker, defender, "Flamethrower", Field(light_screen=True)).max < plain.max
    assert calc.calculate(attacker, defender, "Flamethrower", is_crit=True).max > plain.max
    boosted = PokemonSet("Charizard", nature="Modest", sp={"spa": 32}, boosts={"spa": 2})
    assert calc.calculate(boosted, defender, "Flamethrower").max > plain.max


def test_immunity_gives_zero_damage(calc):
    result = calc.calculate(PokemonSet("Garchomp"), PokemonSet("Corviknight"), "Earthquake")
    assert (result.min, result.max) == (0, 0)
    assert result.ko is None


def test_mega_formes_share_base_movepool(calc):
    assert calc.learnset("Garchomp-Mega") == calc.learnset("Garchomp")
    assert "Earthquake" in calc.learnset("garchomp")


def test_learnset_and_ability_warnings(calc):
    result = calc.calculate(
        PokemonSet("Incineroar", ability="Levitate"), PokemonSet("Garchomp"), "Draco Meteor"
    )
    assert any("cannot learn Draco Meteor" in w for w in result.warnings)
    assert any("cannot have Levitate" in w for w in result.warnings)


def test_species_info_lists_all_abilities(calc):
    info = calc.species("incineroar")
    assert info["name"] == "Incineroar"
    assert info["types"] == ["Fire", "Dark"]
    assert info["abilities"] == ["Blaze", "Intimidate"]
    assert info["base_stats"]["atk"] == 115


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"species": "Notamon"}, "Unknown Pokemon"),
        ({"species": "Garchomp", "item": "Not An Item"}, "Unknown item"),
        ({"species": "Garchomp", "ability": "Not An Ability"}, "Unknown ability"),
        ({"species": "Garchomp", "nature": "Grumpy"}, "Unknown nature"),
    ],
)
def test_unknown_names_are_rejected(calc, kwargs, message):
    with pytest.raises(CalcError, match=message):
        calc.calculate(PokemonSet(**kwargs), PokemonSet("Incineroar"), "Earthquake")


def test_unknown_and_status_moves_are_rejected(calc):
    with pytest.raises(CalcError, match="Unknown move"):
        calc.calculate(PokemonSet("Garchomp"), PokemonSet("Incineroar"), "Not A Move")
    with pytest.raises(CalcError, match="status move"):
        calc.calculate(PokemonSet("Garchomp"), PokemonSet("Incineroar"), "Swords Dance")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sp": {"atk": 33}}, "between 0 and 32"),
        ({"sp": {"atk": 32, "spe": 32, "hp": 3}}, "maximum is 66"),
        ({"sp": {"speed": 4}}, "Unknown stat"),
        ({"boosts": {"atk": 7}}, "between -6 and"),
        ({"boosts": {"hp": 1}}, "Unknown stat"),
        ({"status": "sleepy"}, "Unknown status"),
        ({"cur_hp_percent": 0}, "Current HP"),
    ],
)
def test_invalid_sets_are_rejected(kwargs, message):
    with pytest.raises(CalcError, match=message):
        PokemonSet("Garchomp", **kwargs)


def test_every_learnset_entry_is_a_known_move(calc):
    known = set(calc.all_moves)
    for species in calc.all_species:
        moves = calc.learnset(species)
        assert moves, f"{species} has no movepool"
        assert set(moves) <= known, species


GARCHOMP = PokemonSet("Garchomp", nature="Jolly", sp={"atk": 32, "spe": 32})
INCINEROAR = PokemonSet("Incineroar", sp={"hp": 32, "def": 2})


def test_spread_moves_take_three_quarters_damage_in_doubles(calc):
    singles = calc.calculate(GARCHOMP, INCINEROAR, "Earthquake")
    doubles = calc.calculate(GARCHOMP, INCINEROAR, "Earthquake", Field(doubles=True))
    # 73 base damage * 0.75 spread = 55 -> 46 (85% roll) to 55, then STAB and 2x.
    assert (doubles.min, doubles.max) == (138, 164)
    assert doubles.move["spread"] is True
    assert singles.move["spread"] is False


def test_spread_move_with_one_target_left(calc):
    result = calc.calculate(GARCHOMP, INCINEROAR, "Earthquake", Field(doubles=True), spread=False)
    assert (result.min, result.max) == (186, 218)
    assert result.move["spread"] is False


def test_single_target_moves_are_unaffected_by_doubles(calc):
    singles = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw")
    doubles = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True))
    assert (singles.min, singles.max) == (doubles.min, doubles.max)
    assert doubles.move["spread"] is False


def test_spread_true_needs_a_spread_move_in_doubles(calc):
    with pytest.raises(CalcError, match="only hits one target"):
        calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True), spread=True)
    with pytest.raises(CalcError, match="only applies in doubles"):
        calc.calculate(GARCHOMP, INCINEROAR, "Earthquake", spread=True)


def test_ally_effects(calc):
    base = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True))
    helped = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True, helping_hand=True))
    guarded = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True, friend_guard=True))
    assert helped.max > base.max and "Helping Hand" in helped.description
    assert guarded.max < base.max and "Friend Guard" in guarded.description


def test_screens_are_weaker_in_doubles(calc):
    # Screens cut damage to 2732/4096 in doubles instead of half.
    singles = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(reflect=True))
    doubles = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True, reflect=True))
    veil = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw", Field(doubles=True, aurora_veil=True))
    assert doubles.max > singles.max
    assert (veil.min, veil.max) == (doubles.min, doubles.max)


@pytest.mark.parametrize("effect", ["helping_hand", "friend_guard"])
def test_ally_effects_need_doubles(effect):
    with pytest.raises(CalcError, match="only applies in doubles"):
        Field(**{effect: True})


def test_stat_stages_scale_damage(calc):
    plain = calc.calculate(GARCHOMP, INCINEROAR, "Dragon Claw")
    boosted = calc.calculate(
        PokemonSet("Garchomp", nature="Jolly", sp={"atk": 32, "spe": 32}, boosts={"atk": 2}),
        PokemonSet("Incineroar", sp={"hp": 32, "def": 2}, boosts={"def": 2}),
        "Dragon Claw",
    )
    assert (boosted.min, boosted.max) == (plain.min, plain.max)
    assert boosted.description.startswith("+2 ")


def test_move_info(calc):
    assert calc.move("earthquake") == {
        "name": "Earthquake", "type": "Ground", "category": "Physical",
        "base_power": 100, "priority": 0, "spread": True, "accuracy": 100,
        "offensive_stat": "atk", "defensive_stat": "def", "uses_target_attack": False, "contact": False,
        "multi_hit": False,
    }
    assert calc.move("Aurora Veil")["category"] == "Status"
    assert calc.move("Aqua Jet")["priority"] == 1
    # Weight-based moves have no fixed power.
    assert calc.move("Low Kick")["base_power"] == 0


def test_move_stats_and_accuracy(calc):
    assert calc.move("Psyshock")["defensive_stat"] == "def"
    assert calc.move("Body Press")["offensive_stat"] == "def"
    assert calc.move("Foul Play")["uses_target_attack"] is True
    assert calc.move("Close Combat")["contact"] is True
    assert calc.move("Aerial Ace")["accuracy"] is True
    assert calc.move("Zap Cannon")["accuracy"] == 50
    # Champions raises Crabhammer from 90% to 95%.
    assert calc.move("Crabhammer")["accuracy"] == 95


def test_species_mega_stone(calc):
    assert calc.species("Charizard-Mega-Y")["mega_stone"] == "Charizardite Y"
    assert calc.species("Charizard")["mega_stone"] is None


def test_calculate_many_matches_calculate(calc):
    attacker = PokemonSet("Gardevoir-Mega", nature="Modest", sp={"spa": 32}, item="Gardevoirite")
    defender = PokemonSet("Garchomp-Mega", sp={"hp": 32}, item="Garchompite")
    _, info, results = calc.calculate_many(attacker, defender, ["Hyper Voice", "Moonblast", "Psyshock"])
    assert info["max_hp"] == 108 + 32 + 75
    assert [r.move for r in results] == ["Hyper Voice", "Moonblast", "Psyshock"]
    # Pixilate turns Hyper Voice Fairy-type.
    assert results[0].type == "Fairy"
    for r in results:
        single = calc.calculate(attacker, defender, r.move)
        assert (r.min, r.max, r.max_percent) == (single.min, single.max, single.max_percent)


def test_calculate_many_rejects_status_moves(calc):
    with pytest.raises(CalcError, match="status move"):
        calc.calculate_many(PokemonSet("Garchomp"), PokemonSet("Incineroar"), ["Earthquake", "Swords Dance"])


def test_calculate_batch_matches_calculate(calc):
    moves = ["Earthquake", "Dragon Claw", "Stone Edge"]
    jobs = [
        (GARCHOMP, INCINEROAR, None, [0, 1]),
        (GARCHOMP, INCINEROAR, Field(doubles=True), [0, 2]),
    ]
    (hp1, dmg1), (hp2, dmg2) = calc.calculate_batch(moves, jobs)
    assert hp1 == hp2 == 202
    assert dmg1[0] == (186, 218) and dmg2[0] == (138, 164)
    single = calc.calculate(GARCHOMP, INCINEROAR, "Stone Edge", Field(doubles=True))
    assert dmg2[1] == (single.min, single.max)


def test_legal_species(calc):
    legal = calc.legal_species
    assert len(legal) == len(set(legal)) > 300
    assert {"Incineroar", "Garchomp-Mega", "Meowstic-F", "Aegislash-Both"} <= set(legal)
    # Battle-only formes and cosmetic duplicates are left out.
    assert not {"Aegislash-Blade", "Castform-Rainy", "Mimikyu-Busted", "Vivillon-Fancy"} & set(legal)


def test_multi_hit_flag(calc):
    assert calc.move("Bullet Seed")["multi_hit"] is True
    assert calc.move("Dual Wingbeat")["multi_hit"] is True
    assert calc.move("Earthquake")["multi_hit"] is False


def test_rivalry_depends_on_gender(calc):
    def hit(attacker_gender, defender_gender):
        return calc.calculate(
            PokemonSet("Pyroar", ability="Rivalry", gender=attacker_gender),
            PokemonSet("Incineroar", gender=defender_gender), "Flamethrower",
        ).max
    neutral = calc.calculate(PokemonSet("Pyroar", ability="Unnerve"), PokemonSet("Incineroar"), "Flamethrower").max
    assert hit("M", "M") > neutral > hit("M", "F")
    with pytest.raises(CalcError, match="Gender"):
        PokemonSet("Pyroar", gender="X")


def test_regulations():
    current = Calculator()
    assert current.regulation == "M-C"
    assert set(current.regulations) >= {"M-A", "M-B", "M-C"}
    sizes = {r: len(Calculator(r).legal_species) for r in ("M-A", "M-B", "M-C")}
    assert sizes["M-A"] < sizes["M-B"] < sizes["M-C"]
    with pytest.raises(CalcError, match="Unknown regulation"):
        Calculator("M-Z")
