"""Mechanics of the doubles battle simulator in analysis/doubles_sim.py."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import doubles_sim as sim  # noqa: E402


def spec(name, moves, ability=None, item=None, nature="Hardy", sp=None, pre_ability=None):
    return sim.Spec(name, ability, item, nature, sp or {}, moves, pre_ability=pre_ability)


def new_turn(order=()):
    return {"redirect": {0: None, 1: None}, "wide": {0: False, 1: False}, "quick": {0: False, 1: False},
            "helping": set(), "moved": set(), "order": list(order)}


@pytest.fixture
def garchomp():
    return spec("Garchomp", ["Earthquake", "Dragon Claw", "Protect"], ability="Rough Skin", nature="Jolly",
                sp={"atk": 32, "spe": 32})


def test_fake_out_flinches_unless_inner_focus(garchomp):
    incineroar = spec("Incineroar", ["Fake Out"], ability="Intimidate")
    battle = sim.Battle([incineroar, garchomp], [garchomp, spec("Dragonite", ["Dragon Claw"], ability="Inner Focus")],
                        random.Random(0))
    fake_out_user, target, dragonite = battle.sides[0][0], battle.sides[1][0], battle.sides[1][1]
    battle.use(fake_out_user, "Fake Out", [target], new_turn())
    assert target.flinched
    battle.use(fake_out_user, "Fake Out", [dragonite], new_turn())
    assert not dragonite.flinched
    fake_out_user.turns_active = 1  # only works on the first turn out
    target.flinched = False
    battle.use(fake_out_user, "Fake Out", [target], new_turn())
    assert not target.flinched


def test_protect_blocks_and_repeats_can_fail(garchomp):
    battle = sim.Battle([garchomp], [garchomp], random.Random(0))
    a, b = battle.sides[0][0], battle.sides[1][0]
    battle.use(b, "Protect", [], new_turn())
    assert b.protected
    battle.use(a, "Dragon Claw", [b], new_turn())
    assert b.hp == 100
    fails = 0
    for seed in range(60):
        battle.rng = random.Random(seed)
        b.last_protect = True
        battle.use(b, "Protect", [], new_turn())
        fails += not b.protected
    assert 25 < fails < 55  # a second Protect in a row works 1 time in 3


def test_intimidate_on_entry_and_clear_body_before_mega(garchomp):
    incineroar = spec("Incineroar", ["Flare Blitz"], ability="Intimidate")
    metagross = spec("Metagross-Mega", ["Iron Head"], ability="Tough Claws", item="Metagrossite",
                     pre_ability="Clear Body")
    battle = sim.Battle([garchomp, metagross], [incineroar], random.Random(0))
    battle.enter(battle.sides[1][0])
    assert battle.sides[0][0].boosts["atk"] == -1
    assert battle.sides[0][1].boosts["atk"] == 0


def test_focus_sash_survives_a_ko_from_full(garchomp):
    arcanine = spec("Arcanine-Hisui", ["Head Smash"], ability="Rock Head", item="Focus Sash")
    battle = sim.Battle([garchomp], [arcanine], random.Random(0))
    a, b = battle.sides[0][0], battle.sides[1][0]
    battle.use(a, "Earthquake", [], new_turn())  # Ground is 4x on Fire/Rock
    assert 0 < b.hp <= 1 and b.item is None
    battle.use(a, "Earthquake", [], new_turn())
    assert b.fainted


def test_earthquake_hits_the_ally_but_not_a_flier(garchomp):
    ally = spec("Incineroar", ["Flare Blitz"], ability="Intimidate")
    flier = spec("Dragonite", ["Dragon Claw"], ability="Inner Focus")
    battle = sim.Battle([garchomp, ally], [garchomp, flier], random.Random(1))
    battle.use(battle.sides[0][0], "Earthquake", [], new_turn())
    assert battle.sides[0][1].hp < 100  # the ally
    assert battle.sides[1][0].hp < 100
    assert battle.sides[1][1].hp == 100  # Flying type


def test_electro_shot_charges_outside_rain():
    archaludon = spec("Archaludon", ["Electro Shot"], ability="Stamina", nature="Modest", sp={"spa": 32})
    target = spec("Incineroar", ["Flare Blitz"], ability="Intimidate")
    battle = sim.Battle([archaludon], [target], random.Random(0))
    a, b = battle.sides[0][0], battle.sides[1][0]
    battle.use(a, "Electro Shot", [b], new_turn())
    assert a.charging and a.boosts["spa"] == 1 and b.hp == 100
    battle.weather = "Rain"
    a.charging = None
    battle.use(a, "Electro Shot", [b], new_turn())
    assert a.boosts["spa"] == 2 and b.hp < 100


def test_battle_runs_to_a_result(garchomp):
    team_a = [garchomp, spec("Incineroar", ["Fake Out", "Flare Blitz", "Parting Shot"], ability="Intimidate"),
              spec("Metagross-Mega", ["Iron Head", "Psychic Fangs", "Protect"], ability="Tough Claws",
                   item="Metagrossite", nature="Adamant", sp={"atk": 32, "spe": 32}, pre_ability="Clear Body")]
    team_b = [spec("Rillaboom", ["Fake Out", "Wood Hammer", "Grassy Glide"], ability="Grassy Surge"),
              spec("Gholdengo", ["Make It Rain", "Shadow Ball", "Protect"], ability="Good as Gold", nature="Modest",
                   sp={"spa": 32, "spe": 32}),
              spec("Sneasler", ["Close Combat", "Dire Claw", "Fake Out"], ability="Unburden", nature="Jolly")]
    for seed in range(5):
        battle = sim.Battle(team_a, team_b, random.Random(seed), log=True)
        assert battle.run() in (0.0, 0.5, 1.0)
        assert battle.lines and battle.turn <= battle.max_turns


def test_eruption_weakens_with_the_users_hp(garchomp):
    torkoal = spec("Torkoal", ["Eruption"], ability="Drought", nature="Quiet", sp={"hp": 32, "spa": 32})
    battle = sim.Battle([torkoal], [garchomp], random.Random(0))
    a, b = battle.sides[0][0], battle.sides[1][0]
    full = battle.damage_range(a, b, "Eruption")
    a.hp = 50.0
    half = battle.damage_range(a, b, "Eruption")
    assert half[1] == pytest.approx(full[1] / 2)


def test_weather_changes_hurricane_accuracy():
    assert sim.accuracy("Hurricane") == 0.7
    assert sim.accuracy("Hurricane", "Rain") == 1.0
    assert sim.accuracy("Hurricane", "Sun") == 0.5
    assert sim.accuracy("Blizzard", "Snow") == 1.0


def test_choice_scarf_locks_the_first_move(garchomp):
    scarf = spec("Garchomp", ["Dragon Claw", "Earthquake"], ability="Rough Skin", item="Choice Scarf", nature="Jolly",
                 sp={"atk": 32, "spe": 32})
    target = spec("Incineroar", ["Flare Blitz"], ability="Intimidate")
    battle = sim.Battle([scarf], [target], random.Random(0))
    a = battle.sides[0][0]
    battle.use(a, "Dragon Claw", [battle.sides[1][0]], new_turn())
    assert a.locked == "Dragon Claw"
    for _ in range(5):
        assert battle.choose(a, {})[0] == "Dragon Claw"  # Earthquake would hit harder, but it's locked in


def test_growth_doubles_in_sun_and_solar_beam_needs_it(garchomp):
    venusaur = spec("Venusaur", ["Growth", "Solar Beam"], ability="Chlorophyll", nature="Modest", sp={"spa": 32})
    battle = sim.Battle([venusaur], [garchomp], random.Random(0))
    v, g = battle.sides[0][0], battle.sides[1][0]
    battle.use(v, "Solar Beam", [g], new_turn())
    assert g.hp == 100  # no charge turn modelled: it just fails outside sun
    battle.weather = "Sun"
    battle.use(v, "Growth", [], new_turn())
    assert v.boosts["spa"] == 2 and v.boosts["atk"] == 2
    battle.use(v, "Solar Beam", [g], new_turn())
    assert g.hp < 100


def test_sleep_powder_misses_a_quarter_of_the_time(garchomp):
    venusaur = spec("Venusaur", ["Sleep Powder"], ability="Chlorophyll")
    slept = 0
    for seed in range(200):
        battle = sim.Battle([venusaur], [garchomp], random.Random(seed))
        battle.use(battle.sides[0][0], "Sleep Powder", [battle.sides[1][0]], new_turn())
        slept += battle.sides[1][0].status == "slp"
    assert 125 < slept < 175


def test_unknown_species_gets_no_set():
    if not sim.DB.exists():
        pytest.skip("needs analysis/data/meta.sqlite")
    assert sim.real_spec("Not A Pokemon") is None


def test_bring_keeps_at_most_one_mega(garchomp):
    megas = [spec(n, ["Protect"]) for n in ("Charizard-Mega-Y", "Metagross-Mega", "Delphox-Mega")]
    others = [garchomp, spec("Incineroar", ["Fake Out"]), spec("Rillaboom", ["Fake Out"])]
    for seed in range(20):
        chosen = sim.bring(megas + others, 4, random.Random(seed), fixed=[megas[0]])
        assert chosen[0] is megas[0] and len(chosen) == 4
        assert sum("-Mega" in s.name for s in chosen) == 1


def test_supreme_overlord_grows_with_fainted_allies(garchomp):
    kingambit = spec("Kingambit", ["Kowtow Cleave"], ability="Supreme Overlord", nature="Adamant", sp={"atk": 32})
    battle = sim.Battle([kingambit], [garchomp], random.Random(0))
    a, b = battle.sides[0][0], battle.sides[1][0]
    base = battle.damage_range(a, b, "Kowtow Cleave")[1]
    battle.fainted_count[0] = 2
    assert battle.damage_range(a, b, "Kowtow Cleave")[1] == pytest.approx(base * 1.2)


def test_white_herb_triggers_unburden(garchomp):
    sneasler = spec("Sneasler", ["Close Combat"], ability="Unburden", item="White Herb", nature="Jolly",
                    sp={"atk": 32, "spe": 32})
    battle = sim.Battle([sneasler], [garchomp], random.Random(0))
    a = battle.sides[0][0]
    before = battle.speed(a)
    battle.use(a, "Close Combat", [battle.sides[1][0]], new_turn())
    assert a.boosts["def"] == 0 and a.item is None  # White Herb undid the drops and was used up
    assert battle.speed(a) == pytest.approx(2 * before)


def test_hospitality_heals_the_partner_on_entry(garchomp):
    sinistcha = spec("Sinistcha", ["Matcha Gotcha"], ability="Hospitality")
    battle = sim.Battle([garchomp, sinistcha], [garchomp], random.Random(0))
    battle.sides[0][0].hp = 40.0
    battle.enter(battle.sides[0][1])
    assert battle.sides[0][0].hp == 65.0


def test_rocky_helmet_hurts_contact_attackers():
    incineroar = spec("Incineroar", ["Fake Out", "Flare Blitz"], ability="Intimidate")
    helmet = spec("Indeedee-F", ["Psychic"], ability="Psychic Surge", item="Rocky Helmet")
    battle = sim.Battle([incineroar], [helmet], random.Random(0))
    a, h = battle.sides[0][0], battle.sides[1][0]
    battle.use(a, "Fake Out", [h], new_turn())  # Psychic Terrain isn't up, so it connects
    assert h.flinched
    assert a.hp == pytest.approx(100 - 100 / 6)


def test_resist_berry_is_used_up_and_lum_berry_cures_once():
    kingambit = spec("Kingambit", ["Kowtow Cleave"], ability="Defiant", item="Chople Berry")
    lucario = spec("Lucario-Mega", ["Close Combat"], nature="Adamant", sp={"atk": 32})
    battle = sim.Battle([lucario], [kingambit], random.Random(0))
    a, k = battle.sides[0][0], battle.sides[1][0]
    halved = battle.damage_range(a, k, "Close Combat")
    battle.use(a, "Close Combat", [k], new_turn())
    assert k.item is None
    assert battle.damage_range(a, k, "Close Combat")[0] > halved[0]
    lum = sim.Mon(spec("Garchomp", ["Earthquake"], item="Lum Berry"), 0)
    lum.set_status("brn")
    assert lum.status == "" and lum.item is None
    lum.set_status("brn")
    assert lum.status == "brn"
