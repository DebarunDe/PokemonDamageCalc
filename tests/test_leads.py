"""Lead model in analysis/leads.py: the game solver, turn-1 resolution and the replay faint parser."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import leads  # noqa: E402


def test_solve_rock_paper_scissors():
    M = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    v, x, y = leads.solve(M)
    assert v == pytest.approx(0, abs=1e-6)
    assert x == pytest.approx([1 / 3] * 3, abs=1e-6) and y == pytest.approx([1 / 3] * 3, abs=1e-6)


def test_solve_dominant_row():
    v, x, _ = leads.solve(np.array([[3.0, 2.0], [1.0, 0.0]]))
    assert v == pytest.approx(2.0) and x == pytest.approx([1, 0], abs=1e-6)


def test_turn1_faints():
    log = "|turn|1\n|move|p1a: A|Tackle|p2a: B\n|faint|p2a: B\n|faint|p1b: C\n|turn|2\n|faint|p1a: A\n"
    assert leads.turn1_faints(log) == {"p1": 1, "p2": 1}


def mon(name, speed, types=("Normal",), moves=()):
    return SimpleNamespace(name=name, speed=speed, types=types, ability=None, item=None, moves=list(moves))


def side(*mons):
    return SimpleNamespace(specs=list(mons), speed=[m.speed for m in mons], boosts=[{}, {}])


def test_resolve_fake_out_protect_and_order(monkeypatch):
    monkeypatch.setattr(leads, "TURN2", 0.0)
    # every hit does 60%, Fake Out 10%
    monkeypatch.setattr(leads, "damage",
                        lambda me, i, other, t, moves, w, tr: [10.0 if moves == ["Fake Out"] else 60.0])
    monkeypatch.setattr(leads, "move_info", lambda m: {"priority": 0, "category": "Physical"})
    a, b = side(mon("A1", 50), mon("A2", 50)), side(mon("B1", 100), mon("B2", 100))
    # a: Fake Out on B1, attack B2; b: B1 attacks A1, B2 protects -> B1 flinches, B2 is protected
    pay, ko_a, ko_b = leads.resolve(a, b, [("fake", 0), ("move", "Hit", 1)], [("move", "Hit", 0), ("protect",)],
                                    None, None)
    assert ko_a == 0 and ko_b == 0
    assert pay == pytest.approx(0.10)  # only Fake Out's 10% landed
    # two 60% hits on the same target KO it (+0.5 bonus); the faster side moves first
    pay, ko_a, ko_b = leads.resolve(a, b, [("move", "Hit", 0), ("move", "Hit", 0)], [("pass",), ("pass",)],
                                    None, None)
    assert ko_a == 1 and pay == pytest.approx(1.0 + 0.5)


def test_resolve_switch_wide_guard_and_redirect(monkeypatch):
    monkeypatch.setattr(leads, "TURN2", 0.0)
    # A1's Hit does 60% to B1 / B2 but only 20% to the back Pokemon B3; Spread does 30% to each target
    def dmg(me, i, other, j, moves, w, tr):
        if moves == ["Spread"]:
            return [30.0]
        return [20.0 if other.specs[j].name == "B3" else 60.0]
    monkeypatch.setattr(leads, "damage", dmg)
    monkeypatch.setattr(leads, "move_info", lambda m: {"priority": 0, "category": "Physical"})
    a = side(mon("A1", 50), mon("A2", 50))
    b = side(mon("B1", 100), mon("B2", 100), mon("B3", 100))
    # B1 switches to B3: the hit aimed at slot 0 lands on B3 (20%), B3 does not move this turn
    pay, _, _ = leads.resolve(a, b, [("move", "Hit", 0), ("pass",)], [("switch", 2), ("pass",)], None, None)
    assert pay == pytest.approx(0.20)
    # Wide Guard blocks the spread move entirely
    pay, _, _ = leads.resolve(a, b, [("move", "Spread", None), ("pass",)], [("wideguard",), ("pass",)], None, None)
    assert pay == pytest.approx(0.0)
    # Follow Me on slot 1 pulls the single-target hit aimed at slot 0
    pay, _, _ = leads.resolve(a, b, [("move", "Hit", 0), ("pass",)], [("pass",), ("redirect",)], None, None)
    assert pay == pytest.approx(0.60)
    # Helping Hand: 60% * 1.5 = 90%
    pay, _, _ = leads.resolve(a, b, [("move", "Hit", 0), ("helping",)], [("pass",), ("pass",)], None, None)
    assert pay == pytest.approx(0.90)



class Mon(SimpleNamespace):
    __hash__ = object.__hash__  # leads caches per set


def test_turn2_tailwind_and_setup(monkeypatch):
    # Hit does 60%, 90% at +2 Atk; on turn 2 every active Pokemon attacks greedily
    monkeypatch.setattr(leads, "damage", lambda me, i, other, j, moves, w, tr:
                        [90.0 if me.boosts[i].get("atk") == 2 else 60.0 for _ in moves])
    monkeypatch.setattr(leads, "move_info", lambda m: {"priority": 0, "spread": False, "category": "Status" if m in (
        "Tailwind", "Swords Dance") else "Physical"})
    monkeypatch.setattr(leads, "TURN2", 1.0)
    leads.attack_moves.cache_clear()

    def mk(name, speed, moves):
        return Mon(name=name, speed=speed, types=("Normal",), ability=None, item=None, moves=moves)
    a = leads.Side([mk("A1", 50, ["Hit", "Tailwind"]), mk("A2", 50, ["Hit", "Swords Dance"])])
    b = leads.Side([mk("B1", 90, ["Hit"]), mk("B2", 90, ["Hit"])])
    # turn 1: A sets Tailwind + Swords Dance, B passes. Turn 2: A outspeeds (100 vs 90): 60% + 90% KO B1 before it
    # moves (1.5), B2 hits A1 for 60% -> 0.9
    pay, _, _ = leads.resolve(a, b, [("tailwind",), ("setup", "Swords Dance")], [("pass",), ("pass",)], None, None)
    assert pay == pytest.approx(0.9)
    # Swords Dance without Tailwind: B moves first and KOs A1 (60 + 60), then A2's +2 hit does 90% -> 0.9 - 1.5
    pay, _, _ = leads.resolve(a, b, [("pass",), ("setup", "Swords Dance")], [("pass",), ("pass",)], None, None)
    assert pay == pytest.approx(-0.6)
    leads.attack_moves.cache_clear()
