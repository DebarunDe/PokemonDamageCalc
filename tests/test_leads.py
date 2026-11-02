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


def mon(name, speed, types=("Normal",)):
    return SimpleNamespace(name=name, speed=speed, types=types, ability=None, item=None, moves=[])


def side(*mons):
    return SimpleNamespace(specs=list(mons), speed=[m.speed for m in mons], boosts=[{}, {}])


def test_resolve_fake_out_protect_and_order(monkeypatch):
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
