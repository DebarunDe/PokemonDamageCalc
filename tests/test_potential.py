"""Under-used search in analysis/potential.py: species and pair effects on synthetic teams."""

import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import potential as pt  # noqa: E402


def test_fit_separates_species_and_pair_effects():
    rng = random.Random(0)
    rows = pt.Rows("synthetic")
    pool = [f"S{i:02d}" for i in range(20)]
    for k in range(20000):
        team = rng.sample(pool, 6)
        y = 0.08 * ("S00" in team) + 0.10 * ("S01" in team and "S02" in team) + rng.gauss(0, 0.3)
        rows.add(team, y, 1, f"2026-10-{1 + k % 20:02d}")
    f = pt.fit(rows, lam_s=20, lam_p=20)
    assert f["species"]["S00"] > 0.06 and abs(f["species"]["S05"]) < 0.02
    # two stages: a pair on a quarter of its members' teams is understated, but stands out
    assert f["pair"][("S01", "S02")] > 0.03 and f["pair"][("S01", "S02")] - abs(f["pair"][("S05", "S06")]) > 0.02
    early, late = rows.subset(lambda d: d < "2026-10-11"), rows.subset(lambda d: d >= "2026-10-11")
    assert len(early.teams) == 10000
    slope, se, r, n = pt.carry(early, late, "species", 100, 100)
    assert n == 20 and slope > 0.5 and r > 0.5  # a real species effect carries forward


def test_combine_and_corr():
    v, s = pt.combine([(1.0, 1.0), (3.0, 1.0), (None, None)])
    assert v == pytest.approx(2.0) and s == pytest.approx(2 ** -0.5)
    r, n = pt.corr({k: k for k in range(10)}, {k: 2 * k + 1 for k in range(10)})
    assert r == pytest.approx(1.0) and n == 10
