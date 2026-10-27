"""Switch-in coverage in analysis/switching.py: the scoring rule and the team-paste parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import switching as sw  # noqa: E402


def test_score_counts_threats_with_a_safe_switch_in():
    # attack 0 threatens member 0 (60%), member 1 takes 10% from it: covered
    # attack 1 threatens both (80% / 70%), member 2 takes 45%: half covered
    # attack 2 threatens nobody
    rows = [[60, 80, 10], [10, 70, 10], [90, 45, 10]]
    s, den, detail = sw.score(rows, [1.0, 1.0, 1.0])
    # threatened: (0,a0) covered by 1 -> 1; (0,a1) best mate 45% -> 0.5; (1,a1) -> 0.5; (2,a0) mate 10% -> 1
    assert den == 4.0 and s == (1 + 0.5 + 0.5 + 1) / 4
    assert (0, 0, 1, 10) in detail


def test_parse_paste_reads_megas_and_stat_points():
    text = """Gengar @ Gengarite
Ability: Cursed Body
SPs: 32 HP / 2 SpA / 32 Spe
Modest Nature
- Shadow Ball
- Protect

Pelipper @ Damp Rock
Ability: Drizzle
SPs: 32 HP / 2 Def / 32 SpD
Bold Nature
- Hurricane
- Wide Guard
"""
    gengar, pelipper = sw.parse_paste(text)
    assert gengar.name == "Gengar-Mega" and gengar.ability == "Shadow Tag" and gengar.pre_ability == "Cursed Body"
    assert gengar.sp == {"hp": 32, "spa": 2, "spe": 32} and gengar.nature == "Modest"
    assert pelipper.name == "Pelipper" and pelipper.moves == ["Hurricane", "Wide Guard"]
