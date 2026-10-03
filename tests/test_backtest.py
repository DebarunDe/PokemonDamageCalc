import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import backtest  # noqa: E402
import ingest  # noqa: E402


def test_key_matches_limitless_and_showdown_names():
    assert backtest.key("calyrex-shadow-rider") == backtest.key("Calyrex-Shadow")
    assert backtest.key("tatsugiri-droopy") == backtest.key("Tatsugiri")
    assert backtest.key("Aegislash-Blade") == "aegislash"
    assert backtest.key("Urshifu-Rapid-Strike") == "urshifurapidstrike"


def test_shrink_pulls_small_samples_harder():
    raw = pd.Series([0.6, 0.6, 0.4, 0.4])
    games = pd.Series([1000, 50, 1000, 50])
    shrunk, k = backtest.shrink(raw, games)
    assert k > 0
    assert abs(shrunk[0] - 0.5) > abs(shrunk[1] - 0.5)
    assert shrunk[1] > 0.5 > shrunk[3]


def test_store_usage_table(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "t.sqlite")
    db.executescript(ingest.SCHEMA)
    text = """ Total battles: 1234
 Avg. weight/team: 1.0
 + ---- + ------------------ + --------- + ------ + ------- + ------ + ------- +
 | Rank | Pokemon            | Usage %   | Raw    | %       | Real   | %       |
 + ---- + ------------------ + --------- + ------ + ------- + ------ + ------- +
 | 1    | Urshifu-Rapid-Strike | 50.77012% | 1000   | 40.000% | 900    | 40.000% |
 | 2    | Incineroar         | 45.00000% | 900    | 36.000% | 800    | 36.000% |
"""
    ingest.store_usage_table(db, "gen9vgc2024regg", "2024-06", "bo1", 0, text)
    rows = db.execute("SELECT pokemon, usage, battles FROM showdown_usage ORDER BY usage DESC").fetchall()
    assert rows[0][0] == "Urshifu-Rapid-Strike"
    assert rows[0][1] == pytest.approx(0.5077012)
    assert rows[0][2] == 1234
    assert len(rows) == 2
