import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import rk9  # noqa: E402

ROSTER = """<table><thead><tr><th>Player ID</th></tr></thead><tbody>
<tr>
  <td>5....2</td><td>Michael</td><td> Zaslavsky </td><td>US</td><td>Masters</td><td>mz</td>
  <td><a href="/teamlist/public/LO002-x/abc123" target="_blank"><strong>View</strong></a></td>
  <td class="text-center">42</th>
</tr>
<tr>
  <td>1....9</td><td>Ann</td><td>Lee</td><td>CA</td><td>Junior</td><td>al</td><td></td><td class="text-center">3</th>
</tr>
</tbody></table>"""

TEAM = """
<div class="pokemon bg-light-green-50 p-3">
<img src="x.png" onerror="imgError(this)" style="float: left;">
Raichu
&nbsp;
<b>EN</b> <br>
<b>Ability:</b> Lightning Rod &nbsp;&nbsp;
<b>Held Item:</b> Raichunite Y<br>
<b>Stat Alignment:</b> Timid<br>
<h5>&nbsp;&nbsp;<span class="badge">Fake Out</span>
<span class="badge">Zap Cannon</span></h5>
</div>
<div class="pokemon bg-light-green-50 p-3">
<img src="y.png" onerror="imgError(this)" style="float: left;">
Sirfetch&#39;d
&nbsp;
<b>Ability:</b> Scrappy &nbsp;&nbsp;
<b>Held Item:</b> Leek<br>
<b>Stat Alignment:</b> Adamant<br>
<h5><span class="badge">Close Combat</span></h5>
</div>
<div class="pokemon bg-light-green-50 p-3">
<img src="z.png" onerror="imgError(this)">
Arcanine [Hisuian Form]
&nbsp;
<b>Ability:</b> Rock Head &nbsp;&nbsp;
<b>Held Item:</b> Focus Sash<br>
<b>Stat Alignment:</b> Jolly<br>
</div>
"""

ROUND = """
<div class="col-5 player player1 winner"><span class="name">Jacob<br> Acker [US]<br></span> (5-0-0) 15 pts <br></div>
<div class="col-5 player player2 loser"><span class="name">Lauri<br> Halonen [FI]<br></span> (4-1-0) 12 pts <br></div>
"""


def test_end_date():
    assert rk9.end_date("October 9-11, 2026") == date(2026, 10, 11)
    assert rk9.end_date("October 31-November 1, 2026") == date(2026, 11, 1)
    assert rk9.end_date("June 27, 2026") == date(2026, 6, 27)
    assert rk9.end_date("TBA") is None


def test_parse_events():
    page = """<tr><td>October 9-11, 2026</td><td><img></td>
    <td><a href="/event/pokemon-louisville-2027" target="_blank">2027 Louisville Pok&eacute;mon Regional Championships</a></td></tr>"""
    (event,) = rk9.parse_events(page)
    assert event["slug"] == "pokemon-louisville-2027"
    assert event["name"] == "2027 Louisville Pokémon Regional Championships"
    assert event["end"] == date(2026, 10, 11)


def test_parse_roster():
    players = rk9.parse_roster(ROSTER)
    assert len(players) == 2
    first = players[0]
    assert (first["first"], first["last"], first["country"], first["division"]) == ("Michael", "Zaslavsky", "US", "Masters")
    assert first["standing"] == 42 and first["url"] == "/teamlist/public/LO002-x/abc123"
    assert players[1]["url"] is None


def test_parse_team_and_forms():
    team = rk9.parse_team(TEAM)
    assert [m["species"] for m in team] == ["Raichu", "Sirfetch'd", "Arcanine-Hisui"]
    assert team[0]["ability"] == "Lightning Rod" and team[0]["item"] == "Raichunite Y" and team[0]["nature"] == "Timid"
    assert team[0]["moves"] == ["Fake Out", "Zap Cannon"]
    assert rk9.species_name("Sinistcha [Masterpiece Form]") == "Sinistcha"
    assert rk9.species_name("Garchomp") == "Garchomp"


def test_parse_round_and_keys():
    records = rk9.parse_round(ROUND)
    assert records == {"Jacob Acker [US]": (5, 0, 0), "Lauri Halonen [FI]": (4, 1, 0)}
    assert rk9.player_key("Jacob", "Acker", "US") in records


def test_legal_regulation_picks_oldest_fitting():
    # Each regulation adds to the last, so older teams are legal in every later one too.
    regs = {"M-A": {"Raichu", "Garchomp"}, "M-B": {"Raichu", "Garchomp", "Kingambit"},
            "M-C": {"Raichu", "Garchomp", "Kingambit", "Venusaur"}}
    assert rk9.legal_regulation([["Raichu", "Garchomp"]] * 10, regs) == "M-A"
    assert rk9.legal_regulation([["Raichu", "Kingambit"]] * 10, regs) == "M-B"
    assert rk9.legal_regulation([["Venusaur", "Kingambit"]] * 10, regs) == "M-C"
    assert rk9.legal_regulation([["Pikachu"]] * 10, regs) is None
