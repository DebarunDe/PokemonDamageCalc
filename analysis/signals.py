"""First-pass "undervalued" signals from usage and tournament results.

For each Pokemon in a regulation:
- usage: share of published Limitless teams that include it
- win_rate: match win rate of those teams, shrunk toward the field average
  (a 3-1 Pokemon shouldn't top the list)
- win_rate_vs_skill: wins above what the teams' pilots would be expected to win
  given their record in their other tournaments; strong players bringing a
  Pokemon doesn't make it look strong
- conversion: its share of top-cut teams divided by its share of all teams
  (above 1 = it makes top cut more often than its popularity predicts)
- ladder_lift: Showdown best-of-1 usage at 1760+ divided by usage at all
  ratings (above 1 = strong players use it more)

performance is the average z-score of win_rate_vs_skill, conversion and
ladder_lift; undervalued is how far
performance is above what its usage would predict (the residual of a fit of
performance on log usage), so it doesn't just reward being rare.

The backtest checks whether a regulation's undervalued scores predict which
Pokemon gain usage in the next regulation, beyond simple regression to the
mean.

Usage:
    python analysis/signals.py                   # M-C table + backtests
    python analysis/signals.py --regulation M-B
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"
REGULATIONS = ("M-A", "M-B", "M-C")

PRIOR_GAMES = 100       # win-rate shrinkage strength, in games
PRIOR_TOP_CUT = 5       # top-cut shrinkage strength, in expected top-cut teams
PRIOR_SKILL_GAMES = 30  # player-skill shrinkage strength, in games
MIN_TEAMS = 30          # Pokemon on fewer published teams are left out
MIN_LADDER_USAGE = 0.005


# Showdown and Limitless name a few forms differently; compare on one spelling.
SAME_POKEMON = {
    "Aegislash-Both": "Aegislash", "Aegislash-Blade": "Aegislash",
    "Meowstic-F": "Meowstic", "Toxtricity-Low-Key": "Toxtricity",
}


def canonical(name: str) -> str:
    name = name.replace("\u2019", "'")
    return SAME_POKEMON.get(name, name)


def top_cut_size(players: int) -> int:
    """Top cut for Swiss events of this size: top 12.5%, at least 4."""
    return max(4, math.ceil(players * 0.125))


def player_skill(db: sqlite3.Connection) -> pd.DataFrame:
    """Each player's match win rate in all their *other* tournaments (any
    regulation), shrunk toward the overall average: who they are, excluding the
    event being explained."""
    teams = pd.read_sql_query(
        "SELECT DISTINCT tournament, player, wins, losses FROM team_members", db)
    teams["games"] = teams["wins"] + teams["losses"]
    overall = teams["wins"].sum() / max(1, teams["games"].sum())
    totals = teams.groupby("player")[["wins", "games"]].transform("sum")
    other_wins = totals["wins"] - teams["wins"]
    other_games = totals["games"] - teams["games"]
    teams["skill"] = (other_wins + PRIOR_SKILL_GAMES * overall) / (other_games + PRIOR_SKILL_GAMES)
    teams["skill_games"] = other_games
    return teams[["tournament", "player", "skill", "skill_games"]]


def tournament_stats(db: sqlite3.Connection, regulation: str) -> pd.DataFrame:
    members = pd.read_sql_query(
        """SELECT m.tournament, m.player, m.placing, m.wins, m.losses, m.pokemon, t.players
           FROM team_members m JOIN tournaments t ON t.id = m.tournament
           WHERE t.regulation = ?""",
        db, params=(regulation,),
    )
    if members.empty:
        return pd.DataFrame()
    members["pokemon"] = members["pokemon"].map(canonical)
    teams = members.drop_duplicates(["tournament", "player"]).copy()
    teams["top_cut"] = teams["placing"] <= teams["players"].map(top_cut_size)
    teams = teams.merge(player_skill(db), on=["tournament", "player"], how="left")
    teams["expected_wins"] = teams["skill"] * (teams["wins"] + teams["losses"])
    n_teams, n_top = len(teams), int(teams["top_cut"].sum())
    field_wr = teams["wins"].sum() / max(1, (teams["wins"] + teams["losses"]).sum())

    members = members.drop_duplicates(["tournament", "player", "pokemon"])
    members = members.merge(teams[["tournament", "player", "top_cut", "skill", "expected_wins"]],
                            on=["tournament", "player"])
    g = members.groupby("pokemon")
    stats = pd.DataFrame({
        "teams": g.size(),
        "wins": g["wins"].sum(),
        "losses": g["losses"].sum(),
        "expected_wins": g["expected_wins"].sum(),
        "pilot_skill": g["skill"].mean(),
        "top_cut_teams": g["top_cut"].sum(),
    })
    stats["usage"] = stats["teams"] / n_teams
    stats["win_rate"] = (stats["wins"] + PRIOR_GAMES * field_wr) / (stats["wins"] + stats["losses"] + PRIOR_GAMES)
    # Wins above what its pilots' skill predicts, per game, shrunk toward 0.
    stats["win_rate_vs_skill"] = (stats["wins"] - stats["expected_wins"]) / (
        stats["wins"] + stats["losses"] + PRIOR_GAMES)
    # Actual vs expected top-cut appearances, shrunk toward 1 for small samples.
    expected = stats["teams"] * n_top / n_teams
    stats["conversion"] = (stats["top_cut_teams"] + PRIOR_TOP_CUT) / (expected + PRIOR_TOP_CUT)
    stats.attrs.update(n_teams=n_teams, n_top=n_top, field_wr=field_wr,
                       n_tournaments=members["tournament"].nunique())
    return stats


def ladder_stats(db: sqlite3.Connection, regulation: str) -> pd.DataFrame:
    usage = pd.read_sql_query(
        """SELECT rating, pokemon, SUM(usage * battles) / SUM(battles) AS usage
           FROM showdown_usage WHERE regulation = ? AND ladder = 'bo1'
           GROUP BY rating, pokemon""",
        db, params=(regulation,),
    )
    usage["pokemon"] = usage["pokemon"].map(canonical)
    usage = usage.groupby(["rating", "pokemon"], as_index=False)["usage"].sum()
    usage = usage.pivot(index="pokemon", columns="rating", values="usage")
    if usage.empty or 0 not in usage or 1760 not in usage:
        return pd.DataFrame()
    out = pd.DataFrame({"ladder_usage": usage[0], "ladder_usage_1760": usage[1760]})
    out = out[out["ladder_usage"] >= MIN_LADDER_USAGE].fillna(0)
    out["ladder_lift"] = (out["ladder_usage_1760"] + 0.001) / (out["ladder_usage"] + 0.001)
    return out


def spearman(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) or 1)


def signals(db: sqlite3.Connection, regulation: str) -> pd.DataFrame:
    t = tournament_stats(db, regulation)
    if t.empty:
        return t
    df = t[t["teams"] >= MIN_TEAMS].join(ladder_stats(db, regulation), how="left")
    parts = [zscore(df["win_rate_vs_skill"]), zscore(np.log(df["conversion"]))]
    if df["ladder_lift"].notna().any():
        parts.append(zscore(np.log(df["ladder_lift"])).fillna(0))
    df["performance"] = sum(parts) / len(parts)
    # Residual of performance on log usage: what usage alone doesn't explain.
    x = np.log(df["usage"])
    slope, intercept = np.polyfit(x, df["performance"], 1)
    df["undervalued"] = df["performance"] - (slope * x + intercept)
    df.attrs = t.attrs
    return df.sort_values("undervalued", ascending=False)


def backtest(db: sqlite3.Connection, before: str, after: str) -> dict | None:
    """Does `before`'s undervalued score predict usage change into `after`,
    after accounting for regression to the mean (log usage in `before`)?"""
    a, b = signals(db, before), tournament_stats(db, after)
    if a.empty or b.empty:
        return None
    df = a[["usage", "undervalued", "performance"]].join(b[["usage"]], rsuffix="_after", how="inner")
    df["change"] = np.log(df["usage_after"]) - np.log(df["usage"])
    # Remove the part of the change explained by starting usage alone.
    x = np.log(df["usage"])
    slope, intercept = np.polyfit(x, df["change"], 1)
    df["change_beyond_usage"] = df["change"] - (slope * x + intercept)
    top = df.nlargest(15, "undervalued")
    return {
        "pokemon": len(df),
        "spearman_undervalued_vs_change": spearman(df["undervalued"], df["change_beyond_usage"]),
        "spearman_performance_vs_change": spearman(df["performance"], df["change_beyond_usage"]),
        "top15_median_change_beyond_usage": top["change_beyond_usage"].median(),
        "top15": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regulation", default="M-C", choices=REGULATIONS)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    db = sqlite3.connect(DB)
    df = signals(db, args.regulation)
    if df.empty:
        raise SystemExit(f"No tournament data for {args.regulation}; run analysis/ingest.py first")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"signals_{args.regulation}.csv")
    a = df.attrs
    print(f"{args.regulation}: {a['n_tournaments']} tournaments, {a['n_teams']} published teams, "
          f"{a['n_top']} top-cut teams, field win rate {a['field_wr']:.3f}")
    cols = ["usage", "win_rate", "conversion", "ladder_usage_1760", "ladder_lift", "performance", "undervalued"]
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(f"\nMost undervalued (performance above what usage predicts):\n{df[cols].head(args.top)}")
        print(f"\nMost overvalued:\n{df[cols].tail(10)}")

    for before, after in zip(REGULATIONS, REGULATIONS[1:]):
        result = backtest(db, before, after)
        if not result:
            continue
        print(f"\nBacktest {before} -> {after} ({result['pokemon']} Pokemon in both):")
        print(f"  Spearman(undervalued, usage change beyond mean reversion) = "
              f"{result['spearman_undervalued_vs_change']:.3f}")
        print(f"  Spearman(performance, same) = {result['spearman_performance_vs_change']:.3f}")
        print(f"  Top 15 undervalued: median change beyond mean reversion = "
              f"{result['top15_median_change_beyond_usage']:+.2f} log-usage")


if __name__ == "__main__":
    main()
