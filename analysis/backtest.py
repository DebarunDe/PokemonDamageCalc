"""Month-by-month backtest: does a Pokemon's tournament edge predict its usage?

For every month with enough tournaments, and for each game series (Champions,
and Scarlet/Violet VGC from `ingest.py history`):

    signal  each Pokemon's skill-adjusted edge from that month's tournaments:
            wins per game above what its players' records predict, with player
            skill taken only from tournaments up to that month (no peeking),
            shrunk toward the month's average by sample size
    target  its usage change on the main Showdown ladder at 1760+ from that
            month to the next, beyond the usual drift of rare Pokemon toward
            the average (the residual of change on starting usage)

Transitions within a regulation and across a regulation change are reported
separately. Raw (unadjusted) win rate is reported alongside, as a baseline.

Usage:
    python analysis/backtest.py
    python analysis/backtest.py --series champions
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"

SERIES = {
    "champions": {
        "limitless": ("M-A", "M-B", "M-C"),
        "showdown": lambda reg: reg.startswith("M-"),
    },
    "scarlet-violet": {
        "limitless": ("23S1", "23S2", "23S3", "VGC23", "SVE", "SVF", "SVG", "SVH", "SVI"),
        "showdown": lambda reg: reg.startswith("gen9vgc") and not reg.endswith("bo3"),
    },
}
MIN_TOURNAMENTS = 8      # months with fewer tournaments in the main format are skipped
MIN_GAMES = 50           # Pokemon need this many tournament games in the month
PRIOR_SKILL_GAMES = 30   # player-skill shrinkage, in games
USAGE_FLOOR = 1e-4
TOP = 20


def key(name: str) -> str:
    k = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return "aegislash" if k.startswith("aegislash") else k


def spearman(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def load(db: sqlite3.Connection, formats: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every team (with its tournament's format and month) and every team member."""
    members = pd.read_sql_query(
        f"""SELECT m.tournament, m.player, m.wins, m.losses, m.pokemon, t.regulation AS format, t.date
            FROM team_members m JOIN tournaments t ON t.id = m.tournament
            WHERE t.regulation IN ({",".join("?" * len(formats))})""",
        db, params=formats)
    members["month"] = members["date"].str[:7]
    members["pokemon"] = members["pokemon"].map(key)
    members = members.drop_duplicates(["tournament", "player", "pokemon"])
    teams = members.drop_duplicates(["tournament", "player"])[
        ["tournament", "player", "wins", "losses", "format", "date", "month"]].copy()
    teams["games"] = teams["wins"] + teams["losses"]
    return teams, members


def all_teams(db: sqlite3.Connection) -> pd.DataFrame:
    """Every team in any format, for player skill (skill carries across games)."""
    teams = pd.read_sql_query(
        """SELECT DISTINCT m.tournament, m.player, m.wins, m.losses, t.date
           FROM team_members m JOIN tournaments t ON t.id = m.tournament""", db)
    teams["games"] = teams["wins"] + teams["losses"]
    return teams


def month_signal(month: str, teams: pd.DataFrame, members: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame | None:
    in_month = teams[teams["month"] == month]
    if in_month.empty:
        return None
    main_format = in_month["format"].value_counts().idxmax()
    in_month = in_month[in_month["format"] == main_format]
    if in_month["tournament"].nunique() < MIN_TOURNAMENTS:
        return None

    # Player skill from everything up to the end of this month, leaving out the
    # tournament being explained.
    end = month + "-31T23:59"
    past = history[history["date"] <= end]
    totals = past.groupby("player")[["wins", "games"]].sum()
    overall = past["wins"].sum() / max(1, past["games"].sum())
    t = in_month.join(totals, on="player", rsuffix="_total")
    other_wins = t["wins_total"] - t["wins"]
    other_games = t["games_total"] - t["games"]
    t["skill"] = (other_wins + PRIOR_SKILL_GAMES * overall) / (other_games + PRIOR_SKILL_GAMES)
    t["expected"] = t["skill"] * t["games"]

    m = members.merge(t[["tournament", "player", "expected", "games"]], on=["tournament", "player"])
    g = m.groupby("pokemon").agg(wins=("wins", "sum"), losses=("losses", "sum"),
                                 expected=("expected", "sum"), games=("games", "sum"))
    g = g[g["games"] >= MIN_GAMES]
    if len(g) < 20:
        return None
    raw = (g["wins"] - g["expected"]) / g["games"]
    w = g["games"]
    mean = np.average(raw, weights=w)
    tau2 = max(np.average((raw - mean) ** 2, weights=w) - np.average(0.25 / g["games"], weights=w), 1e-5)
    k = 0.25 / tau2
    g["edge"] = (g["games"] * raw + k * mean) / (g["games"] + k)
    g["win_rate"] = g["wins"] / g["games"]
    g.attrs.update(format=main_format, tournaments=int(in_month["tournament"].nunique()),
                   teams=len(in_month), prior_games=k)
    return g


def ladder(db: sqlite3.Connection, series: dict) -> pd.DataFrame:
    """Main Showdown ladder per month (most battles), and usage at 1760+ and at all ratings."""
    rows = pd.read_sql_query(
        "SELECT regulation, month, rating, battles, pokemon, usage FROM showdown_usage WHERE ladder='bo1'", db)
    rows = rows[rows["regulation"].map(series["showdown"])]
    battles = rows[rows["rating"] == 0].groupby(["month", "regulation"])["battles"].max().reset_index()
    main = battles.loc[battles.groupby("month")["battles"].idxmax()].set_index("month")["regulation"]
    rows = rows[rows["regulation"] == rows["month"].map(main)]
    rows["pokemon"] = rows["pokemon"].map(key)
    return rows.groupby(["month", "rating", "pokemon"], as_index=False).agg(
        usage=("usage", "sum"), regulation=("regulation", "first"))


def run(db: sqlite3.Connection, name: str) -> pd.DataFrame:
    series = SERIES[name]
    teams, members = load(db, series["limitless"])
    if teams.empty:
        return pd.DataFrame()
    history = all_teams(db)
    usage = ladder(db, series)
    months = sorted(set(teams["month"]) & set(usage["month"]))
    results = []
    for month, nxt in zip(months, months[1:]):
        if (pd.Period(nxt) - pd.Period(month)).n != 1:
            continue
        sig = month_signal(month, teams, members, history)
        if sig is None:
            continue
        now = usage[(usage["month"] == month) & (usage["rating"] == 1760)].set_index("pokemon")["usage"]
        later = usage[(usage["month"] == nxt) & (usage["rating"] == 1760)].set_index("pokemon")["usage"]
        # Pokemon missing from next month's ladder entirely were banned or rotated out.
        legal_next = set(usage[(usage["month"] == nxt) & (usage["rating"] == 0)]["pokemon"])
        df = sig[sig.index.isin(legal_next)].copy()
        df["log_now"] = np.log(now.reindex(df.index).fillna(0).clip(lower=USAGE_FLOOR))
        df["log_next"] = np.log(later.reindex(df.index).fillna(0).clip(lower=USAGE_FLOOR))
        df["change"] = df["log_next"] - df["log_now"]
        slope, intercept = np.polyfit(df["log_now"], df["change"], 1)
        df["resid"] = df["change"] - (slope * df["log_now"] + intercept)
        top = df.nlargest(TOP, "edge")
        reg_now = usage.loc[usage["month"] == month, "regulation"].iloc[0]
        reg_next = usage.loc[usage["month"] == nxt, "regulation"].iloc[0]
        results.append({
            "series": name, "month": month, "next": nxt,
            "ladder": reg_now, "next_ladder": reg_next, "new_regulation": reg_now != reg_next,
            "tournament_format": sig.attrs["format"], "tournaments": sig.attrs["tournaments"],
            "teams": sig.attrs["teams"], "pokemon": len(df),
            "edge_spearman": spearman(df["edge"], df["resid"]),
            "win_rate_spearman": spearman(df["win_rate"], df["resid"]),
            "edge_top_beat": (top["resid"] > 0).mean(),
            "edge_top_median": top["resid"].median(),
            "top_edge": ", ".join(top.index[:5]),
        })
    return pd.DataFrame(results)


def summarize(df: pd.DataFrame, label: str) -> str:
    if df.empty:
        return f"{label}: no transitions"
    out = []
    for col in ("edge_spearman", "win_rate_spearman"):
        x = df[col].dropna()
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
        out.append(f"{col.split('_')[0]:8} mean {x.mean():+.3f}, positive in {(x > 0).sum()}/{len(x)}, t = {t:+.1f}")
    beat = df["edge_top_beat"].mean()
    return f"{label} ({len(df)} transitions): " + "; ".join(out) + f"; edge top {TOP} beat mean reversion {beat:.0%}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", choices=("all", *SERIES), default="all")
    args = parser.parse_args()
    # Read-only, so it can run while ingest.py is writing.
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=120)
    names = list(SERIES) if args.series == "all" else [args.series]
    results = pd.concat([run(db, n) for n in names], ignore_index=True)
    if results.empty:
        raise SystemExit("No months with enough data; run analysis/ingest.py (and `ingest.py history`).")
    OUT.mkdir(exist_ok=True)
    results.to_csv(OUT / "backtest_monthly.csv", index=False)
    cols = ["series", "month", "next", "new_regulation", "tournaments", "teams", "pokemon",
            "edge_spearman", "win_rate_spearman", "edge_top_beat", "top_edge"]
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60,
                           "display.float_format", "{:+.3f}".format):
        print(results[cols].to_string(index=False))
    print()
    for name in names:
        r = results[results["series"] == name]
        print(summarize(r, name))
        print("  " + summarize(r[~r["new_regulation"]], "within a regulation"))
        print("  " + summarize(r[r["new_regulation"]], "into a new regulation"))
    print(summarize(results, "all"))


if __name__ == "__main__":
    main()
