"""One command to refresh everything and rebuild the report page.

    python analysis/refresh.py                  # ingest new data, rebuild features, model, page
    python analysis/refresh.py --skip-ingest    # reuse the downloaded data

Steps: download new Showdown months and Limitless tournaments (cached, so only
new data is fetched); build matchup features for the current regulation's
latest ladder month and for each backtest snapshot; score every Pokemon; run
the backtests; write analysis/output/report.json and analysis/output/report.html
(the template in analysis/report_template.html with the data embedded).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import model  # noqa: E402
import signals  # noqa: E402
from champcalc import Calculator  # noqa: E402

OUT = HERE / "output"

# Plain-language labels for the usage model's features, phrased for a
# positive contribution (e.g. a low counter_pressure counts in a Pokemon's favor).
FEATURE_LABELS = {
    "off_ohko_set": "OHKOs the meta", "off_2hko_set": "2HKOs the meta", "off_damage_set": "heavy damage",
    "off_ohko_potential": "OHKO coverage", "off_2hko_potential": "2HKO coverage",
    "off_damage_potential": "damage coverage", "def_ohkod": "rarely OHKO'd", "def_2hkod": "hard to 2HKO",
    "def_damage_taken": "takes little damage", "speed_outspeeds": "outspeeds the meta",
    "speed_tailwind": "fast under Tailwind", "is_mega": "Mega", "base_stat_total": "high base stats",
    "counter_pressure": "rarely checked", "counter_pressure_missing": "ladder data",
    "util_weather": "sets weather", "util_terrain": "sets terrain", "util_attack_drop": "lowers Attack",
    "util_prankster": "Prankster", "util_scaling_attack": "scaling attack", "util_priority_block": "blocks priority",
    "util_speed_ability": "speed ability", "util_fake_out_immune": "ignores Fake Out",
    "util_intimidate_immune": "shrugs off Intimidate", "util_status_immune": "status immunity",
    "util_priority_attack": "priority attack", "util_spread_stab": "spread STAB", "util_speed_drop": "speed control",
    "util_trick_room": "Trick Room", "util_tailwind": "Tailwind", "util_redirection": "redirection",
    "util_fake_out": "Fake Out", "util_intimidate": "Intimidate", "util_pivot": "pivoting",
    "used_speed_drop": "runs speed control", "used_attack_drop": "runs Attack drops",
}


def label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    if feature.startswith("used_"):
        return "runs " + feature[5:].replace("_", " ")
    if feature.startswith("util_"):
        return feature[5:].replace("_", " ")
    return feature


def reasons(text: str) -> list[str]:
    """'speed_outspeeds +0.99, base_stat_total +0.58' -> ['outspeeds the meta', ...] (positive only)."""
    out = []
    for part in str(text).split(", "):
        name, _, value = part.rpartition(" ")
        try:
            if float(value) > 0.05:
                out.append(label(name))
        except ValueError:
            continue
    return out


def run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def number(x, digits=4):
    return None if x is None or x != x else round(float(x), digits)


def rows(frame, columns) -> list[dict]:
    out = []
    for _, r in frame.iterrows():
        row = {c: number(r[c]) if c != "pokemon" else r[c] for c in columns if c in r}
        row["reasons"] = reasons(r.get("reasons", ""))
        out.append(row)
    return out


def monthly_backtest() -> dict | None:
    """backtest.py's month-by-month summary (committed, so the monthly refresh
    has it without downloading the Scarlet/Violet history)."""
    path = HERE / "backtest_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if not args.skip_ingest:
        run("ingest.py")
    db = sqlite3.connect(model.DB)
    regulation = Calculator().regulation
    month = db.execute("SELECT MAX(month) FROM showdown_usage WHERE regulation=?", (regulation,)).fetchone()[0]
    model.SNAPSHOTS[regulation] = month
    for reg, m in {(regulation, month), *(b for b, _ in model.BACKTESTS)}:
        if reg == regulation or not (OUT / f"meta_matchups_{reg}_{m}.csv").exists():
            run("meta_matchups.py", "--regulation", reg, "--month", m)

    df, meta = model.score(regulation, month, db)
    t = signals.tournament_stats(db, regulation)
    battles = db.execute(
        """SELECT battles FROM showdown_usage WHERE regulation=? AND month=? AND ladder='bo1' AND rating=0
           LIMIT 1""", (regulation, month)).fetchone()

    columns = ["pokemon", "usage", "games", "win_rate_shrunk", "pilot_skill", "edge", "conversion", "ladder_lift"]
    proven = df[df["win_rate"].notna() & (df["usage"] < model.POPULAR_USAGE)]
    overused = df[(df["usage"] >= 0.02) & (df["games"] >= model.MIN_EDGE_GAMES)].nsmallest(12, "win_rate_shrunk")
    potential = df[(df["games"] < model.MIN_EDGE_GAMES) & (df["expected_usage"] >= model.MIN_EXPECTED_USAGE)]
    backtests = []
    for before, after in model.BACKTESTS:
        r = model.backtest(db, before, after)
        backtests.append({
            "from": f"{before[0]} ({before[1]})", "to": f"{after[0]} ({after[1]})", "pokemon": r["n"],
            "scores": {name: {"spearman": number(r[name]["spearman"], 3),
                              "top20_median": number(r[name]["top20_median"], 3),
                              "top20_rose": number(r[name]["top20_rose"], 3)}
                       for name in ("win rate", "edge (with data)", "performance", "usage_gap")},
        })
    report = {
        "regulation": regulation,
        "month": month,
        "generated": time.strftime("%Y-%m-%d"),
        "tournaments": int(t.attrs.get("n_tournaments", 0)),
        "teams": int(t.attrs.get("n_teams", 0)),
        "ladder_battles": int(battles[0]) if battles else 0,
        "pokemon": len(df),
        "usage_r2": number(meta["r2"][meta["model"]], 3),
        "edge_prior": meta.get("edge_prior"),
        "prior_strength_games": meta.get("prior_strength_games"),
        "win_rate_prior_games": meta.get("win_rate_prior_games"),
        "field_win_rate": meta.get("field_win_rate"),
        "monthly_backtest": monthly_backtest(),
        "rewards": [label(k) for k, v in meta["coefficients"][::-1].items() if v > 0][:8],
        "proven": rows(proven.sort_values("win_rate_shrunk", ascending=False).head(30), columns),
        "overused": rows(overused, columns),
        "potential": rows(potential.sort_values("usage_gap", ascending=False).head(12),
                          ["pokemon", "usage", "expected_usage", "usage_gap"]),
        "backtests": backtests,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    template = (HERE / "report_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__REPORT__*/null", json.dumps(report))
    (OUT / "report.html").write_text(html, encoding="utf-8")
    print(f"Wrote {OUT / 'report.json'} and {OUT / 'report.html'} ({regulation} {month})")


if __name__ == "__main__":
    main()
