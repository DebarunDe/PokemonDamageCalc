"""Model what the meta rewards, then find Pokemon used less than they should be.

Features (per Pokemon, from meta_matchups.py plus the database):
    offense / defense / speed against the usage-weighted meta, utility access,
    is_mega (only one Mega per team, which caps Mega usage), base stat total,
    and counter_pressure: how often the meta's Pokemon beat it in ladder games
    (Showdown's checks and counters, weighted by the meta's usage).

Target: log usage on the Showdown best-of-1 ladder at 1760+ ("what strong
players use"). A regularized linear model (ridge) and a gradient-boosted model
are compared with repeated cross-validation; predictions are always made for
Pokemon the model did not train on, so a Pokemon's own usage never leaks in.

    usage_gap   = predicted - actual log usage (> 0: has what the meta rewards,
                  but is used less than that predicts)
    performance = tournament win rate, top-cut conversion and ladder rating
                  lift (signals.py), for Pokemon with enough tournament data

Each Pokemon's top reasons are the ridge model's largest feature contributions.

The backtests score one regulation and check which score predicts usage
change into the next, beyond regression to the mean. So far tournament
performance does (Spearman ~ +0.13 to +0.17) and usage_gap does not: the
matchup features explain usage well but miss mechanics such as Unburden,
Armor Tail or Last Respects. So the report has two lists:

    proven      performance among Pokemon under POPULAR_USAGE (the validated
                signal), with the matchup model's reasons
    potential   large usage_gap with little tournament data: a speculative
                watchlist of Pokemon that look strong on paper but are untested

Usage:
    python analysis/model.py                         # M-C, plus backtests
    python analysis/model.py --regulation M-B --month 2026-08
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import signals  # noqa: E402
from champcalc import Calculator  # noqa: E402

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"

# (regulation, month) snapshots: the last full ladder month of each regulation.
SNAPSHOTS = {"M-A": "2026-05", "M-B": "2026-08", "M-C": "2026-09"}
# Backtests: features at the first snapshot vs usage at the second.
BACKTESTS = [(("M-A", "2026-05"), ("M-B", "2026-07")), (("M-B", "2026-08"), ("M-C", "2026-09"))]

USAGE_FLOOR = 1e-4          # Pokemon below the 1760 stats' cutoff
MIN_EXPECTED_USAGE = 0.005  # only flag Pokemon the model expects to see at least this much
POPULAR_USAGE = 0.10        # at or above this ladder usage a Pokemon is not "underused"
MIN_CHECK_GAMES = 10
CALC_NAMES = {"Aegislash": "Aegislash-Both"}


def key(name: str) -> str:
    """One spelling per Pokemon across sources (ids; Aegislash forms merged)."""
    k = re.sub(r"[^a-z0-9]", "", name.lower())
    return "aegislash" if k.startswith("aegislash") else k


def ladder_usage(db, regulation: str, month: str) -> pd.Series:
    rows = db.execute(
        """SELECT pokemon, usage FROM showdown_usage
           WHERE regulation=? AND month=? AND ladder='bo1' AND rating=1760""", (regulation, month)).fetchall()
    s = pd.Series({key(p): u for p, u in rows}, dtype=float)
    return s.groupby(level=0).sum()


def counter_pressure(db, regulation: str, month: str, meta_usage: pd.Series) -> pd.Series:
    """Usage-weighted share of ladder games where the meta's Pokemon beat it."""
    rows = pd.read_sql_query(
        """SELECT pokemon, opponent, n, p FROM showdown_checks
           WHERE regulation=? AND month=? AND ladder='bo1' AND rating=0 AND n >= ?""",
        db, params=(regulation, month, MIN_CHECK_GAMES))
    rows["pokemon"] = rows["pokemon"].map(key)
    rows["w"] = rows["opponent"].map(key).map(meta_usage).fillna(0)
    rows = rows[rows["w"] > 0]
    g = rows.assign(wp=rows["w"] * rows["p"]).groupby("pokemon")
    return g["wp"].sum() / g["w"].sum()


def load_features(regulation: str, month: str, db) -> pd.DataFrame:
    path = OUT / f"meta_matchups_{regulation}_{month}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run: python analysis/meta_matchups.py --regulation {regulation} "
                         f"--month {month}")
    df = pd.read_csv(path)
    df["key"] = df["pokemon"].map(key)
    df = df.drop_duplicates("key").set_index("key")
    calc = Calculator(regulation)
    info = {key(n): calc.species(n) for n in df["pokemon"]}
    df["is_mega"] = df["pokemon"].str.contains("-Mega").astype(int)
    df["base_stat_total"] = [sum(info[k]["base_stats"].values()) for k in df.index]
    usage = ladder_usage(db, regulation, month)
    df["usage"] = usage.reindex(df.index).fillna(0)
    meta = usage[usage >= 0.01]
    cp = counter_pressure(db, regulation, month, meta)
    df["counter_pressure"] = cp.reindex(df.index)
    df["counter_pressure_missing"] = df["counter_pressure"].isna().astype(int)
    # M-A's ladder stats have no checks and counters at all; use a neutral value.
    df["counter_pressure"] = df["counter_pressure"].fillna(cp.median() if len(cp) else 0.5)
    for c in df.columns:
        if c.startswith("util_"):
            df[c] = df[c].astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    # speed_trick_room is ~1 - speed_outspeeds; keeping both only splits one effect in two.
    return [c for c in df.columns
            if c.startswith(("off_", "def_", "speed_", "util_")) and c not in ("speed", "speed_trick_room")
            ] + ["is_mega", "base_stat_total", "counter_pressure", "counter_pressure_missing"]


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))


def fit(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    X = df[feature_columns(df)].astype(float).values
    y = np.log(df["usage"].clip(lower=USAGE_FLOOR)).values
    scores = {}
    for name, model in (("ridge", ridge()),
                        ("boosted", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                                  max_iter=300, random_state=seed))):
        preds = []
        for rep in range(5):
            preds.append(cross_val_predict(model, X, y, cv=KFold(10, shuffle=True, random_state=seed + rep)))
        pred = np.mean(preds, axis=0)
        scores[name] = (1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pred)
    best = max(scores, key=lambda n: scores[n][0])
    out = df.copy()
    out["log_usage"] = y
    out["predicted_log_usage"] = scores[best][1]
    out["expected_usage"] = np.exp(out["predicted_log_usage"])
    out["usage_gap"] = out["predicted_log_usage"] - out["log_usage"]

    # Explanations from the ridge model, fit on everything.
    model = ridge().fit(X, y)
    scaler, reg = model[0], model[1]
    contributions = scaler.transform(X) * reg.coef_
    cols = feature_columns(df)
    out["reasons"] = [
        ", ".join(f"{cols[j]} {contributions[i, j]:+.2f}" for j in np.argsort(-contributions[i])[:3])
        for i in range(len(out))
    ]
    coefs = pd.Series(reg.coef_, index=cols).sort_values()
    return out, {"r2": {n: round(s[0], 3) for n, s in scores.items()}, "model": best,
                 "alpha": reg.alpha_, "coefficients": coefs}


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) or 1)


def score(regulation: str, month: str, db) -> tuple[pd.DataFrame, dict]:
    df, meta = fit(load_features(regulation, month, db))
    perf = signals.signals(db, regulation)
    if not perf.empty:
        perf.index = perf.index.map(key)
        perf = perf[~perf.index.duplicated()]
        df = df.join(perf[["win_rate", "conversion", "ladder_lift", "performance"]], how="left")
    else:
        df["performance"] = np.nan
    gap_z = zscore(df["usage_gap"])
    perf_z = zscore(df["performance"])
    df["undervalued"] = np.where(df["performance"].notna(), (gap_z + perf_z) / 2, gap_z)
    return df.sort_values("undervalued", ascending=False), meta


def spearman(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def backtest(db, before: tuple[str, str], after: tuple[str, str]) -> dict:
    df, _ = score(*before, db)
    later = ladder_usage(db, *after)  # every Pokemon legal before stays legal after
    df["later_log_usage"] = np.log(later.reindex(df.index).fillna(0).clip(lower=USAGE_FLOOR))
    df["change"] = df["later_log_usage"] - df["log_usage"]
    slope, intercept = np.polyfit(df["log_usage"], df["change"], 1)
    df["change_beyond_reversion"] = df["change"] - (slope * df["log_usage"] + intercept)
    flagged = df[df["expected_usage"] >= MIN_EXPECTED_USAGE]
    with_perf = df[df["performance"].notna()]
    result = {"n": len(df), "n_tournament": len(with_perf)}
    for name, frame, column in (("usage_gap", df, "usage_gap"),
                                ("performance", with_perf, "performance"),
                                ("undervalued", with_perf, "undervalued")):
        top = frame.nlargest(20, column)
        result[name] = {
            "spearman": spearman(frame[column], frame["change_beyond_reversion"]),
            "top20_median": top["change_beyond_reversion"].median(),
            "top20_rose": (top["change_beyond_reversion"] > 0).mean(),
            "top": list(top["pokemon"].head(8)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regulation", default="M-C", choices=tuple(SNAPSHOTS))
    parser.add_argument("--month", help="default: the regulation's snapshot month")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    db = sqlite3.connect(DB)
    month = args.month or SNAPSHOTS[args.regulation]

    df, meta = score(args.regulation, month, db)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"undervalued_{args.regulation}_{month}.csv")
    print(f"{args.regulation} {month}: {len(df)} Pokemon. Cross-validated R^2 for log usage: {meta['r2']} "
          f"(using {meta['model']}; ridge alpha {meta['alpha']:.2f})")
    print("\nWhat the meta rewards (ridge coefficients, per standard deviation):")
    c = meta["coefficients"]
    print("  +", ", ".join(f"{k} {v:+.2f}" for k, v in c[::-1].head(8).items()))
    print("  -", ", ".join(f"{k} {v:+.2f}" for k, v in c.head(6).items()))

    proven = df[df["performance"].notna() & (df["usage"] < POPULAR_USAGE)].sort_values("performance", ascending=False)
    potential = df[df["performance"].isna() & (df["expected_usage"] >= MIN_EXPECTED_USAGE)]
    potential = potential.sort_values("usage_gap", ascending=False)
    proven.to_csv(OUT / f"proven_{args.regulation}_{month}.csv")
    potential.to_csv(OUT / f"potential_{args.regulation}_{month}.csv")
    cols = ["pokemon", "usage", "win_rate", "conversion", "ladder_lift", "performance", "reasons"]
    pcols = ["pokemon", "usage", "expected_usage", "usage_gap", "reasons"]
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 80,
                           "display.float_format", "{:.3f}".format):
        print(f"\nProven but underused (ladder usage under {POPULAR_USAGE:.0%}, by tournament performance):")
        print(proven[cols].head(args.top).to_string(index=False))
        print("\nUntested potential (little tournament data; model expects more usage):")
        print(potential[pcols].head(15).to_string(index=False))
        print("\nUnderperforming relative to popularity (2%+ usage, lowest performance):")
        print(df[(df["usage"] >= 0.02) & df["performance"].notna()].nsmallest(8, "performance")[cols]
              .to_string(index=False))

    for before, after in BACKTESTS:
        r = backtest(db, before, after)
        print(f"\nBacktest {before[0]} {before[1]} -> {after[0]} {after[1]} "
              f"({r['n']} Pokemon, {r['n_tournament']} with tournament data). "
              f"Does each score predict usage change beyond mean reversion?")
        for name in ("usage_gap", "performance", "undervalued"):
            x = r[name]
            print(f"  {name:12} Spearman {x['spearman']:+.3f}; top 20: median change {x['top20_median']:+.2f} "
                  f"log-usage, {x['top20_rose']:.0%} beat mean reversion  ({', '.join(x['top'][:5])})")


if __name__ == "__main__":
    main()
