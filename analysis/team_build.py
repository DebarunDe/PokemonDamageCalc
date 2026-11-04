"""Theory team building around one Pokemon (no usage data): partners chosen by calcs against the whole legal roster.

Every legal Pokemon gets its best theory set from movesets.py (theory_movesets.csv: best 4 attacks, ability, item).
For the core Pokemon:
    walls     Pokemon its best set can't 2HKO
    threats   Pokemon whose best set OHKOs it, or 2HKOs it and moves first
Each candidate partner (non-Mega: the core holds the Mega) is scored on
    offence   share of the roster the team (core + partner) can OHKO / 2HKO, beyond the core alone
    walls     share of the core's walls the partner 2HKOs
    answers   share of the core's threats the partner survives (from full HP) and then 2HKOs
plus the support it can bring (Fake Out, Follow Me / Rage Powder, Tailwind, Icy Wind / Electroweb, Psychic Surge,
Intimidate, Wide Guard). Teams are then built greedily on offence + answers over the whole team.

    python analysis/team_build.py Medicham-Mega [--team-size 6]
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import movesets as mv
import theorycraft as tc
from champcalc import CalcError, Field, PokemonSet

SUPPORT = {"Fake Out": "Fake Out", "Follow Me": "redirection", "Rage Powder": "redirection",
           "Tailwind": "Tailwind", "Icy Wind": "speed drop", "Electroweb": "speed drop", "Trick Room": "Trick Room",
           "Wide Guard": "Wide Guard", "Helping Hand": "Helping Hand", "Parting Shot": "Parting Shot",
           "Will-O-Wisp": "Will-O-Wisp", "Snarl": "Snarl"}
SUPPORT_ABILITY = {"Intimidate": "Intimidate", "Psychic Surge": "Psychic Terrain", "Prankster": "Prankster",
                   "Friend Guard": "Friend Guard"}


def best_sets():
    """Best 4-attack theory set per Pokemon from theory_movesets.csv."""
    best = {}
    for r in csv.DictReader(open(tc.OUT / "theory_movesets.csv")):
        if r["k"] == "4" and (r["pokemon"] not in best or float(r["score"]) > float(best[r["pokemon"]]["score"])):
            best[r["pokemon"]] = r
    return best


def pset_of(r):
    t = tc.Theory(r["pokemon"], r["ability"], cat=r["category"])
    item = None if r["item"] == "(none)" else r["item"]
    t.item = item if "-Mega" not in r["pokemon"] else t.item
    return t, r["moves"].split(", ")


def damage_rows(att_t, moves, defenders):
    """[defender][move] average % damage from att_t's set."""
    f = Field(doubles=True)
    att = att_t.pset()
    good = []
    for m in moves:
        try:
            tc.calc.calculate_batch([m], [(att, defenders[0], f, [0])])
            good.append(m)
        except CalcError:
            pass
    if not good:
        return np.zeros((len(defenders), 1)), ["-"], np.ones(1)
    res = tc.calc.calculate_batch(good, [(att, d, f, list(range(len(good)))) for d in defenders])
    P = np.array([[100 * (lo + hi) / 2 / hp for lo, hi in dmg] for hp, dmg in res])
    acc = np.array([1.0 if att_t.ability == "No Guard" else tc.acc_of(m) for m in good])
    return P * 1.0, good, acc


def credit(P, acc):
    return (acc[None, :] * np.where(P >= 100, 1.0, np.where(P >= 50, 0.5, 0.0))).max(axis=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("core")
    p.add_argument("--team-size", type=int, default=6)
    p.add_argument("--threats", type=int, default=60)
    p.add_argument("--with", dest="fixed", nargs="*", default=[], help="members to include before the greedy fill")
    args = p.parse_args(argv)
    best = best_sets()
    defs, speeds = tc.bench()
    roster = tc.ROSTER
    core_t, core_moves = pset_of(best[args.core])
    core_set = core_t.pset()

    # the core's offence over the roster
    P, good, acc = damage_rows(core_t, core_moves, defs)
    core_off = credit(P, acc)
    walls = [d for d, v in zip(roster, core_off) if v == 0]

    # threats: every Pokemon's best set into the core
    print(f"{args.core}: {', '.join(core_moves)} @ {core_t.item}, Speed {core_t.speed}")
    sets = {}
    threat = []
    for s, r in best.items():
        t, mvs = pset_of(r)
        sets[s] = (t, mvs)
        Pc, g, a = damage_rows(t, mvs, [core_set])
        hit = float((a * Pc[0]).max())  # expected % of the best move
        spe = t.speed * (1.5 if t.item == "Choice Scarf" else 1)
        faster = spe > core_t.speed
        level = 1.0 if hit >= 100 else 0.5 if hit >= 50 else 0.0
        if level == 1.0 or (level == 0.5 and faster):
            threat.append((level + 0.5 * faster, hit, s, g[int(np.argmax(a * Pc[0]))], round(spe)))
    threat.sort(reverse=True)
    print(f"\nwalls ({len(walls)}): {', '.join(walls)}")
    print(f"threats: {len(threat)} Pokemon OHKO it, or 2HKO it and move first; the strongest:")
    for lvl, hit, s, m, spe in threat[:20]:
        print(f"  {s:22s} {m:16s} {hit:4.0f}%  Speed {spe}")
    threats = [x[2] for x in threat[:args.threats]]

    # candidates: non-Megas. Their offence on the roster, their walls coverage, and their answers to the threats
    cand = [s for s in best if "-Mega" not in s and s != args.core.split("-Mega")[0]]
    off = {}
    for s in cand:
        t, mvs = sets[s]
        Ps, g, a = damage_rows(t, mvs, defs)
        off[s] = credit(Ps, a)
    idx = {s: i for i, s in enumerate(roster)}
    wall_idx = [idx[w] for w in walls]
    thr_idx = [idx[x] for x in threats if x in idx]
    # survive: each threat's best hit into each candidate (theory set of the candidate)
    survive = {}
    for x in threats:
        t, mvs = sets[x]
        defenders = [sets[s][0].pset() for s in cand]
        Px, g, a = damage_rows(t, mvs, defenders)
        expected = (a[None, :] * Px).max(axis=1)
        for s, e in zip(cand, expected):
            survive[(x, s)] = e < 100
    answers = {s: np.array([float(survive[(x, s)]) * off[s][idx[x]] for x in threats]) for s in cand}

    def support(s):
        t, _ = sets[s]
        out = {SUPPORT[m] for m in tc.learn(s) if m in SUPPORT}
        out |= {SUPPORT_ABILITY[a] for a in tc.abilities(s) if a in SUPPORT_ABILITY}
        return sorted(out)

    rows = []
    for s in cand:
        team_off = np.maximum(core_off, off[s])
        rows.append({"pokemon": s, "offence_gain": float(team_off.mean() - core_off.mean()),
                     "walls": float(off[s][wall_idx].mean()) if wall_idx else 0.0,
                     "answers": float(answers[s].mean()), "speed": sets[s][0].speed,
                     "set": f"{sets[s][0].ability} @ {sets[s][0].item}: {', '.join(sets[s][1])}",
                     "support": ", ".join(support(s))})
    for r in rows:
        r["score"] = r["offence_gain"] + r["walls"] + r["answers"]
    rows.sort(key=lambda r: -r["score"])
    with open(tc.OUT / f"team_{args.core}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nbest partners (offence gain + share of walls it 2HKOs + share of threats it survives and 2HKOs):")
    for r in rows[:25]:
        print(f"  {r['score']:.2f}  {r['pokemon']:20s} gain {r['offence_gain']:.2f} walls {r['walls']:.2f} "
              f"answers {r['answers']:.2f} spe {r['speed']:3d}  [{r['support']}]  {r['set']}")
    sup = [r for r in rows if r["support"] and set(r["support"].split(", ")) & {"Fake Out", "redirection",
                                                                                 "Tailwind", "Intimidate"}]
    print("\nbest support partners (Fake Out / redirection / Tailwind / Intimidate):")
    for r in sup[:15]:
        print(f"  {r['score']:.2f}  {r['pokemon']:20s} walls {r['walls']:.2f} answers {r['answers']:.2f}  "
              f"[{r['support']}]")

    # greedy team: offence over the roster + answers to threats, with at most one Choice Scarf / one of each item
    team = [args.core]
    team_off = core_off.copy()
    team_ans = np.zeros(len(threats))
    for f in args.fixed:
        team.append(f)
        team_off = np.maximum(team_off, off[f])
        team_ans = np.maximum(team_ans, answers[f])
        print(f"  = {f:20s} team offence {team_off.mean():.2f}, threats answered {team_ans.mean():.2f}")
    while len(team) < args.team_size:
        best_s, best_v = None, -1
        for s in cand:
            if s in team or s.split("-")[0] in {x.split("-")[0] for x in team}:  # items are assigned after
                continue
            v = np.maximum(team_off, off[s]).mean() + np.maximum(team_ans, answers[s]).mean()
            if v > best_v:
                best_s, best_v = s, v
        team.append(best_s)
        team_off = np.maximum(team_off, off[best_s])
        team_ans = np.maximum(team_ans, answers[best_s])
        print(f"  + {best_s:20s} team offence {team_off.mean():.2f}, threats answered {team_ans.mean():.2f}")
    print(f"\ngreedy team: {', '.join(team)}")
    print(f"still walled by: {', '.join(d for d, v in zip(roster, team_off) if v == 0) or '-'}")
    print(f"threats nobody answers: {', '.join(x for x, v in zip(threats, team_ans) if v == 0) or '-'}")
    print("who answers each of the top threats:")
    for x in threats[:20]:
        who = [m for m in team[1:] if survive[(x, m)] and off[m][idx[x]] > 0]
        print(f"  {x:22s} {', '.join(who) or '-'}")


if __name__ == "__main__":
    main()
