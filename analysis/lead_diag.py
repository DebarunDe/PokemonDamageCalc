"""Where does the lead model miss? Compares its turn-1 play with what ladder players actually did on turn 1.

    1. actions: the model's equilibrium probability of each kind of turn-1 action (attack, Protect, Fake Out,
       Tailwind, switch, ...) for each lead, against how often players chose it (turn-1 lines of the replay log)
    2. by species: the leads whose modelled play is furthest from what players do
    3. by lead type: predicted vs logged turn-1 KOs dealt and taken, for sides whose leads have Fake Out, Tailwind,
       Trick Room, Intimidate, weather, terrain, redirection
    4. Mega timing: how often a leading Mega Evolves on turn 1 (the model assumes always)

    python analysis/lead_diag.py --games 2000 [--revealed]

Findings (2026-11, 1,947 games, two-turn model, 7,022 observed turn-1 actions):
  - revealed sets barely matter (leads.py validate --revealed: KO r +0.15 vs +0.16, result r +0.104 vs +0.095):
    the misses are in play, not in sets
  - the model switches far too much (23% of turn-1 actions vs 10%) and Protects a little too much (15% vs 13%)
  - it under-plays moves whose payoff comes after turn 2: Tailwind 1.9% vs 3.6% (Whimsicott Tailwinds 53% of the
    time, the model ~0), Trick Room 2.3% vs 3.7% (Farigiraf 48% vs 25%), setup 0.9% vs 4.3% (Gholdengo Nasty Plot
    26%), redirection 0.5% vs 1.9% (Indeedee-F Follow Me 41%: it protects the Trick Room setter)
  - 8% of turn-1 actions are status moves it doesn't model (Encore, Parting Shot, Spore, Taunt, Will-O-Wisp, ...)
  - KOs are under-predicted overall (0.14 vs 0.21 per side), most for weather setters (dealt 0.15 vs 0.29); sides
    with Fake Out or Intimidate take twice the KOs the model expects (0.09 vs 0.18, 0.08 vs 0.17)
  - a leading Mega Evolves on turn 1 in 73% of games, not always
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import numpy as np

import leads as ld

KINDS = ["attack", "Protect", "Fake Out", "Tailwind", "Trick Room", "setup", "redirect", "Wide Guard",
         "Helping Hand", "other status", "switch"]
ACTION_KIND = {"fake": "Fake Out", "protect": "Protect", "tailwind": "Tailwind", "trickroom": "Trick Room",
               "setup": "setup", "redirect": "redirect", "wideguard": "Wide Guard", "helping": "Helping Hand",
               "switch": "switch", "move": "attack", "pass": "other status",
               "status": "other status"}


def move_kind(m):
    if m in ld.PROTECT:
        return "Protect"
    if m in ("Fake Out", "Tailwind", "Trick Room", "Wide Guard", "Helping Hand"):
        return m
    if m in ld.SETUP:
        return "setup"
    if m in ld.REDIRECT:
        return "redirect"
    info = ld.move_info(m)
    return "other status" if info and info["category"] == "Status" else "attack"


def _detail(args):
    """Worker: value, expected KOs, and each lead's probability of each kind of action."""
    try:
        v, ka, kb, d = ld.matchup(*args)
    except Exception:
        return None
    out = []
    for acts, p in ((d["acts_a"], d["px"]), (d["acts_b"], d["py"])):
        marg = [Counter(), Counter()]
        for joint, q in zip(acts, p):
            for slot in (0, 1):
                marg[slot][ACTION_KIND[joint[slot][0]]] += q
        out.append([dict(m) for m in marg])
    return v, ka, kb, out


def turn1(log):
    """Turn-1 facts per side: {'p1': {'species': {slot: species}, 'kind': {slot: kind}, 'mega': bool}, ...}.
    Slot letters come from the opening switch lines; switches before the end of turn 1 count as 'switch'."""
    from replays import base
    out = {s: {"species": {}, "kind": {}, "mega": False} for s in ("p1", "p2")}
    phase = "pre"
    for line in log.split("\n"):
        parts = line.split("|")
        if line.startswith("|turn|1"):
            phase = "t1"
            continue
        if line.startswith("|turn|2") or (phase == "t1" and line.startswith("|upkeep")):
            break
        if len(parts) < 3 or not re.match(r"p[12][ab]:", parts[2]):
            continue
        side, slot = parts[2][:2], parts[2][2]
        if phase == "pre" and parts[1] == "switch":
            out[side]["species"][slot] = base(parts[3].split(",")[0])
        elif phase == "t1":
            if parts[1] == "move" and not any(p.startswith("[from]") for p in parts[4:]):
                out[side]["kind"].setdefault(slot, move_kind(parts[3]))
            elif parts[1] in ("switch", "drag") and slot not in out[side]["kind"]:
                out[side]["kind"][slot] = "switch"
            elif parts[1] == "-mega":
                out[side]["mega"] = True
    return out


def features(specs, weather_terrain):
    f = set()
    moves = {m for s in specs for m in s.moves}
    for m in ("Fake Out", "Tailwind", "Trick Room"):
        if m in moves:
            f.add(m)
    if moves & ld.REDIRECT:
        f.add("redirection")
    if any(s.pre_ability == "Intimidate" for s in specs):
        f.add("Intimidate")
    if any(s.pre_ability in ld.WEATHER or s.ability in ld.WEATHER for s in specs):
        f.add("sets weather")
    if any(s.pre_ability in ld.TERRAIN or s.ability in ld.TERRAIN for s in specs):
        f.add("sets terrain")
    if any("-Mega" in s.name for s in specs):
        f.add("Mega lead")
    return f


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--revealed", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--horizon", type=int, default=ld.HORIZON)
    p.add_argument("--turn2", type=float, default=ld.TURN2)
    p.add_argument("--switch-cost", type=float, default=ld.SWITCH_COST)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    ld.TURN2, ld.HORIZON, ld.SWITCH_COST = args.turn2, args.horizon, args.switch_cost
    from replays import base
    db, sides, jobs = ld.load_games(args.games, seed=args.seed, revealed=args.revealed)
    args_of = dict(jobs)
    pred, act, n_obs = Counter(), Counter(), 0
    by_species = defaultdict(lambda: [Counter(), Counter(), 0])
    groups = defaultdict(lambda: [[], [], [], []])  # feature -> predicted KOs dealt, logged dealt, pred taken, taken
    mega = Counter()
    done = 0
    for rid, (v, ka, kb, marg) in ld.run_jobs(jobs, args.workers, fn=_detail):
        log = db.execute("SELECT log FROM replays WHERE id = ?", (rid,)).fetchone()[0]
        t1 = turn1(log)
        faints = ld.turn1_faints(log)
        a, b = args_of[rid][0], args_of[rid][1]
        for si, side in enumerate(("p1", "p2")):
            specs, info = (a, b)[si], t1[side]
            names = [base(s.name.split("-Mega")[0]) for s in specs]
            for letter, kind in info["kind"].items():
                sp = info["species"].get(letter)
                idx = names.index(sp) if sp in names else "ab".index(letter)
                m = marg[si][idx]
                pred.update(m)
                act[kind] += 1
                n_obs += 1
                bs = by_species[specs[idx].name]
                bs[0].update(m)
                bs[1][kind] += 1
                bs[2] += 1
            mine, theirs = (ka, kb) if si == 0 else (kb, ka)
            dealt, taken = faints["p2" if si == 0 else "p1"], faints[side]
            for f in features(specs, None) | {"all"}:
                g = groups[f]
                g[0].append(mine), g[1].append(dealt), g[2].append(theirs), g[3].append(taken)
            if any("-Mega" in s.name for s in specs):
                mega[info["mega"]] += 1
        done += 1
        if done % 500 == 0:
            print(f"  {done} games", flush=True)

    print(f"\n{done} games, {n_obs} observed turn-1 lead actions ({'revealed' if args.revealed else 'tournament'} "
          "sets)")
    print("1. Turn-1 action kinds: model (equilibrium probability) vs players")
    for k in KINDS:
        if pred[k] or act[k]:
            print(f"   {k:13s} model {100 * pred[k] / n_obs:5.1f}%   players {100 * act[k] / n_obs:5.1f}%")
    print("2. Leads whose modelled play is furthest from players' (100+ observed turn-1 actions; total variation "
          "distance)")
    rows = []
    for name, (pm, am, n) in by_species.items():
        if n >= 100:
            tv = 0.5 * sum(abs(pm[k] / n - am[k] / n) for k in KINDS)
            rows.append((tv, name, pm, am, n))
    for tv, name, pm, am, n in sorted(rows, reverse=True)[:15]:
        top = lambda c: ", ".join(f"{k} {100 * c[k] / n:.0f}%" for k, _ in c.most_common(3))
        print(f"   {name:22s} {tv:.2f}  model: {top(pm)}  |  players: {top(am)}  ({n})")
    print("3. Turn-1 KOs by the side's lead type: predicted vs logged, dealt and taken")
    for f, (pd, ad, pt, at) in sorted(groups.items(), key=lambda kv: -len(kv[1][0])):
        if len(pd) >= 100:
            r = np.corrcoef(np.array(pd) - pt, np.array(ad) - at)[0, 1]
            print(f"   {f:13s} {len(pd):5d} sides  dealt {np.mean(pd):.2f} vs {np.mean(ad):.2f}   taken "
                  f"{np.mean(pt):.2f} vs {np.mean(at):.2f}   KO-difference r {r:+.2f}")
    if mega:
        print(f"4. Sides leading a Mega: it Mega Evolved on turn 1 in {mega[True]} of {sum(mega.values())} "
              f"({100 * mega[True] / sum(mega.values()):.0f}%)")


if __name__ == "__main__":
    main()
