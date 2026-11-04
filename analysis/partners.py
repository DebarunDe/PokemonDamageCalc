"""Partners for one fixed set, by theory (no usage data): what the set can't hit, what threatens it, who fixes both.

The core set is given exactly (moves, item state, spread, boosts, terrain). Against every legal Pokemon's best theory
set (theory_movesets.csv):
    walls     Pokemon the core can't 2HKO with its attacks, at +0 and after its setup boost
    threats   Pokemon whose best hit does 50%+ to the core (it takes a hit on its setup turn), in the given terrain
Each candidate partner (its best theory set) is scored on
    walls     share of the core's walls (after setup) it 2HKOs
    answers   share of the threats it survives from full HP and 2HKOs
    shared    the core's weaknesses it shares (Psychic / Ground / Flying for Sneasler)
    support   Fake Out, Follow Me / Rage Powder, Intimidate, Wide Guard, the terrain the core's seed needs
Grassy Terrain halves Earthquake / Bulldoze; terrain boosts (x1.3 for grounded attackers) come from the calc.

    python analysis/partners.py Sneasler --moves "Close Combat" "Dire Claw" --boost atk=2 --seed "Grassy Seed"
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import team_build as tb
import team_roles as tr
import theorycraft as tc
from champcalc import CalcError, Field, PokemonSet

SEED = {"Grassy Seed": ("Grassy", "def", {"Grassy Surge"}), "Electric Seed": ("Electric", "def", {"Electric Surge"}),
        "Psychic Seed": ("Psychic", "spd", {"Psychic Surge"}), "Misty Seed": ("Misty", "spd", {"Misty Surge"})}
HALVED_IN_GRASSY = {"Earthquake", "Bulldoze", "Magnitude"}
SUPPORT = {"Fake Out": "Fake Out", "Follow Me": "Follow Me", "Rage Powder": "Rage Powder", "Wide Guard": "Wide Guard",
           "Tailwind": "Tailwind", "Icy Wind": "Icy Wind", "Helping Hand": "Helping Hand",
           "Parting Shot": "Parting Shot",
           "Taunt": "Taunt", "Quick Guard": "Quick Guard"}


def hits(att, moves, defenders, field):
    """[defender][move] average % damage, the moves the calc accepted."""
    good = []
    for m in moves:
        try:
            tc.calc.calculate_batch([m], [(att, defenders[0], field, [0])])
            good.append(m)
        except CalcError:
            pass
    if not good:
        return np.zeros((len(defenders), 1)), ["-"]
    res = tc.calc.calculate_batch(good, [(att, d, field, list(range(len(good)))) for d in defenders])
    return np.array([[100 * (lo + hi) / 2 / hp for lo, hi in dmg] for hp, dmg in res]), good


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pokemon")
    p.add_argument("--moves", nargs="+", required=True, help="the core's attacks")
    p.add_argument("--ability", default=None)
    p.add_argument("--nature", default="Adamant")
    p.add_argument("--sp", default="hp=32,atk=32,spe=2")
    p.add_argument("--boost", default="atk=2", help="its setup boost, e.g. atk=2")
    p.add_argument("--seed", choices=sorted(SEED), default=None)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--acc-exp", type=float, default=1.0,
                   help="accuracy exponent for our side's KO credit (2: lean towards consistent moves); partner "
                        "sets come from movesets.py run with the same --acc-exp. Threats stay at expected value")
    args = p.parse_args(argv)
    s = args.pokemon
    sp = {k: int(v) for k, v in (x.split("=") for x in args.sp.split(","))}
    boost = {k: int(v) for k, v in (x.split("=") for x in args.boost.split(","))} if args.boost else {}
    terrain, seed_stat, setters = SEED[args.seed] if args.seed else (None, None, set())
    ability = args.ability or ("Unburden" if "Unburden" in tc.abilities(s) else tc.abilities(s)[0])
    def_boost = {seed_stat: 1} if seed_stat else {}
    field = Field(doubles=True, terrain=terrain)
    defs, _ = tc.bench()
    roster = tc.ROSTER
    best = tb.best_sets()  # opponents: expected value
    ours = best if args.acc_exp == 1.0 else tb.best_sets(f"theory_movesets_acc{args.acc_exp:g}.csv")
    ae = args.acc_exp

    # the core's offence, +0 and after setup (item used: no item)
    core0 = PokemonSet(s, nature=args.nature, ability=ability, sp=sp)
    core2 = PokemonSet(s, nature=args.nature, ability=ability, sp=sp, boosts=boost)
    acc = np.array([tc.acc_of(m) for m in args.moves])
    P0, g0 = hits(core0, args.moves, defs, field)
    P2, _ = hits(core2, args.moves, defs, field)
    cred = lambda P: (acc[None, :len(g0)] ** ae * np.where(P >= 100, 1, np.where(P >= 50, .5, 0))).max(axis=1)
    c0, c2 = cred(P0), cred(P2)
    walls = [i for i, v in enumerate(c2) if v == 0]
    print(f"{s} ({ability}, {', '.join(args.moves)}; {args.sp}; seed {args.seed or '-'}, terrain {terrain or '-'})")
    print(f"  +0: power {c0.mean():.3f}, OHKO {(c0 > .5).mean():.0%}; after {args.boost}: power {c2.mean():.3f}, "
          f"OHKO {(c2 > .5).mean():.0%}")
    print(f"  walls even after setup ({len(walls)}): {', '.join(roster[i] for i in walls)}")

    # threats into the core: seed boost on, in the terrain
    me = PokemonSet(s, nature=args.nature, ability=ability, sp=sp, boosts=def_boost)
    sets = {o: tb.pset_of(r) for o, r in best.items()}
    threat = []
    for o, (ot, mvs) in sets.items():
        Pc, g = hits(ot.pset(), mvs, [me], field)
        a = np.array([tc.acc_of(m) for m in g])
        h = Pc[0] * a
        if terrain == "Grassy":
            h = h * np.array([0.5 if m in HALVED_IN_GRASSY else 1.0 for m in g])
        k = int(np.argmax(h))
        if h[k] >= 50:
            threat.append((float(h[k]), o, g[k]))
    threat.sort(reverse=True)
    from collections import Counter
    print(f"  threats (50%+ to it, seed boost and terrain on): {len(threat)}; by move type: " + ", ".join(
        f"{t} {n}" for t, n in Counter(tc.minfo(m)["type"] for _, _, m in threat).most_common(6)))
    weak = tr.weaknesses(s, ability)

    # candidates; each threat's best hit into every candidate, one batch per threat
    rows = []
    thr = [o for _, o, _ in threat]
    oursets = sets if ours is best else {o: tb.pset_of(r) for o, r in ours.items()}
    cands = [c for c in oursets if c.split("-")[0] != s.split("-")[0]]
    cdefs = [oursets[c][0].pset() for c in cands]
    survive = {}
    for o in thr:
        ot, om = sets[o]
        Px, gx = hits(ot.pset(), om, cdefs, field)
        ax = np.array([tc.acc_of(m) for m in gx])
        hx = Px * ax[None, :]
        if terrain == "Grassy":
            hx = hx * np.array([0.5 if m in HALVED_IN_GRASSY else 1.0 for m in gx])[None, :]
        for c, v in zip(cands, hx.max(axis=1)):
            survive[(o, c)] = v < 100
    # strict test for the walls: the partner moves first and KOs it that turn, alone (clean) or with the core's
    # hit after setup (pair); wall Speed from its best theory set (Choice Scarf x1.5)
    core_hit = P2.max(axis=1)
    wall_speed = {i: sets[roster[i]][0].speed * (1.5 if sets[roster[i]][0].item == "Choice Scarf" else 1)
                  for i in walls if roster[i] in sets}
    for c in cands:
        ct, cm = oursets[c]
        Pc, g = hits(ct.pset(), cm, defs, field)
        a = np.array([tc.acc_of(m) for m in g])
        off = (a[None, :] ** ae * np.where(Pc >= 100, 1, np.where(Pc >= 50, .5, 0))).max(axis=1)
        wall_cov = float(np.mean([off[i] > 0 for i in walls])) if walls else 0.0
        cspe = ct.speed * (1.5 if ct.item == "Choice Scarf" else 1)
        pri = np.array([tc.minfo(m)["priority"] > 0 for m in g])
        clean, pair, missed = [], [], []
        for i in walls:
            first = (cspe > wall_speed.get(i, 999)) | pri  # per move: faster, or a priority move
            c1 = float(np.max(a ** ae * (Pc[i] >= 100) * first))
            c2_ = float(np.max(a ** ae * (Pc[i] + core_hit[i] >= 100) * first))
            clean.append(c1)
            pair.append(c2_)
            if c2_ == 0:
                missed.append(roster[i])
        # survive each threat's best hit (partner's theory set), then 2HKO it
        surv = [float(survive[(o, c)]) * (off[roster.index(o)] > 0) for o in thr]
        sup = sorted({SUPPORT[m] for m in tc.learn(c) if m in SUPPORT}
                     | ({"Intimidate"} if "Intimidate" in tc.abilities(c) else set())
                     | ({f"sets {terrain} Terrain"} if set(tc.abilities(c)) & setters else set()))
        cw = tr.weaknesses(c, ct.ability)
        rows.append({"partner": c, "walls": round(wall_cov, 3), "clean": round(float(np.mean(clean)), 3),
                     "pair": round(float(np.mean(pair)), 3), "missed": ", ".join(missed),
                     "answers": round(float(np.mean(surv)), 3),
                     "pair_offence": round(float(np.maximum(c2, off).mean()), 3),
                     "shared": ", ".join(sorted(weak & cw)),
                     "resists": ", ".join(t for t in sorted(weak) if tr.IMMUNE_ABILITY.get(ct.ability) == t or
                                          np.prod([tr.CHART[t].get(x, 1.0) for x in tc.info(c)["types"]]) < 1),
                     "support": ", ".join(sup), "speed": ct.speed,
                     "set": f"{ct.ability} @ {ct.item}: {', '.join(cm)}",
                     "setter": bool(set(tc.abilities(c)) & setters)})
    for r in rows:
        r["score"] = round(r["pair"] + r["answers"] - 0.15 * len([x for x in r["shared"].split(", ") if x])
                           + 0.05 * len([x for x in r["support"].split(", ") if x]), 3)
    rows.sort(key=lambda r: -r["score"])
    with open(tc.OUT / f"partners_{s}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if setters:
        print(f"\nterrain setters for the {args.seed}:")
        for r in [r for r in rows if r["setter"]]:
            print(f"  {r['partner']:18s} clean {r['clean']:.2f} pair {r['pair']:.2f} (2HKO {r['walls']:.2f}) "
                  f"answers {r['answers']:.2f} team {r['pair_offence']:.2f} "
                  f"shared [{r['shared']}] resists [{r['resists']}] support [{r['support']}]  {r['set']}")
    print("\nbest partners (pair: share of the core's walls it moves first on and KOs that turn, alone or with the "
          "core's +2 hit; + answers - 0.15 per shared weakness + 0.05 per support option):")
    for r in rows[:args.top]:
        print(f"  {r['score']:.2f} {r['partner']:18s} clean {r['clean']:.2f} pair {r['pair']:.2f} (2HKO "
              f"{r['walls']:.2f}) answers {r['answers']:.2f} spe {r['speed']} shared [{r['shared']}] "
              f"[{r['support']}]  {r['set']}")
        if r["missed"]:
            print(f"        misses: {r['missed']}")


if __name__ == "__main__":
    main()
