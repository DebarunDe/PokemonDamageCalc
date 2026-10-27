"""Switch-in coverage: for the attacks that threaten each member of a team, is there a teammate that can take them?

The team-building tools score each Pokemon on its own (does it beat a threat, does it survive the threat's hit). They
never ask "when Archaludon is threatened, who comes in for that hit?". This scores it, with damage calcs:

    attacks   the damaging moves of the most used Pokemon (real tournament sets), each weighted by its user's usage
    threat    an attack doing 50%+ (max roll) to a member: it 2HKOs it or better
    covered   a teammate takes under 40% from that attack (40-60% counts half)
    score     the usage-weighted share of threatening attacks, over all members, that are covered

Calcs are doubles, no weather, against each species' most common complete tournament set (team-specific sets and
spreads are not used). `validate` tests whether the score says anything about results: teams' wins above
expectation beyond what their Pokemon's individual strengths predict (potential.py), in tournaments and on the
ladder separately.

    python analysis/switching.py validate
    python analysis/switching.py team teams/archaludon_rain.txt --members Archaludon Pelipper Venusaur Gengar
"""

from __future__ import annotations

import argparse
import re

import doubles_sim as ds
import interactions as ix
from champcalc import CalcError, Field

THREAT, SAFE, OKAY = 50.0, 40.0, 60.0
FIELD = Field(doubles=True)


def attacks(n=30):
    """(attacker Spec, move, weight) for the damaging moves of the n most used Pokemon; a Pokemon's weight is its
    usage, split evenly over its damaging moves."""
    bench = ix.Bench(n=n)
    out = []
    for spec, w in bench.opp:
        moves = [m for m in spec.moves if ds.calc.move(m)["category"] != "Status"]
        out += [(spec, m, w / bench.total / len(moves)) for m in moves]
    return out


def damage(defenders, atks, chunk=400):
    """{defender key: [max % from each attack]} for {key: Spec} defenders."""
    moves = sorted({m for _, m, _ in atks})
    index = {m: i for i, m in enumerate(moves)}
    by_attacker = {}
    for k, (spec, m, _) in enumerate(atks):
        by_attacker.setdefault(id(spec), (spec, []))[1].append(k)
    jobs, keys = [], []
    for key, d in defenders.items():
        for spec, ks in by_attacker.values():
            jobs.append((ix.pset(spec), ix.pset(d), FIELD, [index[atks[k][1]] for k in ks]))
            keys.append((key, ks))
    out = {key: [0.0] * len(atks) for key in defenders}
    for start in range(0, len(jobs), chunk):
        try:
            res = ds.calc.calculate_batch(moves, jobs[start:start + chunk])
        except CalcError:  # one bad set in the chunk: fall back to one job at a time
            res = []
            for job in jobs[start:start + chunk]:
                try:
                    res += ds.calc.calculate_batch(moves, [job])
                except CalcError:
                    res.append((1, [(0, 0)] * len(job[3])))
        for (key, ks), (hp, dmg) in zip(keys[start:start + chunk], res):
            for k, (_, hi) in zip(ks, dmg):
                out[key][k] = 100 * hi / hp
    return out


def score(rows, weights, threat=THREAT, safe=SAFE, okay=OKAY):
    """Switch-in coverage of a team from its members' damage rows (max % taken from each attack). Returns
    (score, threatened weight, list of (member index, attack index, best teammate index, % it takes))."""
    num = den = 0.0
    detail = []
    for i, row in enumerate(rows):
        for a, taken in enumerate(row):
            if taken < threat:
                continue
            mates = [(rows[j][a], j) for j in range(len(rows)) if j != i]
            best, j = min(mates) if mates else (999.0, -1)
            got = 1.0 if best < safe else 0.5 if best < okay else 0.0
            num += weights[a] * got
            den += weights[a]
            detail.append((i, a, j, best))
    return (num / den if den else 1.0), den, detail


def team_specs(names):
    """Specs for team member names (most common tournament set), skipping names without one."""
    return [s for s in (ds.real_spec(n) for n in names) if s is not None]


# ------------------------------------------------------------------------------------------------- validation

def validate(n=30):
    import numpy as np
    import potential as pt

    atks = attacks(n)
    weights = [w for _, _, w in atks]
    tour, lad = pt.tournaments(), pt.ladder()
    ft = pt.fit(tour, lam_p=1e12)
    to_form = {b: s for s, b in pt.ladder_names(ft).items()}  # ladder preview name -> dominant tournament form
    names = sorted({s for t in tour.teams for s in t} | {to_form.get(s, s) for t in lad.teams for s in t})
    specs = {nm: ds.real_spec(nm) for nm in names}
    specs = {k: v for k, v in specs.items() if v is not None}
    print(f"{len(atks)} attacks from the top {n}; damage for {len(specs)} species", flush=True)
    dmg = damage(specs, atks)

    def run(rows, label, rename=lambda s: s):
        f = pt.fit(rows, lam_p=1e12)
        out, clusters = [], []
        for t, y, g in zip(rows.teams, rows.y, rows.g):
            members = [rename(s) for s in t if rename(s) in dmg]
            if len(members) < 5:
                continue
            clusters.append(frozenset(t))
            resid = y - f["intercept"] - sum(f["species"][s] for s in t)
            grid = {(th, sf): score([dmg[m] for m in members], weights, th, sf, sf + 20)[0]
                    for th in (50, 70) for sf in (30, 40, 50)}
            out.append((grid, resid, g))
        print(f"\n{label}: {len(out)} teams; result above expectation beyond the Pokemon's individual strengths")
        print("  correlation of switch-in coverage with that result (threat cut / safe cut), and the slope in points per "
              "game for a 0 -> 1 change in coverage:")
        g = np.array([o[2] for o in out], dtype=float)
        r = np.array([o[1] for o in out])
        for key in out[0][0]:
            x = np.array([o[0][key] for o in out])
            xc, rc = x - np.average(x, weights=g), r - np.average(r, weights=g)
            sxx = np.sum(g * xc ** 2)
            b = np.sum(g * xc * rc) / sxx
            se = np.sqrt(np.sum(g ** 2 * xc ** 2 * 0.25 / g)) / sxx
            corr = np.sum(g * xc * rc) / np.sqrt(sxx * np.sum(g * rc ** 2))
            print(f"    threat {key[0]}%, safe {key[1]}%: r {corr:+.3f}  slope {100 * b:+6.1f} ± {100 * se:4.1f}  "
                  f"(coverage mean {np.average(x, weights=g):.2f}, sd {np.sqrt(np.average(xc ** 2, weights=g)):.2f})")
        x = np.array([o[0][(THREAT, SAFE)] for o in out])
        # copies of the same team are not independent: resample whole team compositions
        cluster = {}
        ids = np.array([cluster.setdefault(c, len(cluster)) for c in clusters])
        members = [np.flatnonzero(ids == c) for c in range(len(cluster))]
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(300):
            idx = np.concatenate([members[c] for c in rng.integers(0, len(members), len(members))])
            gg, xx, rr = g[idx], x[idx], r[idx]
            xc, rc = xx - np.average(xx, weights=gg), rr - np.average(rr, weights=gg)
            boot.append(np.sum(gg * xc * rc) / np.sum(gg * xc ** 2))
        print(f"  default cuts, slope with teams resampled by composition ({len(members)} distinct teams): "
              f"{100 * np.mean(boot):+.1f} ± {100 * np.std(boot):.1f}")
        order = np.argsort(x)
        print("  by fifth of coverage (default cuts): mean coverage -> result")
        for q in range(5):
            idx = order[q * len(order) // 5:(q + 1) * len(order) // 5]
            gg = g[idx]
            print(f"    {np.average(x[idx], weights=gg):.2f} -> {100 * np.average(r[idx], weights=gg):+5.2f} "
                  f"± {100 * (0.25 / gg.sum()) ** 0.5:.2f}")

    run(tour, "Tournament teams")
    run(lad, "Ladder sides", rename=lambda s: to_form.get(s, s))


# ------------------------------------------------------------------------------------------------- one team

def parse_paste(text):
    """Showdown-format team text -> list of Specs (SPs line for stat points)."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [x.strip() for x in block.splitlines() if x.strip() and not x.strip().startswith("#")]
        if not lines:
            continue
        name, _, item = lines[0].partition(" @ ")
        name, item = name.strip(), item.strip() or None
        ability, nature, sp, moves = None, "Hardy", {}, []
        for x in lines[1:]:
            if x.startswith("Ability:"):
                ability = x.split(":", 1)[1].strip()
            elif x.endswith("Nature"):
                nature = x.split()[0]
            elif x.startswith(("SPs:", "EVs:")):
                for part in x.split(":", 1)[1].split("/"):
                    v, stat = part.split()
                    sp[{"HP": "hp", "Atk": "atk", "Def": "def", "SpA": "spa", "SpD": "spd", "Spe": "spe"}[stat]] = int(v)
            elif x.startswith("- "):
                moves.append(x[2:].strip())
        form = next((f for f in ds.calc.species(name)["other_formes"]
                     if item and (ds.calc.species(f)["mega_stone"] or "").lower() == item.lower()), None)
        if form:  # holding its Mega Stone: score the Mega form
            out.append(ds.Spec(form, ds.calc.species(form)["abilities"][0], item, nature, sp, moves, pre_ability=ability))
        else:
            out.append(ds.Spec(name, ability, item, nature, sp, moves))
    return out


def report_team(specs, n=30):
    atks = attacks(n)
    weights = [w for _, _, w in atks]
    dmg = damage({i: s for i, s in enumerate(specs)}, atks)
    rows = [dmg[i] for i in range(len(specs))]
    sc, den, detail = score(rows, weights)
    print(f"Switch-in coverage of {', '.join(s.name for s in specs)}: {sc:.2f} "
          f"(threatening attacks weigh {den:.2f} usage-units)")
    per = {}
    for i, a, j, best in detail:
        per.setdefault(i, []).append((weights[a], a, j, best))
    for i, items in per.items():
        items.sort(reverse=True)
        bad = [x for x in items if x[3] >= SAFE]
        print(f"  {specs[i].name}: {len(items)} threatening attacks, {len(bad)} without a safe switch-in")
        for w, a, j, best in bad[:6]:
            spec, mv, _ = atks[a]
            print(f"     {spec.name} {mv} ({rows[i][a]:.0f}% to it): best switch-in {specs[j].name if j >= 0 else '-'} "
                  f"takes {best:.0f}%")
    return sc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--top", type=int, default=30)
    t = sub.add_parser("team")
    t.add_argument("paste")
    t.add_argument("--members", nargs="*", help="only these members (names as in the paste, base species names)")
    args = p.parse_args(argv)
    if args.cmd == "validate":
        validate(args.top)
    else:
        specs = parse_paste(open(args.paste).read())
        if args.members:
            specs = [s for s in specs if s.name.split("-Mega")[0] in args.members or s.name in args.members]
        report_team(specs)


if __name__ == "__main__":
    main()
