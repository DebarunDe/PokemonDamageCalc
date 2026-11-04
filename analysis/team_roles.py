"""Theory team building with the roles teambuilding guides ask for (no usage data).

team_build.py picks partners by calcs alone, so it never asks for speed control or Fake Out. This adds the checklist
the guides agree on (Stratagem, the Champions guide on BlueStacks, MysticVGC's checklist), as requirements:

    speed     two members with speed control: Tailwind, Icy Wind / Electroweb, Thunder Wave, a Choice Scarf
              (Trick Room only when the core is slow)
    fakeout   Fake Out or redirection (Follow Me / Rage Powder)
    pivot     Intimidate or redirection
    priority  a priority attack (Fake Out doesn't count)
    weakness  no type weakness shared by three members (immunity abilities count)
    items     one item each (assigned after)

Each candidate gets role versions of its set, following the guides' move-slot outlines:
    attacker  its best 3 attacks + Protect
    support   its best 2 attacks + one utility move (speed control, Fake Out, redirection or a priority attack)
              + Protect
Offence and answers are the team_build.py measures (share of the roster the team 2HKOs; share of the core's
threats a member survives and 2HKOs). A beam search finds the best team that meets every requirement, and the run
is repeated with each requirement dropped to show what it costs.

    python analysis/team_roles.py Medicham-Mega
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations

import numpy as np

import team_build as tb
import theorycraft as tc

SPEED_MOVES = {"Tailwind", "Icy Wind", "Electroweb", "Thunder Wave", "Trick Room", "Bleakwind Storm"}
REDIRECT = {"Follow Me", "Rage Powder"}
PRIORITY = {"Bullet Punch", "Aqua Jet", "Sucker Punch", "Extreme Speed", "Grassy Glide", "Shadow Sneak", "Ice Shard",
            "Mach Punch", "Vacuum Wave", "Accelerock", "Jet Punch", "Quick Attack", "Thunderclap", "Upper Hand"}
TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic",
         "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
# attacking type -> {defending type: multiplier} (only entries that aren't 1)
CHART = {
    "Normal": {"Rock": .5, "Ghost": 0, "Steel": .5},
    "Fire": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": .5, "Dragon": .5, "Steel": 2},
    "Water": {"Fire": 2, "Water": .5, "Grass": .5, "Ground": 2, "Rock": 2, "Dragon": .5},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Ground": 0, "Flying": 2, "Dragon": .5},
    "Grass": {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5, "Ground": 2, "Flying": .5, "Bug": .5, "Rock": 2,
              "Dragon": .5, "Steel": .5},
    "Ice": {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": .5, "Flying": .5, "Psychic": .5, "Bug": .5, "Rock": 2, "Ghost": 0,
                 "Dark": 2, "Steel": 2, "Fairy": .5},
    "Poison": {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5, "Ghost": .5, "Steel": 0, "Fairy": 2},
    "Ground": {"Fire": 2, "Electric": 2, "Grass": .5, "Poison": 2, "Flying": 0, "Bug": .5, "Rock": 2, "Steel": 2},
    "Flying": {"Electric": .5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": .5, "Steel": .5},
    "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": .5, "Dark": 0, "Steel": .5},
    "Bug": {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5, "Flying": .5, "Psychic": 2, "Ghost": .5,
            "Dark": 2, "Steel": .5, "Fairy": .5},
    "Rock": {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5, "Flying": 2, "Bug": 2, "Steel": .5},
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon": {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark": {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5, "Fairy": .5},
    "Steel": {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2, "Rock": 2, "Steel": .5, "Fairy": 2},
    "Fairy": {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2, "Dark": 2, "Steel": .5},
}
IMMUNE_ABILITY = {"Levitate": "Ground", "Earth Eater": "Ground", "Flash Fire": "Fire", "Well-Baked Body": "Fire",
                  "Water Absorb": "Water", "Storm Drain": "Water", "Dry Skin": "Water", "Volt Absorb": "Electric",
                  "Lightning Rod": "Electric", "Motor Drive": "Electric", "Sap Sipper": "Grass"}
REQUIREMENTS = ["speed", "fakeout", "pivot", "priority", "weakness"]


@lru_cache(maxsize=None)
def weaknesses(s, ability):
    types = tc.info(s)["types"]
    out = set()
    for atk in TYPES:
        m = 1.0
        for d in types:
            m *= CHART[atk].get(d, 1.0)
        if IMMUNE_ABILITY.get(ability) == atk:
            m = 0
        if m >= 2:
            out.add(atk)
    return out


class Variant:
    def __init__(self, s, role, moves, offence, answers, roles, ability, item, speed):
        self.s, self.role, self.moves, self.off, self.ans = s, role, moves, offence, answers
        self.roles, self.ability, self.item, self.speed = roles, ability, item, speed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("core")
    p.add_argument("--threats", type=int, default=60)
    p.add_argument("--beam", type=int, default=60)
    p.add_argument("--unburden", choices=["white-herb", "grassy-seed", "electric-seed"],
                   help="core runs Unburden: its item is used up (no item damage, double Speed); seeds need their "
                        "terrain setter on the team")
    args = p.parse_args(argv)
    best = tb.best_sets()
    defs, speeds = tc.bench()
    roster = tc.ROSTER
    idx = {s: i for i, s in enumerate(roster)}
    core_t, core_moves = tb.pset_of(best[args.core])
    import csv
    want_item = "(none)" if args.unburden else best[args.core]["item"]
    k3 = [r for r in csv.DictReader(open(tc.OUT / "theory_movesets.csv")) if r["pokemon"] == args.core and r["k"] == "3"
          and r["item"] == want_item and (args.unburden or r["ability"] == best[args.core]["ability"])]
    seed_terrain = None
    if args.unburden:
        # item used up: damage as itemless (Acrobatics at full power), Speed doubled from then on
        core_t.ability, core_t.item = "Unburden", None
        core_t.speed = core_t.speed * 2
        seed_terrain = {"grassy-seed": {"Grassy Surge", "Seed Sower"},
                        "electric-seed": {"Electric Surge", "Hadron Engine"}}.get(args.unburden)
    if k3:
        core_moves = max(k3, key=lambda r: float(r["score"]))["moves"].split(", ")
    P, g, a = tb.damage_rows(core_t, core_moves, defs)
    core_off = tb.credit(P, a)
    slow_core = core_t.speed < float(np.median(speeds))

    # threats to the core (as in team_build.py)
    sets, threat = {}, []
    for s, r in best.items():
        t, mvs = tb.pset_of(r)
        sets[s] = (t, mvs)
        Pc, g2, a2 = tb.damage_rows(t, mvs, [core_t.pset()])
        hit = float((a2 * Pc[0]).max())
        faster = t.speed * (1.5 if t.item == "Choice Scarf" else 1) > core_t.speed
        if hit >= 100 or (hit >= 50 and faster):
            threat.append((hit + 100 * faster, s))
    threats = [s for _, s in sorted(threat, reverse=True)[:args.threats]]
    core_mega = "-Mega" in args.core  # one Mega per team: partners may include one only if the core isn't a Mega
    cand = [s for s in best if (not core_mega or "-Mega" not in s) and s.split("-")[0] != args.core.split("-")[0]]
    survive = {}
    for x in threats:
        t, mvs = sets[x]
        Px, g2, a2 = tb.damage_rows(t, mvs, [sets[s][0].pset() for s in cand])
        exp = (a2[None, :] * Px).max(axis=1)
        for s, e in zip(cand, exp):
            survive[(x, s)] = float(e < 100)
    thr_idx = [idx[x] for x in threats]

    # role variants
    variants = []
    for n, s in enumerate(cand):
        t, best4 = sets[s]
        learn = tc.learn(s)
        utility = sorted((SPEED_MOVES | REDIRECT | PRIORITY | {"Fake Out"}) & learn)
        if not slow_core:
            utility = [u for u in utility if u != "Trick Room"]
        attack_util = [u for u in utility if tc.minfo(u) and tc.minfo(u)["category"] != "Status" and u != "Fake Out"]
        pool = list(dict.fromkeys(best4 + attack_util))
        Pm, good, acc = tb.damage_rows(t, pool, defs)
        C = acc[None, :] * np.where(Pm >= 100, 1.0, np.where(Pm >= 50, 0.5, 0.0))
        col = {m: j for j, m in enumerate(good)}
        att = [m for m in best4 if m in col]

        def vec(ms):
            js = [col[m] for m in ms if m in col]
            return C[:, js].max(axis=1) if js else np.zeros(len(defs))
        base_roles = set()
        if "Intimidate" in tc.abilities(s):
            base_roles.add("pivot")
        ability = "Intimidate" if "pivot" in base_roles else t.ability

        def add(role, moves, extra_roles):
            off = vec(moves)
            ans = np.array([survive[(x, s)] * off[idx[x]] for x in threats])
            roles = set(base_roles) | extra_roles
            variants.append(Variant(s, role, moves, off, ans, roles, ability, t.item, t.speed))
        if t.item == "Choice Scarf":  # Scarf: four attacks, no Protect
            add("attacker (Scarf)", att[:4], {"speed"})
        else:
            add("attacker", att[:3] + ["Protect"], set())
        # best 2 attacks of its 4 + a utility move
        pairs = sorted(combinations(att, 2), key=lambda pr: -float(vec(list(pr)).mean())) if len(att) >= 2 else [
            tuple(att)]
        two = list(pairs[0]) if pairs else []
        for u in utility:
            r = set()
            if u in SPEED_MOVES:
                r.add("speed")
            if u in REDIRECT:
                r |= {"fakeout", "pivot"}
            if u == "Fake Out":
                r.add("fakeout")
            if u in PRIORITY:
                r.add("priority")
            moves = two + [u] + ["Protect"]
            add(f"support ({u})", moves, r)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1} / {len(cand)} candidates", flush=True)
    print(f"{len(variants)} role versions of {len(cand)} candidates")

    core_weak = weaknesses(args.core, core_t.ability)

    def met(team):
        roles = [v.roles for v in team]
        speed = sum("speed" in r for r in roles) + (1 if core_t.item == "Choice Scarf" or args.unburden else 0)
        out = {"speed": speed >= 2, "fakeout": any("fakeout" in r for r in roles),
               "pivot": any("pivot" in r for r in roles), "priority": any("priority" in r for r in roles)}
        weak = [core_weak] + [weaknesses(v.s, v.ability) for v in team]
        out["weakness"] = all(sum(t in w for w in weak) < 3 for t in TYPES)
        out["mega"] = sum("-Mega" in v.s for v in team) <= (0 if core_mega else 1)
        if seed_terrain:
            out["terrain"] = any(set(tc.abilities(v.s)) & seed_terrain for v in team)
        return out

    def value(team):
        off = core_off.copy()
        ans = np.zeros(len(threats))
        for v in team:
            off = np.maximum(off, v.off)
            ans = np.maximum(ans, v.ans)
        return float(off.mean() + ans.mean()), float(off.mean()), float(ans.mean())

    def search(required):
        beam = [()]
        for depth in range(5):
            nxt = {}
            for team in beam:
                used = {v.s.split("-")[0] for v in team}
                for v in variants:
                    if v.s.split("-")[0] in used:
                        continue
                    new = tuple(sorted(team + (v,), key=lambda x: (x.s, x.role)))
                    key = tuple((x.s, x.role) for x in new)
                    if key in nxt:
                        continue
                    m = met(new)
                    score = value(new)[0] + sum(1.0 for r in required if m[r])
                    nxt[key] = (score, new)
            beam = [t for _, t in sorted(nxt.values(), key=lambda x: -x[0])[:args.beam]]
        ok = [t for t in beam if all(met(t)[r] for r in required)]
        return max(ok or beam, key=lambda t: value(t)[0])

    print(f"\n{args.core} ({', '.join(core_moves)}), Speed {core_t.speed}; weak to {', '.join(sorted(core_weak))}")
    free = search(["mega"])
    if seed_terrain:
        REQUIREMENTS.append("terrain")
    REQUIREMENTS.append("mega")
    full = search(REQUIREMENTS)
    fv, ff = value(free), value(full)
    print(f"\nno requirements:  offence {fv[1]:.3f}, threats answered {fv[2]:.3f}  "
          f"({', '.join(f'{v.s} [{v.role}]' for v in free)})")
    print(f"all requirements: offence {ff[1]:.3f}, threats answered {ff[2]:.3f}  ({sum(met(full).values())} of "
          f"{len(REQUIREMENTS)} met)")
    print("cost of each requirement (value without it minus value with all):")
    for r in [x for x in REQUIREMENTS if x != "mega"]:
        t = search([x for x in REQUIREMENTS if x != r])
        print(f"  {r:9s} {value(t)[0] - ff[0]:+.3f}")
    print("\nteam with every requirement:")
    print(f"  {args.core:20s} {', '.join(core_moves + ['Protect'])} @ {core_t.item}")
    for v in full:
        print(f"  {v.s:20s} {', '.join(v.moves):60s} [{v.role}; {', '.join(sorted(v.roles)) or '-'}] "
              f"{v.ability}, Speed {v.speed}")
    m = met(full)
    print("requirements: " + ", ".join(f"{r} {'yes' if m[r] else 'NO'}" for r in REQUIREMENTS))
    weak = [core_weak] + [weaknesses(v.s, v.ability) for v in full]
    shared = {t: sum(t in w for w in weak) for t in TYPES if sum(t in w for w in weak) >= 2}
    print(f"weaknesses shared by two or more: {shared}")


if __name__ == "__main__":
    main()
