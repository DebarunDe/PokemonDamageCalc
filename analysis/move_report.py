"""Everything a Pokemon learns, scored against every legal Pokemon weighted equally (no usage data).

    1. each damaging move alone: power (OHKO 1 / 2HKO 0.5, accuracy included), OHKO share, and what it adds to the
       best 3-attack set (the extra share of the roster that only it KOs)
    2. best 3 attacks + one more slot, for each fourth-slot option (Protect, Fake Out, Swords Dance at +2, ...)
    3. who threatens it (best theory sets, theory_movesets.csv): OHKO it, or 2HKO it and move first, at its normal
       Speed and at a second Speed (--speed2, e.g. Unburden)
    4. its status moves, listed with what they do

Configurations (--config): name=item,ability,speed multiplier ("none" = no item). Default for Sneasler: itemless
Unburden active (White Herb or seed used: itemless, double Speed) and White Herb still held (turn 1).

    python analysis/move_report.py Sneasler
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

import movesets as mv
import team_build as tb
import theorycraft as tc

NOTES = {
    "Fake Out": "flinches on its first turn out (+3 priority)", "Protect": "blocks a turn",
    "Swords Dance": "+2 Attack", "Bulk Up": "+1 Attack / +1 Defense", "Agility": "+2 Speed",
    "Coaching": "+1 Attack / +1 Defense to its partner", "Taunt": "stops status moves (Trick Room, Tailwind setters)",
    "Quick Guard": "blocks priority for its side this turn", "Feint": "breaks Protect (+2 priority, weak)",
    "Toxic": "badly poisons", "Toxic Spikes": "poisons Pokemon switching in", "Screech": "-2 Defense on a target",
    "Switcheroo": "swaps items (e.g. hands over a Choice Scarf / Flame Orb); also triggers Unburden",
    "Endure": "survives a hit at 1 HP", "Substitute": "25% HP decoy", "Rest": "full heal, sleeps",
    "Sleep Talk": "uses a move while asleep", "Spite": "-4 PP", "Rain Dance": "sets rain", "Sunny Day": "sets sun",
    "Nasty Plot": "+2 Sp. Atk (wrong stat)", "Calm Mind": "+1 Sp. Atk / Sp. Def (wrong stat)",
    "U-turn": "attacks and switches out", "Upper Hand": "+3 priority, only if the target uses a priority move",
    "Quick Attack": "+1 priority", "Vacuum Wave": "+1 priority (special)", "Rock Slide": "spread, 30% flinch",
    "Dire Claw": "50% chance to poison, paralyse or put to sleep", "Throat Chop": "stops sound moves for 2 turns",
    "Rock Tomb": "-1 Speed", "Low Sweep": "-1 Speed", "Acid Spray": "-2 Sp. Def", "Lash Out": "double if its "
    "stats were lowered this turn", "Reversal": "stronger at low HP", "Facade": "double when statused",
    "Thief": "steals the target's item (triggers Unburden if it had none)", "Trailblaze": "+1 Speed",
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pokemon")
    p.add_argument("--config", nargs="*", default=None)
    p.add_argument("--speed2", type=float, default=2.0, help="second Speed multiplier for the threat check")
    args = p.parse_args(argv)
    s = args.pokemon
    defs, speeds = tc.bench()
    speeds = np.array(speeds)
    configs = args.config or ["Unburden active (item used)=none,Unburden,2",
                              "White Herb still held (turn 1)=White Herb,Unburden,1"]
    learn = sorted(tc.learn(s))
    cat = tc.category(s)
    dmg = [m for m in learn if tc.minfo(m) and tc.minfo(m)["category"] == cat and m not in mv.EXCLUDE
           and m not in ("Fake Out",)]
    status = [m for m in learn if tc.minfo(m) and tc.minfo(m)["category"] == "Status"]
    other_cat = [m for m in learn if tc.minfo(m) and tc.minfo(m)["category"] not in ("Status", cat)]
    excluded = [m for m in learn if tc.minfo(m) and tc.minfo(m)["category"] == cat and m in mv.EXCLUDE]
    print(f"{s}: {tc.info(s)['types']}, base {tc.info(s)['base_stats']}, abilities {tc.abilities(s)}")
    print(f"{len(learn)} moves: {len(dmg)} {cat.lower()} attacks scored, {len(status)} status, "
          f"{len(other_cat)} of the other category ({', '.join(other_cat)}), excluded: {', '.join(excluded)}")
    for conf in configs:
        name, spec = conf.split("=")
        item, ability, smult = spec.split(",")
        t = tc.Theory(s, ability, cat=cat)
        t.item = None if item == "none" else item
        t.speed = round(tc.speed_of(t.pset()) * float(smult))
        good, P = mv.pct_matrix(t, dmg, None)
        acc = np.array([tc.acc_of(m) for m in good])
        scale = np.array([mv.MULTI_ACC[m](a) if m in mv.MULTI_ACC else 1.0 for m, a in zip(good, acc)])
        acc = np.where(scale < 1, 1.0, acc)
        P = P * scale[None, :]
        C = mv.credit(P, acc)
        mask = (t.speed > speeds).astype(float)
        w = (1 + mask) / len(speeds)
        best3 = mv.search(C, w, 3)[0]
        b3 = list(best3[1])
        base = C[:, b3].max(axis=1)
        print(f"\n=== {name}: {t.item or 'no item'}, {ability}, Speed {t.speed} (moves before "
              f"{mask.mean():.0%} of the roster)")
        print(f"best 3 attacks: {', '.join(good[j] for j in b3)}  (score {best3[0]:.3f}; power {base.mean():.3f}, "
              f"OHKO {(base > 0.5).mean():.0%}, can't 2HKO {(base == 0).mean():.0%})")
        print(f"\n{'move':16s} {'type':9s} {'BP':>4s} {'acc':>4s}  {'power':>5s} {'OHKO':>5s}  {'adds':>5s}  "
              f"note / Pokemon only it KOs (vs the best 3)")
        rows = []
        for j, m in enumerate(good):
            i = tc.minfo(m)
            col = C[:, j]
            others = [k for k in b3 if k != j]
            ref = C[:, others].max(axis=1) if j in b3 else base
            adds = (np.maximum(ref, col) - ref)
            only = [d for d, a in zip(tc.ROSTER, adds) if a > 0]
            rows.append((float(w @ np.maximum(ref, col)) - float(w @ ref), m, i, col, only, j in b3))
        for gain, m, i, col, only, inb in sorted(rows, key=lambda r: (-r[5], -r[0])):
            note = NOTES.get(m, "")
            who = ", ".join(only[:6]) + (" ..." if len(only) > 6 else "")
            print(f"{m + (' *' if inb else ''):16s} {i['type']:9s} {i['base_power'] or '-':>4} "
                  f"{tc.acc_of(m):4.0%}  {col.mean():5.2f} {(col > 0.5).mean():5.0%}  {gain:+5.3f}  "
                  f"{note}{'; ' if note and who else ''}{who}")
        # fourth slot
        print("\nfourth slot (with the best 3 attacks):")
        f4 = mv.search(C, w, 4)[0]
        print(f"  best 4th attack: {', '.join(good[j] for j in f4[1])}  score {f4[0]:.3f} "
              f"(+{f4[0] - best3[0]:.3f} over 3)")
        t2 = tc.Theory(s, ability, cat=cat)
        t2.item = t.item
        stat = "atk" if cat == "Physical" else "spa"
        orig = t2.pset
        t2.pset = lambda boosts=None, status="", _o=orig: _o({stat: 2}, status)
        g2, P2 = mv.pct_matrix(t2, [good[j] for j in b3], None)
        P2 = P2 * scale[b3][None, :]
        C2 = mv.credit(P2, acc[b3])
        sd = float(w @ C2.max(axis=1))
        print(f"  Swords Dance (+2): the best 3 then score {sd:.3f} (+{sd - best3[0]:.3f}); "
              f"OHKO {(C2.max(axis=1) > 0.5).mean():.0%}")
        # pairs with Fake Out: the 2 best attacks + Fake Out + Protect
        two = mv.search(C, w, 2)[0]
        print(f"  best 2 attacks (for a Fake Out / Swords Dance / Protect set): "
              f"{', '.join(good[j] for j in two[1])}  score {two[0]:.3f} ({two[0] - best3[0]:+.3f} vs 3 attacks)")

    # threats
    best = tb.best_sets()
    t = tc.Theory(s, "Unburden" if "Unburden" in tc.abilities(s) else tc.abilities(s)[0], cat=cat)
    t.item = None
    me = t.pset()
    base_spe = t.speed
    print("\n=== who threatens it (32 / 32 theory spread; their best theory sets)")
    rows = []
    for o, r in best.items():
        ot, mvs = tb.pset_of(r)
        Pc, g, a = tb.damage_rows(ot, mvs, [me])
        hit = float((a * Pc[0]).max())
        m = g[int(np.argmax(a * Pc[0]))]
        ospe = ot.speed * (1.5 if ot.item == "Choice Scarf" else 1)
        pri = tc.minfo(m)["priority"] > 0
        rows.append((hit, o, m, ospe, pri))
    for label, spe in (("normal Speed", base_spe), (f"Speed x{args.speed2:g}", base_spe * args.speed2)):
        ohko = [r for r in rows if r[0] >= 100]
        fast2 = [r for r in rows if 50 <= r[0] < 100 and (r[3] > spe or r[4])]
        fast1 = [r for r in ohko if r[3] > spe or r[4]]
        print(f"  {label} ({spe:.0f}): {len(ohko)} Pokemon OHKO it ({len(fast1)} of them first), "
              f"{len(fast2)} more 2HKO it first")
    by_type = {}
    for hit, o, m, ospe, pri in rows:
        if hit >= 100:
            by_type.setdefault(tc.minfo(m)["type"], []).append(o)
    print("  OHKOs by the attacking move's type: " + "; ".join(f"{k} {len(v)}" for k, v in
                                                            sorted(by_type.items(), key=lambda kv: -len(kv[1]))))
    print("  hardest hitters: " + ", ".join(f"{o} {m} {h:.0f}%" for h, o, m, _, _ in sorted(rows, reverse=True)[:10]))

    print("\n=== status moves")
    for m in status:
        print(f"  {m:14s} {NOTES.get(m, '')}")


if __name__ == "__main__":
    main()
