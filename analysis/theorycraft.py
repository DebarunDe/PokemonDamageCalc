"""Theorycraft: team ideas from mechanics alone, with no usage data.

Every combo is found by rule from the legal roster (abilities, learnsets, types) and measured with damage calcs
against every legal Pokemon weighted equally (Serious nature, 32 HP / 17 Def / 17 SpD, first ability, no item),
so a Pokemon nobody plays counts as much as the most used one.

The beneficiary gets a theory set: 32 in its attacking stat and Speed (Trick Room: 32 HP, 0 Speed, -Spe nature),
its best damaging move of each type from its learnset (up to six types), the combo's move if it needs one.

    power        share of the roster its best move OHKOs (1) or 2HKOs (0.5), accuracy included
    first        the same, counting only Pokemon it moves before (bench Speed: 16 SP, neutral nature)
    gain         change in power + first when the combo is on

Families:
    ally-hit     a partner hits it on purpose to trigger its ability (Justified + Beat Up, Steam Engine + Surf,
                 Defiant + Bulldoze, Anger Point + an always-crit move, Sap Sipper + Petal Blizzard, ...)
    free spread  a partner immune to its ally-hitting spread move (Discharge + Lightning Rod, Surf + Storm Drain, ...)
    field        weather / terrain setters with the abilities and moves they power up
    speed        Trick Room and Tailwind for the Pokemon that gain the most first strikes
    partner      partner abilities that multiply damage (Power Spot, Battery, Steely Spirit, Plus / Minus)
    gravity      Gravity for inaccurate moves

`tried` is how many tournament teams have ever paired the two (any regulation): information, not a weight.

    python analysis/theorycraft.py [--per-family 8]
Output: analysis/output/theory_ideas.csv and a report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import doubles_sim as ds
from champcalc import CalcError, Field, PokemonSet

calc = ds.calc
OUT = Path(__file__).parent / "output"
ROSTER = list(calc.legal_species)

SKIP_MOVES = {"Hyper Beam", "Giga Impact", "Meteor Beam", "Sky Attack", "Skull Bash", "Solar Beam", "Solar Blade",
              "Explosion", "Self-Destruct", "Misty Explosion", "Fling", "Natural Gift", "Last Resort", "Dream Eater",
              "Belch", "Focus Punch", "Future Sight", "Doom Desire", "Synchronoise", "Burn Up", "Double Shock",
              "Steel Roller", "Shell Trap", "Beak Blast", "Counter", "Mirror Coat", "Metal Burst", "Bide",
              "Endeavor", "Final Gambit", "Super Fang", "Ruination", "Nature's Madness", "Night Shade",
              "Seismic Toss", "Psywave", "Sonic Boom", "Dragon Rage", "Rollout", "Ice Ball", "Outrage",
              "Petal Dance", "Thrash", "Uproar", "Fake Out", "First Impression", "Sucker Punch", "Upper Hand",
              "Electro Shot", "Snore", "Spit Up", "Stomping Tantrum", "Retaliate", "Rage Fist", "Last Respects",
              "Hard Press", "Crush Grip", "Wring Out", "Eruption", "Water Spout", "Dragon Energy", "Electro Ball",
              "Gyro Ball", "Grass Knot", "Low Kick", "Heat Crash", "Heavy Slam", "Punishment", "Trump Card",
              "Pain Split", "Mind Blown", "Head Smash", "Wood Hammer", "Brave Bird", "Flare Blitz", "Volt Tackle",
              "Wave Crash", "Double-Edge", "Head Charge", "Light of Ruin", "Chloroblast", "Steel Beam"}
ALLY_SPREAD = {"Earthquake", "Discharge", "Surf", "Bulldoze", "Lava Plume", "Sludge Wave", "Petal Blizzard",
               "Boomburst", "Parabolic Charge", "Magnitude", "Searing Shot", "Brutal Swing", "Mind Blown",
               "Explosion", "Self-Destruct"}
WEATHER_SETTERS = {"Drought": "Sun", "Drizzle": "Rain", "Sand Stream": "Sand", "Snow Warning": "Snow",
                   "Orichalcum Pulse": "Sun", "Mega Sol": "Sun"}
TERRAIN_SETTERS = {"Electric Surge": "Electric", "Grassy Surge": "Grassy", "Psychic Surge": "Psychic",
                   "Misty Surge": "Misty", "Seed Sower": "Grassy", "Hadron Engine": "Electric"}
WEATHER_ABILITIES = {"Swift Swim": "Rain", "Chlorophyll": "Sun", "Sand Rush": "Sand", "Slush Rush": "Snow",
                     "Solar Power": "Sun", "Sand Force": "Sand", "Protosynthesis": "Sun", "Rain Dish": "Rain",
                     "Flower Gift": "Sun", "Harvest": "Sun", "Orichalcum Pulse": "Sun"}
TERRAIN_ABILITIES = {"Surge Surfer": "Electric", "Quark Drive": "Electric", "Hadron Engine": "Electric",
                     "Grass Pelt": "Grassy"}
SPEED_ABILITY = {"Swift Swim", "Chlorophyll", "Sand Rush", "Slush Rush", "Surge Surfer"}
WEATHER_MOVES = {"Weather Ball": None, "Thunder": "Rain", "Hurricane": "Rain", "Blizzard": "Snow",
                 "Hydro Steam": "Sun"}
TERRAIN_MOVES = {"Expanding Force": "Psychic", "Rising Voltage": "Electric", "Grassy Glide": "Grassy",
                 "Terrain Pulse": None, "Psyblade": "Electric"}
TYPE_IMMUNE = {"Ground": {"Flying"}, "Electric": {"Ground"}, "Poison": {"Steel"}, "Normal": {"Ghost"},
               "Fire": set(), "Water": set(), "Grass": set(), "Dark": set()}
ABILITY_IMMUNE = {"Ground": {"Levitate", "Earth Eater"}, "Electric": {"Lightning Rod", "Volt Absorb", "Motor Drive"},
                  "Water": {"Water Absorb", "Storm Drain", "Dry Skin"}, "Fire": {"Flash Fire", "Well-Baked Body"},
                  "Grass": {"Sap Sipper"}, "Normal": {"Soundproof"}, "Poison": set(), "Dark": set()}
ANY_IMMUNE = {"Telepathy"}  # immune to its partner's moves of every type


# ------------------------------------------------------------------------------------------------- roster data

@lru_cache(maxsize=None)
def info(s):
    return calc.species(s)


@lru_cache(maxsize=None)
def learn(s):
    return frozenset(calc.learnset(s))


def abilities(s):
    return info(s)["abilities"]


def stone(s):
    return info(s).get("mega_stone") if "-Mega" in s else None


@lru_cache(maxsize=None)
def minfo(m):
    try:
        return calc.move(m)
    except CalcError:
        return None


def acc_of(m):
    a = minfo(m)["accuracy"]
    return 1.0 if a is True else a / 100


def attack_moves(s, cat, must=(), n_types=6):
    """Best damaging move of each type in the learnset for one category, by power x STAB x accuracy."""
    types = set(info(s)["types"])
    best = {}
    for m in learn(s):
        i = minfo(m)
        if not i or i["category"] != cat or not i["base_power"] or m in SKIP_MOVES or i["priority"] < 0:
            continue
        v = i["base_power"] * (3 if i["multi_hit"] else 1) * (1.5 if i["type"] in types else 1) * acc_of(m)
        if i["type"] not in best or v > best[i["type"]][0]:
            best[i["type"]] = (v, m)
    out = [m for _, m in sorted(best.values(), reverse=True)[:n_types]]
    return list(dict.fromkeys(list(must) + out))


def category(s):
    b = info(s)["base_stats"]
    return "Physical" if b["atk"] >= b["spa"] else "Special"


BENCH = None
BENCH_SPEED = None


def bench():
    global BENCH, BENCH_SPEED
    if BENCH is None:
        BENCH = [PokemonSet(s, nature="Serious", ability=abilities(s)[0], item=stone(s), sp={"hp": 32, "def": 17,
                                                                                               "spd": 17})
                 for s in ROSTER]
        cache = OUT / "theory_bench_speed.json"
        sp = json.loads(cache.read_text()) if cache.exists() else {}
        for s in ROSTER:
            if s not in sp:
                sp[s] = speed_of(PokemonSet(s, nature="Serious", ability=abilities(s)[0], item=stone(s),
                                            sp={"spe": 16}))
        cache.write_text(json.dumps(sp))
        BENCH_SPEED = [sp[s] for s in ROSTER]
    return BENCH, BENCH_SPEED


def speed_of(pset):
    a, _, _ = calc.calculate_many(pset, PokemonSet("Snorlax"), ["Earthquake"])
    return a["stats"]["spe"]


# ------------------------------------------------------------------------------------------------- measuring

class Theory:
    """A beneficiary's theory set: PokemonSet fields plus its moves and Speed."""

    def __init__(self, s, ability, cat=None, tr=False, must=(), item=None):
        self.s, self.ability, self.cat = s, ability, cat or category(s)
        stat = "atk" if self.cat == "Physical" else "spa"
        if tr:
            nature = "Brave" if stat == "atk" else "Quiet"
            self.sp = {"hp": 32, stat: 32, "spd": 2}
        else:
            nature = "Adamant" if stat == "atk" else "Modest"
            self.sp = {"hp": 2, stat: 32, "spe": 32}
        self.nature, self.item = nature, item or stone(s)
        self.moves = attack_moves(s, self.cat, must)
        self.speed = speed_of(self.pset())

    def pset(self, boosts=None, status=""):
        return PokemonSet(self.s, nature=self.nature, ability=self.ability, item=self.item, sp=self.sp,
                          boosts={k: max(-6, min(6, v)) for k, v in (boosts or {}).items() if v}, status=status)


_profiles = {}


def profile(t: Theory, weather=None, terrain=None, boosts=None, type_mult=None, cat_mult=1.0, speed_mult=1.0,
            trick_room=False, acc_mult=1.0, only=None):
    """(power, first): roster shares it OHKOs (1) / 2HKOs (0.5), and the same counting only Pokemon it moves
    before. type_mult: {type: damage multiplier}; only: restrict to these moves (spread-move checks)."""
    moves = only or t.moves
    key = (t.s, t.ability, t.nature, t.item, tuple(moves), weather, terrain, tuple(sorted((boosts or {}).items())),
           tuple(sorted((type_mult or {}).items())), cat_mult, speed_mult, trick_room, acc_mult)
    if key in _profiles:
        return _profiles[key]
    defs, speeds = bench()
    f = Field(doubles=True, weather=weather, terrain=terrain)
    att = t.pset(boosts)
    try:
        res = calc.calculate_batch(moves, [(att, d, f, list(range(len(moves)))) for d in defs])
    except CalcError:
        res = None
    if res is None:  # one move the calc can't do: drop moves one at a time
        good = []
        for m in moves:
            try:
                calc.calculate_batch([m], [(att, defs[0], f, [0])])
                good.append(m)
            except CalcError:
                pass
        if not good:
            _profiles[key] = (0.0, 0.0)
            return _profiles[key]
        moves = good
        res = calc.calculate_batch(moves, [(att, d, f, list(range(len(moves)))) for d in defs])
    mult = [(type_mult or {}).get(minfo(m)["type"], 1.0) * cat_mult for m in moves]
    acc = [1.0 if t.ability == "No Guard" else min(1.0, acc_of(m) * acc_mult) for m in moves]
    if weather == "Rain":
        acc = [1.0 if m in ("Thunder", "Hurricane") else a for m, a in zip(moves, acc)]
    if weather == "Snow":
        acc = [1.0 if m == "Blizzard" else a for m, a in zip(moves, acc)]
    spe = t.speed * speed_mult * (1 + 0.5 * boosts.get("spe", 0) if boosts and boosts.get("spe", 0) > 0 else 1)
    power = first = 0.0
    for (hp, dmg), d_spe in zip(res, speeds):
        val = 0.0
        for (lo, hi), mu, a in zip(dmg, mult, acc):
            pct = 100 * (lo + hi) / 2 / hp * mu
            val = max(val, a * (1.0 if pct >= 100 else 0.5 if pct >= 50 else 0.0))
        power += val
        if (spe < d_spe) if trick_room else (spe > d_spe):
            first += val
    n = len(res)
    _profiles[key] = (power / n, first / n)
    return _profiles[key]


def gain(before, after):
    return (after[0] - before[0]) + (after[1] - before[1])


# ------------------------------------------------------------------------------------------------- idea families

def learners(move):
    return [s for s in ROSTER if move in learn(s)]


def with_ability(ab):
    return [s for s in ROSTER if ab in abilities(s)]


def idea(family, title, enabler, beneficiary, how, before, after, extra=""):
    return {"family": family, "title": title, "enabler": enabler, "beneficiary": beneficiary, "how": how,
            "power_before": round(before[0], 3), "power_after": round(after[0], 3),
            "first_before": round(before[1], 3), "first_after": round(after[1], 3),
            "gain": round(gain(before, after), 3), "extra": extra}


ALLY_HIT = [  # (ability, enabler moves, effect, description)
    ("Justified", ["Beat Up"], {"boosts": {"atk": 4}}, "Beat Up into it: one Justified boost per hit (+4 with "
                                                        "four healthy Pokemon)"),
    ("Rattled", ["Beat Up"], {"boosts": {"spe": 4}}, "Beat Up into it: +1 Speed per hit"),
    ("Steam Engine", ["Surf", "Lava Plume", "Searing Shot"], {"boosts": {"spe": 6}},
     "a Water / Fire spread move from its partner: +6 Speed"),
    ("Anger Point", ["Frost Breath", "Storm Throw", "Wicked Blow", "Surging Strikes", "Flower Trick"],
     {"boosts": {"atk": 6}}, "an always-crit move from its partner: maximum Attack"),
    ("Defiant", ["Bulldoze", "Parting Shot", "Charm", "Fake Tears", "Screech", "Tickle"], {"boosts": {"atk": 2}},
     "a partner's stat drop on it (Bulldoze also hits the foes): +2 Attack"),
    ("Competitive", ["Bulldoze", "Parting Shot", "Charm", "Fake Tears", "Screech", "Tickle"],
     {"boosts": {"spa": 2}}, "a partner's stat drop on it: +2 Sp. Atk"),
    ("Motor Drive", ["Discharge", "Parabolic Charge"], {"boosts": {"spe": 1}},
     "its partner's Discharge: immune, +1 Speed"),
    ("Lightning Rod", ["Discharge", "Parabolic Charge"], {"boosts": {"spa": 1}},
     "its partner's Discharge: immune, +1 Sp. Atk"),
    ("Storm Drain", ["Surf"], {"boosts": {"spa": 1}}, "its partner's Surf: immune, +1 Sp. Atk"),
    ("Sap Sipper", ["Petal Blizzard"], {"boosts": {"atk": 1}}, "its partner's Petal Blizzard: immune, +1 Attack"),
    ("Thermal Exchange", ["Lava Plume", "Searing Shot"], {"boosts": {"atk": 1}},
     "its partner's Fire spread move: +1 Attack"),
    ("Flash Fire", ["Lava Plume", "Searing Shot"], {"type_mult": {"Fire": 1.5}},
     "its partner's Fire spread move: immune, Fire moves x1.5"),
    ("Weak Armor", ["Bulldoze", "Brutal Swing", "Earthquake"], {"boosts": {"spe": 2, "def": -1}},
     "a partner's physical spread move: +2 Speed"),
    ("Electromorphosis", ["Bulldoze", "Discharge", "Surf", "Earthquake"], {"type_mult": {"Electric": 2.0}},
     "a partner's spread hit charges it: next Electric move x2"),
    ("Wind Rider", ["Tailwind"], {"boosts": {"atk": 1}, "speed_mult": 2.0},
     "its partner's Tailwind: +1 Attack (and Tailwind's Speed)"),
    ("Wind Power", ["Tailwind"], {"type_mult": {"Electric": 2.0}, "speed_mult": 2.0},
     "its partner's Tailwind charges it: next Electric move x2 (and Tailwind's Speed)"),
]


def ally_cost(enabler, move, t: Theory):
    """% of its partner's HP an enabler's move takes (enabler with no attack investment, crit for the always-crit
    moves, spread reduction for spread moves); 0 for status moves."""
    i = minfo(move)
    if i["category"] == "Status":
        return 0.0
    att = PokemonSet(enabler, nature="Serious", ability=abilities(enabler)[0], sp={"hp": 32})
    # the calc gives Beat Up 0 power (it depends on the party): four ~15-power hits, about 0.75 of a Crunch
    used, scale = ("Crunch", 0.75) if move == "Beat Up" else (move, 1.0)
    try:
        (hp, [(lo, hi)]), = calc.calculate_batch([used], [(att, t.pset(), Field(doubles=True), [0])])
    except CalcError:
        return 99.0
    return 100 * scale * (lo + hi) / 2 / hp


ABSORB = {"Motor Drive", "Lightning Rod", "Storm Drain", "Sap Sipper", "Flash Fire", "Volt Absorb", "Water Absorb"}


def bulk(s):
    b = info(s)["base_stats"]
    return b["hp"] + b["def"] + b["spd"]


def ally_hit():
    out = []
    for ab, moves, eff, how in ALLY_HIT:
        bens = with_ability(ab)
        enablers = sorted({(s, m) for m in moves for s in learners(m)})
        if not bens or not enablers:
            continue
        for b in bens:
            t = Theory(b, ab, cat="Special" if "spa" in eff.get("boosts", {}) or ab in ("Wind Power",
                                                                                         "Electromorphosis")
                       else ("Physical" if "atk" in eff.get("boosts", {}) else None))
            before = profile(t)
            after = profile(t, boosts=eff.get("boosts"), type_mult=eff.get("type_mult"),
                            speed_mult=eff.get("speed_mult", 1.0))
            cost = {}
            for e, m in enablers:
                if e != b:
                    c = ally_cost(e, m, t)
                    if c == 0 and minfo(m)["category"] != "Status" and ab not in ABSORB:
                        continue  # immune to the move (Bulldoze on a Flying type): nothing triggers
                    if c < cost.get(e, (999, ""))[0]:
                        cost[e] = (c, m)
            names = sorted(cost, key=lambda e: cost[e][0])
            gentlest = "; ".join(f"{e} {cost[e][1]} ({cost[e][0]:.0f}%)" for e in names[:5])
            out.append(idea("ally-hit", f"{b} ({ab}) + partner {'/'.join(moves[:2])}", names, b, how, before, after,
                            extra=f"gentlest enablers (damage to it): {gentlest}"))
    return out


def free_spread():
    out = []
    for m in sorted(ALLY_SPREAD):
        i = minfo(m)
        if not i or m in ("Explosion", "Self-Destruct", "Mind Blown"):
            continue
        typ = i["type"]
        partners = {s for s in ROSTER if set(info(s)["types"]) & TYPE_IMMUNE.get(typ, set())}
        ab_partners = {s for s in ROSTER if set(abilities(s)) & (ABILITY_IMMUNE.get(typ, set()) | ANY_IMMUNE)}
        for a in learners(m):
            t = Theory(a, abilities(a)[0], cat=i["category"], must=(m,))
            p = profile(t, only=[m])
            ab = sorted(s for s in ab_partners if s != a)
            ty = sorted(s for s in partners - ab_partners if s != a)
            if not ab and not ty:
                continue
            out.append(idea("free spread", f"{a} {m} with an immune partner", ab + ty, a,
                            f"{m} hits both foes and its partner; partners immune by ability: "
                            f"{', '.join(ab[:8]) or '-'}; by type: {len(ty)}", (0.0, 0.0), p,
                            extra=f"ability partners: {len(ab)}"))
    return out


def field_ideas():
    out = []
    setters_w = defaultdict(list)
    for s in ROSTER:
        for ab in abilities(s):
            if ab in WEATHER_SETTERS:
                setters_w[WEATHER_SETTERS[ab]].append(s)
            if ab in TERRAIN_SETTERS:
                setters_w[TERRAIN_SETTERS[ab] + " Terrain"].append(s)
    jobs = []  # (beneficiary, ability, must move, weather, terrain, reason)
    for s in ROSTER:
        for ab in abilities(s):
            if ab in WEATHER_ABILITIES:
                jobs.append((s, ab, (), WEATHER_ABILITIES[ab], None, ab))
            if ab in TERRAIN_ABILITIES:
                jobs.append((s, ab, (), None, TERRAIN_ABILITIES[ab], ab))
        for m, w in WEATHER_MOVES.items():
            if m in learn(s):
                for ww in ([w] if w else ["Sun", "Rain", "Sand", "Snow"]):
                    jobs.append((s, abilities(s)[0], (m,), ww, None, m))
        for m, tr in TERRAIN_MOVES.items():
            if m in learn(s):
                for tt in ([tr] if tr else ["Electric", "Grassy", "Psychic", "Misty"]):
                    jobs.append((s, abilities(s)[0], (m,), None, tt, m))
    for s, ab, must, w, tr, why in jobs:
        cat = minfo(must[0])["category"] if must else None
        t = Theory(s, ab, cat=cat, must=must)
        before = profile(t)
        after = profile(t, weather=w, terrain=tr, speed_mult=2.0 if ab in SPEED_ABILITY else 1.0)
        cond = w or (tr + " Terrain")
        setters = sorted((x for x in setters_w.get(cond, []) if x != s), key=lambda x: -bulk(x))
        if not setters:
            continue
        out.append(idea("field", f"{s} ({why}) in {cond}", setters, s, f"{cond} from a partner powers its {why}",
                        before, after))
    return out


def speed_ideas():
    out = []
    tr_setters = sorted(learners("Trick Room"), key=lambda x: -bulk(x))
    tw_setters = sorted(learners("Tailwind"), key=lambda x: -bulk(x))
    for s in ROSTER:
        b = info(s)["base_stats"]
        if max(b["atk"], b["spa"]) < 90:
            continue
        if b["spe"] <= 70:
            t = Theory(s, abilities(s)[0], tr=True)
            fast = Theory(s, abilities(s)[0])
            out.append(idea("speed", f"{s} under Trick Room", [x for x in tr_setters if x != s], s,
                            "Trick Room from a partner: it moves before almost everything", profile(fast),
                            profile(t, trick_room=True)))
        t = Theory(s, abilities(s)[0])
        out.append(idea("speed", f"{s} under Tailwind", [x for x in tw_setters if x != s], s,
                        "Tailwind from a partner doubles its Speed", profile(t), profile(t, speed_mult=2.0)))
    return out


PARTNER_MULT = [("Power Spot", None, 1.3, "Power Spot partner: its moves x1.3"),
                ("Battery", "Special", 1.3, "Battery partner: its special moves x1.3"),
                ("Steely Spirit", "Steel", 1.5, "Steely Spirit partner: its Steel moves x1.5"),
                ("Flower Gift", "Physical-sun", 1.5, "Flower Gift partner in sun: its Attack x1.5")]


def partner_ideas():
    out = []
    for ab, what, mult, how in PARTNER_MULT:
        partners = with_ability(ab)
        if not partners:
            continue
        for s in ROSTER:
            b = info(s)["base_stats"]
            if max(b["atk"], b["spa"]) < 100:
                continue
            if what == "Special" and b["spa"] < b["atk"]:
                continue
            t = Theory(s, abilities(s)[0])
            if what == "Steel":
                if "Steel" not in {minfo(m)["type"] for m in t.moves}:
                    continue
                after = profile(t, type_mult={"Steel": mult})
            elif what == "Physical-sun":
                if t.cat != "Physical":
                    continue
                after = profile(t, weather="Sun", cat_mult=mult)
            else:
                after = profile(t, cat_mult=mult)
            out.append(idea("partner", f"{s} with a {ab} partner", [x for x in partners if x != s], s, how,
                            profile(t), after))
    # Plus / Minus: both partners boost each other
    pm = with_ability("Plus") + with_ability("Minus")
    for a, c in combinations(pm, 2):
        for s in (a, c):
            t = Theory(s, "Plus" if "Plus" in abilities(s) else "Minus", cat="Special")
            out.append(idea("partner", f"{s} with {c if s == a else a} (Plus / Minus)", [c if s == a else a], s,
                            "Plus and Minus partners: Sp. Atk x1.5", profile(t), profile(t, cat_mult=1.5)))
    return out


def gravity_ideas():
    out = []
    setters = sorted(learners("Gravity"), key=lambda x: -bulk(x))
    for s in ROSTER:
        t = Theory(s, abilities(s)[0])
        if t.ability == "No Guard":
            continue
        if all(acc_of(m) >= 0.9 for m in t.moves):
            # add its strongest inaccurate move, if any, so Gravity has something to fix
            inacc = [m for m in learn(s) if minfo(m) and minfo(m)["category"] == t.cat and minfo(m)["base_power"]
                     and acc_of(m) < 0.9 and m not in SKIP_MOVES]
            if not inacc:
                continue
            t = Theory(s, abilities(s)[0], must=tuple(sorted(inacc, key=lambda m: -minfo(m)["base_power"])[:2]))
        out.append(idea("gravity", f"{s} under Gravity", [x for x in setters if x != s], s,
                        "Gravity from a partner: accuracy x5/3 (Ground moves also hit Flying / Levitate, "
                        "not counted)", profile(t), profile(t, acc_mult=5 / 3)))
    return out


# ------------------------------------------------------------------------------------------------- report

def tried_pairs():
    """Tournament teams (any regulation) that had both Pokemon, by base species name pair."""
    from replays import base
    try:
        db = ds.db()
    except Exception:
        return Counter()
    teams = defaultdict(set)
    for tour, player, name in db.execute("SELECT tournament, player, pokemon FROM team_members"):
        teams[(tour, player)].add(base(name))
    c = Counter()
    for t in teams.values():
        for a, b in combinations(sorted(t), 2):
            c[(a, b)] += 1
    return c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--per-family", type=int, default=8)
    p.add_argument("--families", nargs="*", default=["ally-hit", "free spread", "field", "speed", "partner",
                                                      "gravity"])
    args = p.parse_args(argv)
    from replays import base
    OUT.mkdir(exist_ok=True)
    fams = {"ally-hit": ally_hit, "free spread": free_spread, "field": field_ideas, "speed": speed_ideas,
            "partner": partner_ideas, "gravity": gravity_ideas}
    ideas = []
    for f in args.families:
        print(f"{f}...", flush=True)
        ideas += fams[f]()
    tried = tried_pairs()
    for x in ideas:
        b = base(x["beneficiary"])
        x["tried"] = "; ".join(f"{e} {tried.get(tuple(sorted((b, base(e)))), 0)}x" for e in x["enabler"][:5])
        x["enabler_count"] = len(x["enabler"])
        x["enablers"] = ", ".join(x["enabler"][:12]) + (" ..." if len(x["enabler"]) > 12 else "")
        del x["enabler"]
    keys = ["family", "title", "beneficiary", "gain", "power_before", "power_after", "first_before", "first_after",
            "enabler_count", "enablers", "tried", "how", "extra"]
    with open(OUT / "theory_ideas.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in sorted(ideas, key=lambda x: (x["family"], -x["gain"])):
            w.writerow({k: x[k] for k in keys})
    print(f"\n{len(ideas)} ideas -> {OUT / 'theory_ideas.csv'}")
    for f in args.families:
        rows = sorted((x for x in ideas if x["family"] == f), key=lambda x: -x["gain"])
        seen = set()
        print(f"\n== {f} (gain = change in power + first strike; shares of the whole legal roster)")
        k = 0
        for x in rows:
            if x["beneficiary"] in seen:
                continue
            seen.add(x["beneficiary"])
            print(f"  {x['gain']:+.2f}  {x['title']}: power {x['power_before']:.2f} -> {x['power_after']:.2f}, "
                  f"first {x['first_before']:.2f} -> {x['first_after']:.2f}; enablers ({x['enabler_count']}): "
                  f"{x['enablers'][:90]}")
            print(f"           tournament teams with both (first five enablers): {x['tried']}" +
                  (f"\n           {x['extra']}" if x["extra"] else ""))
            k += 1
            if k >= args.per_family:
                break


if __name__ == "__main__":
    main()
