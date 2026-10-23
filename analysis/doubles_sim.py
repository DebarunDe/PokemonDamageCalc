"""Turn-based doubles battle simulator for Pokemon Champions, for testing cores against real teams.

Each battle plays out turn by turn with @smogon/calc damage (via champcalc), random damage rolls,
accuracy and critical hits. Both sides use the same heuristic AI: take KOs and focus fire, Protect
when about to be KO'd, set up when it's safe (counting a partner's Fake Out or redirection), Fake Out
the biggest threat on the first turn, set Tailwind or Trick Room when slower, and so on.

Mechanics covered:
    speed order with priority, Tailwind, Trick Room, paralysis and weather speed abilities
    Fake Out (blocked by Inner Focus, Ghost types, Armor Tail and Psychic Terrain), flinching moves
    Protect and its variants (a second use in a row works 1 time in 3), Wide Guard, Quick Guard
    Follow Me / Rage Powder, Helping Hand, Reflect / Light Screen / Aurora Veil
    Intimidate, weather and terrain on entry, Defiant / Competitive, Clear Body and friends,
    a Mega's pre-Mega ability blocking Intimidate on entry (e.g. Metagross's Clear Body)
    setup moves, Parting Shot, Haze, Taunt, sleep, burn, paralysis, healing moves
    Focus Sash, Sitrus Berry, Leftovers, Life Orb, Choice Scarf, recoil, drain, Knock Off
    Sucker Punch, Rage Fist, Last Respects, Eruption / Water Spout, Electro Shot, Steel Roller, Grassy Glide,
    spread moves (and Earthquake hitting the ally), replacements from the back
    Choice item locking; Hurricane / Thunder / Blizzard accuracy in weather; Solar Beam and Growth in sun;
    accuracy of status moves (Sleep Powder 75%), Wide Lens, Ancient Power's stat boost
    Supreme Overlord, Unburden, Hospitality, Friend Guard, Rough Skin / Iron Barbs; White Herb, Rocky Helmet,
    Lum Berry and type-resist berries (used up after one hit)
Not covered: voluntary switching (other than Parting Shot), Encore, Substitute,
abilities beyond those above and the ones the calc applies to damage.

Sets come from real tournament teams (analysis/data/meta.sqlite): each Pokemon's most common complete
set (item, ability, nature, moves), with its most common ladder spread for that nature.

Usage:
    python analysis/doubles_sim.py battle "Metagross-Mega,Arcanine-Hisui" "Incineroar,Gholdengo"
    python analysis/doubles_sim.py core "Rillaboom,Metagross-Mega,Arcanine-Hisui" --teams 400

`core` plays the core (2, 3 or 4 Pokemon, first two lead) against that many real tournament teams:
the official Regionals' top 32s plus a sample of recent online top 8s. The opponent brings as many
Pokemon as the core, picked at random from its six, and every battle is played twice.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from champcalc import CalcError, Calculator, Field, PokemonSet

DB = Path(__file__).parent / "data" / "meta.sqlite"
REGULATION = "M-C"
LADDER_MONTH = "2026-09"
RECENT = "2026-09-26"  # online events on the current meta
STATS = ("hp", "atk", "def", "spa", "spd", "spe")

PROTECTS = {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap", "Burning Bulwark",
            "Obstruct"}
BOOST = {
    "Swords Dance": {"atk": 2}, "Dragon Dance": {"atk": 1, "spe": 1}, "Nasty Plot": {"spa": 2},
    "Calm Mind": {"spa": 1, "spd": 1}, "Quiver Dance": {"spa": 1, "spd": 1, "spe": 1}, "Bulk Up": {"atk": 1, "def": 1},
    "Shell Smash": {"atk": 2, "spa": 2, "spe": 2, "def": -1, "spd": -1}, "Agility": {"spe": 2},
    "Iron Defense": {"def": 2}, "Coil": {"atk": 1, "def": 1}, "Growth": {"atk": 1, "spa": 1},
    "Victory Dance": {"atk": 1, "def": 1, "spe": 1}, "Tidy Up": {"atk": 1, "spe": 1},
    "Clangorous Soul": {"atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1}, "Curse": {"atk": 1, "def": 1, "spe": -1},
    "Work Up": {"atk": 1, "spa": 1},
}
# Spread moves that also hit the user's ally.
ALLY_SPREAD = {"Earthquake", "Surf", "Discharge", "Bulldoze", "Sludge Wave", "Lava Plume", "Boomburst", "Explosion",
               "Self-Destruct", "Petal Blizzard", "Magnitude", "Searing Shot", "Parabolic Charge"}
RECOIL = {"Flare Blitz": 0.33, "Brave Bird": 0.33, "Wave Crash": 0.33, "Double-Edge": 0.33, "Wood Hammer": 0.33,
          "Head Smash": 0.5, "Wild Charge": 0.25, "Light of Ruin": 0.5, "Take Down": 0.25, "Volt Tackle": 0.33}
DRAIN = {"Drain Punch": 0.5, "Giga Drain": 0.5, "Horn Leech": 0.5, "Matcha Gotcha": 0.5, "Draining Kiss": 0.75,
         "Parabolic Charge": 0.5, "Leech Life": 0.5, "Bitter Blade": 0.5}
SELF_DROP = {"Close Combat": {"def": -1, "spd": -1}, "Draco Meteor": {"spa": -2}, "Overheat": {"spa": -2},
             "Leaf Storm": {"spa": -2}, "Fleur Cannon": {"spa": -2}, "Make It Rain": {"spa": -1},
             "Superpower": {"atk": -1, "def": -1}, "Hammer Arm": {"spe": -1}, "Armor Cannon": {"def": -1, "spd": -1},
             "Headlong Rush": {"def": -1, "spd": -1}}
TARGET_DROP = {"Icy Wind": {"spe": -1}, "Electroweb": {"spe": -1}, "Rock Tomb": {"spe": -1}, "Bulldoze": {"spe": -1},
               "Snarl": {"spa": -1}, "Struggle Bug": {"spa": -1}, "Breaking Swipe": {"atk": -1}, "Lunge": {"atk": -1},
               "Mystical Fire": {"spa": -1}, "Spirit Break": {"spa": -1}, "Trop Kick": {"atk": -1},
               "Thunderous Kick": {"def": -1}, "Low Sweep": {"spe": -1}, "Mud Shot": {"spe": -1}}
SELF_BOOST_ON_HIT = {"Flame Charge": {"spe": 1}, "Power-Up Punch": {"atk": 1}, "Trailblaze": {"spe": 1},
                     "Aqua Step": {"spe": 1}, "Rapid Spin": {"spe": 1}, "Torch Song": {"spa": 1}}
ALL_STATS_CHANCE = {"Ancient Power", "Silver Wind", "Ominous Wind"}  # 10% chance to raise every stat
STATUS_CHANCE = {"Scald": ("brn", 0.3), "Matcha Gotcha": ("brn", 0.2), "Sacred Fire": ("brn", 0.5),
                 "Lava Plume": ("brn", 0.3), "Nuzzle": ("par", 1.0), "Zap Cannon": ("par", 1.0),
                 "Body Slam": ("par", 0.3),
                 "Discharge": ("par", 0.3), "Thunder": ("par", 0.3), "Thunderbolt": ("par", 0.1),
                 "Fire Blast": ("brn", 0.1), "Flamethrower": ("brn", 0.1), "Heat Wave": ("brn", 0.1)}
FLINCH_CHANCE = {"Rock Slide": 0.3, "Air Slash": 0.3, "Iron Head": 0.3, "Zen Headbutt": 0.2, "Dark Pulse": 0.2,
                 "Bite": 0.3, "Headbutt": 0.3, "Waterfall": 0.2, "Icicle Crash": 0.3}
# Power falls with the user's HP (the calc assumes full HP).
HP_SCALED = {"Eruption", "Water Spout", "Dragon Energy"}
HEAL = {"Recover", "Roost", "Slack Off", "Soft-Boiled", "Moonlight", "Morning Sun", "Synthesis", "Strength Sap",
        "Shore Up", "Milk Drink"}
SLEEP_MOVES = {"Spore", "Sleep Powder", "Hypnosis"}
# Moves the AI never picks: charge or recharge turns, or conditions the simulator doesn't model.
NO_SELECT = {"Hyper Beam", "Giga Impact", "Focus Punch", "Dream Eater", "Last Resort",
             "Future Sight", "Doom Desire", "Belch", "Meteor Beam", "Sky Attack", "Bounce", "Fly", "Dig", "Dive",
             "Phantom Force", "Shadow Force"}
INTIMIDATE_IMMUNE = {"Clear Body", "Hyper Cutter", "White Smoke", "Full Metal Body", "Inner Focus", "Oblivious",
                     "Own Tempo", "Scrappy", "Mirror Armor", "Guard Dog"}
DROP_IMMUNE = {"Clear Body", "White Smoke", "Full Metal Body", "Mirror Armor"}
PRIORITY_BLOCKERS = {"Armor Tail", "Queenly Majesty", "Dazzling"}
FIRST_HIT_HALVED = {"Multiscale", "Shadow Shield"}
WEATHER_ABILITIES = {"Drought": "Sun", "Drizzle": "Rain", "Sand Stream": "Sand", "Snow Warning": "Snow",
                     "Orichalcum Pulse": "Sun"}
TERRAIN_ABILITIES = {"Grassy Surge": "Grassy", "Psychic Surge": "Psychic", "Electric Surge": "Electric",
                     "Misty Surge": "Misty", "Seed Sower": "Grassy", "Hadron Engine": "Electric"}
SPEED_ABILITIES = {"Swift Swim": "Rain", "Chlorophyll": "Sun", "Sand Rush": "Sand", "Slush Rush": "Snow"}
CRIT_RATE = 1 / 24
RESIST_BERRIES = {"Chople Berry": "Fighting", "Occa Berry": "Fire", "Passho Berry": "Water", "Wacan Berry": "Electric",
                  "Rindo Berry": "Grass", "Yache Berry": "Ice", "Kebia Berry": "Poison", "Shuca Berry": "Ground",
                  "Coba Berry": "Flying", "Payapa Berry": "Psychic", "Tanga Berry": "Bug", "Charti Berry": "Rock",
                  "Kasib Berry": "Ghost", "Haban Berry": "Dragon", "Colbur Berry": "Dark", "Babiri Berry": "Steel",
                  "Roseli Berry": "Fairy", "Chilan Berry": "Normal"}
CONTACT_PUNISH = {"Rough Skin": 1 / 8, "Iron Barbs": 1 / 8}  # share of the attacker's HP lost on contact

calc = Calculator()
_moves: dict[str, dict] = {}


def move_info(name: str) -> dict:
    if name not in _moves:
        try:
            _moves[name] = calc.move(name)
        except Exception:
            _moves[name] = {"category": "Status", "base_power": 0, "priority": 0, "spread": False, "accuracy": True}
    return _moves[name]


# Moves that only work in sun (they skip the charge turn there).
SUN_ONLY = {"Solar Beam", "Solar Blade"}
CHOICE_ITEMS = {"Choice Scarf", "Choice Band", "Choice Specs"}


def accuracy(name: str, weather: str | None = None) -> float:
    if name in ("Hurricane", "Thunder") and weather in ("Rain", "Sun"):
        return 1.0 if weather == "Rain" else 0.5
    if name == "Blizzard" and weather == "Snow":
        return 1.0
    acc = move_info(name)["accuracy"]
    return 1.0 if acc is True else acc / 100


# ---------------------------------------------------------------------------------------------------- sets

class Spec:
    """One Pokemon's set. `pre_ability` is a Mega's ability before it Mega Evolves (it matters on entry)."""

    def __init__(self, name, ability, item, nature, sp, moves, pre_ability=None):
        self.name, self.ability, self.item, self.nature, self.sp = name, ability, item, nature, dict(sp)
        self.moves = list(moves)
        self.pre_ability = pre_ability or ability
        self.types = tuple(calc.species(name)["types"])
        attacker, _, _ = calc.calculate_many(PokemonSet(name, ability=ability, item=item, nature=nature, sp=sp),
                                             PokemonSet("Snorlax"), ["Earthquake"])
        self.max_hp, self.speed = attacker["stats"]["hp"], attacker["stats"]["spe"]

    def __repr__(self):
        return f"{self.name} @ {self.item} / {self.ability} / {self.nature} {self.sp}: {', '.join(self.moves)}"


def parse_spread(spread: str) -> tuple[str, dict[str, int]] | None:
    """'Adamant:32/32/0/0/2/0' -> ('Adamant', {...}); None if not a legal Stat Point spread."""
    nature, _, values = spread.partition(":")
    try:
        sp = dict(zip(STATS, map(int, values.split("/"))))
    except ValueError:
        return None
    if len(sp) != 6 or max(sp.values()) > 32 or sum(sp.values()) > 66:
        return None
    return nature, {k: v for k, v in sp.items() if v}


_db: sqlite3.Connection | None = None
_specs: dict[str, Spec | None] = {}


def db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    return _db


def real_spec(name: str) -> Spec | None:
    """A Pokemon's most common complete tournament set, with its most common ladder spread for that nature."""
    if name in _specs:
        return _specs[name]
    rows = db().execute(
        "SELECT m.item, m.ability, m.nature, m.moves FROM team_members m JOIN tournaments t ON t.id = m.tournament "
        "WHERE t.regulation = ? AND m.pokemon = ?", (REGULATION, name)).fetchall()
    _specs[name] = None
    if not rows:
        return None
    full = Counter((r[0], r[1], r[2], tuple(sorted(json.loads(r[3])))) for r in rows if r[2] and r[3]).most_common(1)
    if full and full[0][1] >= 3:
        item, pre, nature, moves = full[0][0]
    else:  # too few copies of any one set: take each field's most common value
        def top(values, default=None):
            c = Counter(v for v in values if v).most_common(1)
            return c[0][0] if c else default
        item, pre, nature = top(r[0] for r in rows), top(r[1] for r in rows), top((r[2] for r in rows), "Hardy")
        moves = top((tuple(sorted(json.loads(r[3]))) for r in rows if r[3]), ())
    try:
        legal = calc.species(name)["abilities"]
    except CalcError:  # a name the calc doesn't know, e.g. one entered in another language
        return None
    ability = pre if (pre in legal or not legal) else legal[0]
    sp = None
    for (value,) in db().execute(
            "SELECT value FROM showdown_details WHERE regulation = ? AND month = ? AND ladder = 'bo1' AND rating = 0 "
            "AND pokemon = ? AND kind = 'spread' ORDER BY share DESC LIMIT 10", (REGULATION, LADDER_MONTH, name)):
        parsed = parse_spread(value)
        if parsed and parsed[0] == nature:
            sp = parsed[1]
            break
    if sp is None:
        base = calc.species(name)["base_stats"]
        sp = {("atk" if base["atk"] >= base["spa"] else "spa"): 32, "spe": 32, "hp": 2}
    moves = [m for m in moves if m in calc.all_moves] or ["Struggle"]
    for it in (item, None):
        try:
            _specs[name] = Spec(name, ability, it, nature, sp, moves, pre_ability=pre)
            break
        except CalcError:
            continue
    return _specs[name]


# ---------------------------------------------------------------------------------------------------- battle

class Mon:
    def __init__(self, spec: Spec, side: int):
        self.s, self.side = spec, side
        self.hp = 100.0  # percent of max HP
        self.boosts: Counter = Counter()
        self.status, self.sleep = "", 0
        self.item = spec.item
        self.fainted = False
        self.turns_active = 0
        self.last_protect = self.protected = self.flinched = False
        self.taunt = 0
        self.hits_taken = 0
        self.charging = None
        self.locked = None  # move a Choice item locks it into
        self.item_lost = False  # for Unburden

    @property
    def name(self):
        return self.s.name

    @property
    def ability(self):
        return self.s.ability

    def lose_item(self):
        if self.item:
            self.item, self.item_lost = None, True

    def boost(self, stat, n, by_foe=False):
        if by_foe and n < 0 and self.ability in DROP_IMMUNE:
            return
        self.boosts[stat] = max(-6, min(6, self.boosts[stat] + n))
        if by_foe and n < 0:
            if self.ability == "Defiant":
                self.boosts["atk"] = min(6, self.boosts["atk"] + 2)
            if self.ability == "Competitive":
                self.boosts["spa"] = min(6, self.boosts["spa"] + 2)
        if self.item == "White Herb" and any(v < 0 for v in self.boosts.values()):
            for k in [k for k, v in self.boosts.items() if v < 0]:
                self.boosts[k] = 0
            self.lose_item()

    def set_status(self, status, sleep=0):
        if self.item == "Lum Berry":
            self.lose_item()
            return
        self.status, self.sleep = status, sleep

    def grounded(self):
        return "Flying" not in self.s.types and self.ability != "Levitate"


class Battle:
    _damage_cache: dict = {}

    def __init__(self, side_a: list[Spec], side_b: list[Spec], rng: random.Random, max_turns=20, log=False):
        """Each side is a list of Specs; the first two lead and the rest come in, in order, as replacements."""
        self.rng, self.max_turns, self.log_on = rng, max_turns, log
        self.sides = [[Mon(s, 0) for s in side_a], [Mon(s, 1) for s in side_b]]
        self.active = [list(range(min(2, len(side_a)))), list(range(min(2, len(side_b))))]
        self.bench = [list(range(2, len(side_a))), list(range(2, len(side_b)))]
        self.weather, self.weather_turns = None, 0
        self.terrain, self.terrain_turns = None, 0
        self.trick_room = 0
        self.tailwind, self.reflect, self.light_screen, self.veil = [0, 0], [0, 0], [0, 0], [0, 0]
        self.fainted_count = [0, 0]
        self.turn = 0
        self.lines: list[str] = []

    def log(self, text):
        if self.log_on:
            self.lines.append(text)

    def act(self, side):
        return [self.sides[side][i] for i in self.active[side] if i is not None and not self.sides[side][i].fainted]

    def foes(self, m):
        return self.act(1 - m.side)

    def ally(self, m):
        others = [x for x in self.act(m.side) if x is not m]
        return others[0] if others else None

    def alive(self, side):
        return [m for m in self.sides[side] if not m.fainted]

    # ---- entry
    def enter(self, m):
        m.turns_active, m.locked = 0, None
        weather = WEATHER_ABILITIES.get(m.ability)
        if weather and not (self.weather == weather and self.weather_turns > 0):
            self.weather, self.weather_turns = weather, 5
            self.log(f"  {m.name} sets {weather}")
        terrain = TERRAIN_ABILITIES.get(m.ability)
        if terrain:
            self.terrain, self.terrain_turns = terrain, 5
            self.log(f"  {m.name} sets {terrain} Terrain")
        if m.ability == "Intimidate":
            for f in self.foes(m):
                if f.s.pre_ability in INTIMIDATE_IMMUNE or f.ability in INTIMIDATE_IMMUNE:
                    if f.ability == "Guard Dog":
                        f.boost("atk", 1)
                    continue
                f.boost("atk", -1, by_foe=True)
                self.log(f"  {m.name} Intimidates {f.name}")
        ally = self.ally(m)
        if m.ability == "Hospitality" and ally:
            ally.hp = min(100.0, ally.hp + 25)

    # ---- damage
    def field(self, defender, helping_hand=False):
        side = defender.side
        return Field(weather=self.weather, terrain=self.terrain, reflect=self.reflect[side] > 0,
                     light_screen=self.light_screen[side] > 0, aurora_veil=self.veil[side] > 0,
                     helping_hand=helping_hand, doubles=True,
                     friend_guard=any(x.ability == "Friend Guard" for x in self.act(side) if x is not defender))

    def damage_range(self, att, dfn, move, helping_hand=False):
        """(min %, max %) of the defender's max HP. Spread moves already take the 0.75x doubles penalty."""
        f = self.field(dfn, helping_hand)
        multiscale = dfn.ability in FIRST_HIT_HALVED and dfn.hp >= 100
        key = (att.name, att.ability, att.item, att.s.nature, tuple(sorted(att.s.sp.items())),
               tuple(sorted((k, v) for k, v in att.boosts.items() if v)), att.status,
               dfn.name, dfn.ability, dfn.item, dfn.s.nature, tuple(sorted(dfn.s.sp.items())),
               tuple(sorted((k, v) for k, v in dfn.boosts.items() if v)), multiscale,
               f.weather, f.terrain, f.reflect, f.light_screen, f.aurora_veil, f.friend_guard, helping_hand)
        cache = Battle._damage_cache
        if key not in cache:
            moves = [x for x in att.s.moves if move_info(x)["category"] != "Status"]
            try:
                a = PokemonSet(att.name, ability=att.ability, item=att.item, nature=att.s.nature, sp=att.s.sp,
                               boosts={k: v for k, v in att.boosts.items() if v},
                               status="brn" if att.status == "brn" else "")
                d = PokemonSet(dfn.name, ability=dfn.ability, item=dfn.item, nature=dfn.s.nature, sp=dfn.s.sp,
                               boosts={k: v for k, v in dfn.boosts.items() if v},
                               cur_hp_percent=100 if multiscale else 99)
                results = calc.calculate_many(a, d, moves, f)[2] if moves else []
                cache[key] = {r.move: (r.min_percent, r.max_percent) for r in results}
            except CalcError:
                cache[key] = {}
        lo, hi = cache[key].get(move, (0.0, 0.0))
        mult = 1.0
        if move == "Rage Fist":
            mult = min(7, 1 + att.hits_taken)
        elif move == "Last Respects":
            mult = 1 + self.fainted_count[att.side]
        elif move in HP_SCALED:
            mult = max(att.hp, 1.0) / 100
        if att.ability == "Supreme Overlord":
            mult *= 1 + 0.1 * min(5, self.fainted_count[att.side])
        return lo * mult, hi * mult

    def speed(self, m):
        spe = m.s.speed
        b = m.boosts["spe"]
        spe *= (2 + b) / 2 if b >= 0 else 2 / (2 - b)
        if self.tailwind[m.side] > 0:
            spe *= 2
        if m.status == "par":
            spe *= 0.5
        if SPEED_ABILITIES.get(m.ability) == self.weather and self.weather:
            spe *= 2
        if m.item == "Choice Scarf":
            spe *= 1.5
        if m.ability == "Unburden" and m.item_lost:
            spe *= 2
        return spe

    def priority(self, m, move):
        p = move_info(move)["priority"]
        if move == "Grassy Glide" and self.terrain == "Grassy":
            p = 1
        if m.ability == "Prankster" and move_info(move)["category"] == "Status":
            p += 1
        return p

    # ---- AI
    def threat(self, m, to):
        """Expected % damage of m's best attack into `to`."""
        best = 0.0
        for mv in m.s.moves:
            if move_info(mv)["category"] == "Status" or mv in NO_SELECT or (mv in SUN_ONLY and self.weather != "Sun"):
                continue
            lo, hi = self.damage_range(m, to, mv)
            best = max(best, (lo + hi) / 2 * accuracy(mv, self.weather))
        return best

    @staticmethod
    def ko_chance(lo, hi, hp):
        if hi <= 0:
            return 0.0
        if lo >= hp:
            return 1.0
        if hi < hp:
            return 0.0
        return (hi - hp) / (hi - lo) if hi > lo else 1.0

    def fake_out_blocked(self, target, foes):
        return ("Ghost" in target.s.types or target.ability == "Inner Focus"
                or (self.terrain == "Psychic" and target.grounded())
                or any(x.ability in PRIORITY_BLOCKERS for x in foes))

    def attack_options(self, m, foes, ally, planned):
        first = m.turns_active == 0
        for mv in m.s.moves:
            info = move_info(mv)
            if info["category"] == "Status" or mv in NO_SELECT:
                continue
            if mv in ("Fake Out", "First Impression") and not first:
                continue
            if (mv == "Steel Roller" and not self.terrain) or (mv in SUN_ONLY and self.weather != "Sun"):
                continue
            acc = accuracy(mv, self.weather)
            for targets in ([foes] if info["spread"] else [[f] for f in foes]):
                value = 0.0
                for t in targets:
                    lo, hi = self.damage_range(m, t, mv)
                    if info["spread"] and len(foes) == 1:
                        lo, hi = lo / 0.75, hi / 0.75
                    remaining = max(1.0, t.hp - planned.get(id(t), 0.0))
                    if t.item == "Focus Sash" and t.hp >= 100 and lo < 200:
                        lo, hi = min(lo, 99.0), min(hi, 99.0)
                    value += min(remaining, (lo + hi) / 2) * acc / 100 + 0.6 * self.ko_chance(lo, hi, remaining) * acc
                    if mv == "Fake Out" and not self.fake_out_blocked(t, foes):
                        if ally:
                            value += 0.35 + max(self.threat(t, ally), self.threat(t, m)) / 120
                            if any(x in BOOST for x in ally.s.moves):
                                value += 0.4  # buys the setup partner a free turn
                        else:
                            value += 0.35 + self.threat(t, m) / 150
                if mv in ALLY_SPREAD and ally:
                    lo, hi = self.damage_range(m, ally, mv)
                    value -= 1.2 * (lo + hi) / 2 / 100 + self.ko_chance(lo, hi, ally.hp)
                if mv == "Sucker Punch":
                    value *= 0.8
                if info["priority"] > 0 and mv != "Fake Out":
                    value *= 1.1
                yield value, mv, targets

    def status_options(self, m, foes, ally, incoming):
        first = m.turns_active == 0
        my_speed = self.speed(m)
        for mv in m.s.moves:
            if move_info(mv)["category"] != "Status":
                continue
            value, targets = 0.0, []
            if mv in PROTECTS:
                if not m.last_protect and incoming >= 0.9 * m.hp:
                    value = 0.95
                elif not m.last_protect and first and incoming >= 0.8 * m.hp:
                    value = 0.3
            elif mv in BOOST:
                b = BOOST[mv]
                key = max(b, key=lambda k: b[k])
                if m.boosts[key] < 2 and incoming < 0.55 * m.hp:
                    value = 0.9 if first or m.boosts[key] == 0 else 0.6
                if mv == "Agility" and all(self.speed(f) < my_speed for f in foes):
                    value = 0.0
            elif mv == "Tailwind":
                slowest = min(self.speed(x) for x in self.act(m.side))
                if self.tailwind[m.side] <= 0 and any(self.speed(f) > slowest for f in foes):
                    value = 1.1
            elif mv == "Trick Room":
                ours = self.act(m.side)
                mine = sum(self.speed(x) for x in ours) / len(ours)
                theirs = sum(self.speed(f) for f in foes) / len(foes)
                if self.trick_room <= 0 and mine < theirs:
                    value = 1.2
            elif mv in ("Follow Me", "Rage Powder"):
                if ally and ((any(x in BOOST for x in ally.s.moves) and ally.boosts["atk"] + ally.boosts["spa"] < 2)
                             or ally.hp < 50):
                    value = 0.85
                elif ally and first:
                    value = 0.4
            elif mv in ("Reflect", "Light Screen", "Aurora Veil"):
                timer = {"Reflect": self.reflect, "Light Screen": self.light_screen, "Aurora Veil": self.veil}[mv]
                if timer[m.side] <= 0 and (mv != "Aurora Veil" or self.weather == "Snow"):
                    value = 0.8
            elif mv in SLEEP_MOVES:
                options = [f for f in foes if not f.status and "Grass" not in f.s.types and f.item != "Safety Goggles"
                           and f.ability not in ("Overcoat", "Good as Gold", "Insomnia", "Vital Spirit")]
                if options:
                    targets, value = [max(options, key=lambda f: self.threat(f, m))], 1.4 * accuracy(mv)
            elif mv == "Will-O-Wisp":
                options = [f for f in foes if not f.status and "Fire" not in f.s.types and f.ability != "Good as Gold"
                           and any(move_info(x)["category"] == "Physical" for x in f.s.moves)]
                if options:
                    targets, value = [max(options, key=lambda f: self.threat(f, m))], 0.75
            elif mv == "Thunder Wave":
                options = [f for f in foes if not f.status and not {"Electric", "Ground"} & set(f.s.types)
                           and self.speed(f) > my_speed and f.ability != "Good as Gold"]
                if options:
                    targets, value = [max(options, key=lambda f: self.threat(f, m))], 0.5
            elif mv == "Helping Hand":
                value = 0.35 if ally else 0.0
            elif mv == "Parting Shot":
                if self.bench[m.side] and incoming >= 0.7 * m.hp and not (first and "Fake Out" in m.s.moves):
                    targets, value = [max(foes, key=lambda f: self.threat(f, m))], 0.45
            elif mv == "Wide Guard":
                if not m.last_protect and any(move_info(x)["spread"] and move_info(x)["category"] != "Status"
                                              for f in foes for x in f.s.moves):
                    value = 0.4
            elif mv == "Haze":
                if any(sum(max(0, b) for b in f.boosts.values()) >= 2 for f in foes):
                    value = 0.85
            elif mv == "Taunt":
                options = [f for f in foes if f.taunt <= 0 and any(
                    move_info(x)["category"] == "Status" and x not in PROTECTS for x in f.s.moves)]
                if options and first:
                    targets, value = [options[0]], 0.5
            elif mv in HEAL:
                value = 0.6 if m.hp < 50 else 0.0
            if m.taunt > 0 and mv not in PROTECTS:
                value = 0.0
            if value > 0:
                yield value, mv, targets

    def choose(self, m, planned):
        foes, ally = self.foes(m), self.ally(m)
        if not foes:
            return None
        # Expected damage coming at m, minus whatever the partner is Faking Out or redirecting this turn.
        fake_out_target, redirector = planned.get(("fake out", m.side)), planned.get(("redirect", m.side))
        incoming = sum(self.threat(f, m) for f in foes if id(f) != fake_out_target) * (0.75 if ally else 1.0)
        if redirector is not None and redirector is not m:
            incoming *= 0.35
        options = sorted(list(self.attack_options(m, foes, ally, planned)) +
                         list(self.status_options(m, foes, ally, incoming)), key=lambda o: -o[0])
        if m.locked and m.item in CHOICE_ITEMS:
            options = [o for o in options if o[1] == m.locked] or [(0.0, m.locked, foes[:1])]
        if not options:
            return "Struggle", []
        # Mostly the best option; now and then a close second, so the AI isn't fully predictable.
        pick = options[0]
        if len(options) > 1 and options[1][0] >= 0.5 * options[0][0] and self.rng.random() >= 0.9:
            pick = options[1]
        _, mv, targets = pick
        if move_info(mv)["category"] != "Status":
            for t in targets:
                lo, hi = self.damage_range(m, t, mv)
                planned[id(t)] = planned.get(id(t), 0.0) + (lo + hi) / 2
        if mv == "Fake Out" and targets:
            planned[("fake out", m.side)] = id(targets[0])
        if mv in ("Follow Me", "Rage Powder"):
            planned[("redirect", m.side)] = m
        return mv, targets

    # ---- turn
    def run_turn(self):
        self.turn += 1
        actions, planned = {}, {}
        for side in (0, 1):
            def decides_first(x):  # supports commit first, so partners can plan around them
                redirects = {"Follow Me", "Rage Powder"} & set(x.s.moves)
                support = (x.turns_active == 0 and "Fake Out" in x.s.moves) or redirects
                return 0 if support else 1, -self.speed(x)
            for m in sorted(self.act(side), key=decides_first):
                if m.charging:
                    actions[id(m)] = (m, *m.charging)
                    continue
                choice = self.choose(m, planned)
                if choice:
                    actions[id(m)] = (m, *choice)
        for m, _, _ in actions.values():
            m.protected = m.flinched = False
        order = sorted(actions.values(), key=lambda a: (-self.priority(a[0], a[1]),
                                                        (1 if self.trick_room > 0 else -1) * self.speed(a[0]),
                                                        self.rng.random()))
        turn = {"redirect": {0: None, 1: None}, "wide": {0: False, 1: False}, "quick": {0: False, 1: False},
                "helping": set(), "moved": set(), "order": order}
        for m, mv, targets in order:
            if m.fainted:
                continue
            turn["moved"].add(id(m))
            if m.flinched:
                self.log(f"  {m.name} flinched")
                m.last_protect = False
                continue
            if m.status == "slp":
                m.sleep -= 1
                if m.sleep > 0:
                    self.log(f"  {m.name} is asleep")
                    continue
                m.status = ""
            if m.status == "par" and self.rng.random() < 0.25:
                self.log(f"  {m.name} is paralysed")
                continue
            self.use(m, mv, targets, turn)
        self.end_turn()

    def use(self, m, mv, targets, turn):
        info = move_info(mv)
        foes, ally = self.foes(m), self.ally(m)
        if m.item in CHOICE_ITEMS:
            m.locked = mv
        if mv in PROTECTS:
            ok = not m.last_protect or self.rng.random() < 1 / 3
            m.protected = m.last_protect = ok
            self.log(f"  {m.name} {mv} {'ok' if ok else 'failed'}")
            return
        m.last_protect = False
        if info["category"] == "Status":
            self.use_status(m, mv, targets, foes, ally, turn)
            return
        if mv == "Fake Out" and m.turns_active > 0:
            return
        if (mv == "Steel Roller" and not self.terrain) or (mv in SUN_ONLY and self.weather != "Sun"):
            return
        if mv == "Electro Shot":
            if self.weather != "Rain" and not m.charging:
                m.boost("spa", 1)
                m.charging = (mv, targets)
                self.log(f"  {m.name} charges Electro Shot")
                return
            if self.weather == "Rain":
                m.boost("spa", 1)
            m.charging = None
        spread = info["spread"]
        if spread:
            hit = list(foes) + ([ally] if mv in ALLY_SPREAD and ally else [])
        else:
            t = targets[0] if targets else None
            if t is None or t.fainted:
                t = foes[0] if foes else None
            if t is None:
                return
            redirect = turn["redirect"][1 - m.side]
            if (redirect and not redirect[0].fainted and redirect[0] is not t
                    and m.ability not in ("Stalwart", "Propeller Tail")
                    and not (redirect[1] == "Rage Powder" and ("Grass" in m.s.types or m.ability == "Overcoat"
                                                               or m.item == "Safety Goggles"))):
                t = redirect[0]
            hit = [t]
        dealt, recoil_contact = 0.0, 0.0
        for t in hit:
            if t.fainted:
                continue
            if t.protected:
                self.log(f"  {t.name} protected from {m.name}'s {mv}")
                continue
            if spread and turn["wide"][t.side]:
                continue
            if self.priority(m, mv) > 0 and t.side != m.side and (
                    turn["quick"][t.side] or (self.terrain == "Psychic" and t.grounded())
                    or any(x.ability in PRIORITY_BLOCKERS for x in self.act(t.side))):
                continue
            if mv == "Sucker Punch":
                target_action = next((a for a in turn["order"] if a[0] is t), None)
                if id(t) in turn["moved"] or not target_action or move_info(target_action[1])["category"] == "Status":
                    continue
            hit_chance = accuracy(mv, self.weather) * (1.1 if m.item == "Wide Lens" else 1.0)
            if m.ability != "No Guard" and self.rng.random() > hit_chance:
                self.log(f"  {m.name} {mv} missed {t.name}")
                continue
            lo, hi = self.damage_range(m, t, mv, helping_hand=id(m) in turn["helping"])
            if spread and len([x for x in hit if not x.fainted]) == 1:
                lo, hi = lo / 0.75, hi / 0.75
            if hi <= 0:
                continue
            d = self.rng.uniform(lo, hi)
            if self.rng.random() < CRIT_RATE:
                d *= 1.5
            if t.item == "Focus Sash" and t.hp >= 100 and d >= 100:
                d = 99.0
                t.lose_item()
            d = min(d, t.hp)
            t.hp -= d
            t.hits_taken += 1
            dealt += d * t.s.max_hp / 100
            self.log(f"  {m.name} {mv} -> {t.name} {d:.0f}% (hp {t.hp:.0f})")
            if move_info(mv).get("contact"):
                recoil_contact += CONTACT_PUNISH.get(t.ability, 0.0) + (1 / 6 if t.item == "Rocky Helmet" else 0.0)
            if RESIST_BERRIES.get(t.item) == move_info(mv).get("type"):
                t.lose_item()  # the calc halved this hit; the berry is gone now
            if t.hp <= 0:
                self.faint(t)
                continue
            if t.item == "Sitrus Berry" and t.hp <= 50:
                t.hp = min(100.0, t.hp + 25)
                t.lose_item()
            if mv == "Fake Out" and t.ability != "Inner Focus":
                t.flinched = True
            if (mv in FLINCH_CHANCE and id(t) not in turn["moved"] and t.ability != "Inner Focus"
                    and self.rng.random() < FLINCH_CHANCE[mv]):
                t.flinched = True
            for stat, n in TARGET_DROP.get(mv, {}).items():
                t.boost(stat, n, by_foe=True)
            if mv in STATUS_CHANCE and not t.status:
                status, chance = STATUS_CHANCE[mv]
                immune = (status == "brn" and "Fire" in t.s.types) or (status == "par" and "Electric" in t.s.types)
                if not immune and self.rng.random() < chance:
                    t.set_status(status)
            if mv == "Knock Off" and t.item and not t.item.endswith("ite") and "ite " not in t.item:
                t.lose_item()
        if recoil_contact:
            m.hp -= 100 * recoil_contact
            if m.hp <= 0:
                self.faint(m)
                return
        if dealt > 0:
            if mv in RECOIL and m.ability != "Rock Head":
                m.hp -= RECOIL[mv] * dealt / m.s.max_hp * 100
            if mv in DRAIN:
                m.hp = min(100.0, m.hp + DRAIN[mv] * dealt / m.s.max_hp * 100)
            if m.item == "Life Orb" and m.ability != "Magic Guard":
                m.hp -= 10
            for stat, n in {**SELF_DROP.get(mv, {}), **SELF_BOOST_ON_HIT.get(mv, {})}.items():
                m.boost(stat, n)
            if mv in ALL_STATS_CHANCE and self.rng.random() < 0.1:
                for stat in ("atk", "def", "spa", "spd", "spe"):
                    m.boost(stat, 1)
            if mv == "Steel Roller":
                self.terrain, self.terrain_turns = None, 0
            if m.hp <= 0:
                self.faint(m)
            elif m.item == "Sitrus Berry" and m.hp <= 50:
                m.hp = min(100.0, m.hp + 25)
                m.lose_item()

    def use_status(self, m, mv, targets, foes, ally, turn):
        side = m.side
        if mv in BOOST:
            for stat, n in BOOST[mv].items():
                m.boost(stat, 2 * n if mv == "Growth" and self.weather == "Sun" else n)
            self.log(f"  {m.name} {mv} -> {dict(m.boosts)}")
        elif mv == "Tailwind":
            self.tailwind[side] = 4
            self.log(f"  {m.name} Tailwind")
        elif mv == "Trick Room":
            self.trick_room = 0 if self.trick_room > 0 else 5
            self.log(f"  {m.name} Trick Room")
        elif mv in ("Follow Me", "Rage Powder"):
            turn["redirect"][side] = (m, mv)
            self.log(f"  {m.name} {mv}")
        elif mv in ("Reflect", "Light Screen", "Aurora Veil"):
            timer = {"Reflect": self.reflect, "Light Screen": self.light_screen, "Aurora Veil": self.veil}[mv]
            timer[side] = 8 if m.item == "Light Clay" else 5
        elif mv == "Helping Hand" and ally:
            turn["helping"].add(id(ally))
        elif mv == "Wide Guard":
            turn["wide"][side] = True
        elif mv == "Quick Guard":
            turn["quick"][side] = True
        elif mv == "Haze":
            for x in self.act(0) + self.act(1):
                x.boosts = Counter()
        elif mv in HEAL:
            m.hp = min(100.0, m.hp + 50)
        elif targets:
            t = targets[0]
            if t.fainted or t.protected or t.ability == "Good as Gold":
                return
            if m.ability == "Prankster" and "Dark" in t.s.types:
                return
            if self.rng.random() > accuracy(mv, self.weather) * (1.1 if m.item == "Wide Lens" else 1.0):
                self.log(f"  {m.name} {mv} missed {t.name}")
                return
            if mv in SLEEP_MOVES and not t.status:
                t.set_status("slp", 3)
                self.log(f"  {m.name} {mv} -> {t.name} falls asleep")
            elif mv == "Will-O-Wisp" and not t.status:
                t.set_status("brn")
            elif mv == "Thunder Wave" and not t.status:
                t.set_status("par")
            elif mv == "Taunt":
                t.taunt = 3
            elif mv == "Parting Shot":
                for f in foes:
                    f.boost("atk", -1, by_foe=True)
                    f.boost("spa", -1, by_foe=True)
                self.log(f"  {m.name} Parting Shot")
                self.switch_out(m)

    def faint(self, m):
        if not m.fainted:
            m.fainted, m.hp = True, 0.0
            self.fainted_count[m.side] += 1
            self.log(f"  {m.name} fainted")

    def switch_out(self, m):
        side = m.side
        if not self.bench[side]:
            return
        slot = next(k for k, i in enumerate(self.active[side]) if i is not None and self.sides[side][i] is m)
        new = self.bench[side].pop(0)
        self.bench[side].append(self.sides[side].index(m))
        m.boosts, m.turns_active, m.locked = Counter(), 0, None
        self.active[side][slot] = new
        self.enter(self.sides[side][new])

    def end_turn(self):
        for m in self.act(0) + self.act(1):
            if (self.weather == "Sand" and not {"Rock", "Ground", "Steel"} & set(m.s.types)
                    and m.ability not in ("Overcoat", "Magic Guard", "Sand Rush", "Sand Force")):
                m.hp -= 6.25
            if m.status == "brn":
                m.hp -= 6.25
            if m.item == "Leftovers":
                m.hp = min(100.0, m.hp + 6.25)
            if self.terrain == "Grassy" and m.grounded():
                m.hp = min(100.0, m.hp + 6.25)
            if m.hp <= 0:
                self.faint(m)
            m.turns_active += 1
            m.taunt = max(0, m.taunt - 1)
        for timer in (self.tailwind, self.reflect, self.light_screen, self.veil):
            for s in (0, 1):
                timer[s] = max(0, timer[s] - 1)
        if self.weather_turns > 0:
            self.weather_turns -= 1
            if self.weather_turns == 0:
                self.weather = None
        if self.terrain_turns > 0:
            self.terrain_turns -= 1
            if self.terrain_turns == 0:
                self.terrain = None
        self.trick_room = max(0, self.trick_room - 1)
        for side in (0, 1):
            for slot, i in enumerate(self.active[side]):
                if i is not None and self.sides[side][i].fainted:
                    if self.bench[side]:
                        new = self.bench[side].pop(0)
                        self.active[side][slot] = new
                        self.log(f"  side {side} sends in {self.sides[side][new].name}")
                        self.enter(self.sides[side][new])
                    else:
                        self.active[side][slot] = None

    def run(self) -> float:
        """Play to the end. 1 = side 0 wins, 0 = side 1 wins; at the turn limit, more HP left wins."""
        for side in (0, 1):
            for m in sorted(self.act(side), key=lambda x: -self.speed(x)):
                self.enter(m)
        while self.alive(0) and self.alive(1) and self.turn < self.max_turns:
            self.log(f"Turn {self.turn + 1}: " + " | ".join(f"{m.name} {m.hp:.0f}%" for m in self.act(0)) + "  vs  "
                     + " | ".join(f"{m.name} {m.hp:.0f}%" for m in self.act(1)))
            self.run_turn()
        if self.alive(0) and not self.alive(1):
            return 1.0
        if self.alive(1) and not self.alive(0):
            return 0.0
        a = sum(m.hp for m in self.alive(0)) / len(self.sides[0])
        b = sum(m.hp for m in self.alive(1)) / len(self.sides[1])
        return 1.0 if a > b else (0.5 if a == b else 0.0)


# ---------------------------------------------------------------------------------------------------- opponents

def opponent_teams(n: int, seed: int = 7) -> list[tuple[str, list[str]]]:
    """Up to n real teams as (pool, species): every official Regional top 32 (up to n/2), the rest a random
    sample of recent online top 8s."""
    def teams(where):
        out = defaultdict(list)
        for t, p, mon in db().execute(
                "SELECT m.tournament, m.player, m.pokemon FROM team_members m "
                "JOIN tournaments t ON t.id = m.tournament "
                f"WHERE t.regulation = ? AND {where} ORDER BY m.tournament, m.player, m.slot", (REGULATION,)):
            out[(t, p)].append(mon)
        return list(out.values())
    official = teams("t.organizer = 'rk9' AND m.placing <= 32")[: n // 2]
    online = teams(f"t.organizer != 'rk9' AND t.date >= '{RECENT}' AND m.placing <= 8")
    online = random.Random(seed).sample(online, min(len(online), n - len(official)))
    return [("official", t) for t in official] + [("online", t) for t in online]


def bring(team: list[Spec], n: int, rng: random.Random, fixed=()) -> list[Spec]:
    """n Pokemon from a team: `fixed` first, the rest at random, with at most one Mega (only one can Mega Evolve)."""
    chosen = list(fixed)
    megas = sum("-Mega" in s.name for s in chosen)
    for s in rng.sample([s for s in team if s not in chosen], len(team) - len(chosen)):
        if len(chosen) == n:
            break
        if "-Mega" in s.name:
            if megas:
                continue
            megas += 1
        chosen.append(s)
    return chosen


def play_core(core: list[Spec], team: list[Spec], seed: int, reps: int = 2) -> list[float]:
    """Results of `reps` battles: the core against as many of the team's Pokemon, picked at random."""
    results = []
    for rep in range(reps):
        rng = random.Random(seed * 1000 + rep)
        opp = bring(team, len(core), rng)
        results.append(Battle(core, opp, random.Random(rng.randrange(10**9))).run())
    return results


# ---------------------------------------------------------------------------------------------------- CLI

def _lookup(names: str) -> list[Spec]:
    out = []
    for name in names.split(","):
        spec = real_spec(name.strip())
        if spec is None:
            sys.exit(f"No tournament set for {name.strip()}")
        out.append(spec)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("battle", help="one battle with a turn-by-turn log")
    b.add_argument("side_a")
    b.add_argument("side_b")
    b.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("core", help="win rate of a core against real teams")
    c.add_argument("core", help="comma-separated; the first two lead")
    c.add_argument("--teams", type=int, default=400)
    c.add_argument("--reps", type=int, default=2)
    args = parser.parse_args(argv)
    if args.cmd == "battle":
        side_a, side_b = _lookup(args.side_a), _lookup(args.side_b)
        for s in side_a + side_b:
            print(s)
        battle = Battle(side_a, side_b, random.Random(args.seed), log=True)
        result = battle.run()
        print("\n".join(battle.lines))
        print("side A wins" if result == 1 else "side B wins" if result == 0 else "draw")
        return
    core = _lookup(args.core)
    for s in core:
        print(s)
    by_pool = defaultdict(list)
    for i, (pool, names) in enumerate(opponent_teams(args.teams)):
        team = [s for s in (real_spec(n) for n in names) if s is not None]
        if len(team) >= len(core):
            by_pool[pool] += play_core(core, team, seed=i, reps=args.reps)
    every = [r for v in by_pool.values() for r in v]
    se = (sum((r - sum(every) / len(every)) ** 2 for r in every) / len(every)) ** 0.5 / len(every) ** 0.5
    print(f"win rate {100 * sum(every) / len(every):.1f}% (+/- {100 * se:.1f}) over {len(every)} battles; "
          + ", ".join(f"{pool} {100 * sum(v) / len(v):.1f}%" for pool, v in by_pool.items()))


if __name__ == "__main__":
    main()
