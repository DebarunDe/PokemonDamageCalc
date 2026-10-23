"""Synergy model: which Pokemon work well together, learned from tournament results, so it can suggest cores
that haven't been played much (or at all).

Each set a Pokemon is commonly run with (e.g. Choice Scarf vs Bulk Up Annihilape) is its own unit, with its
strength shrunk toward its species'; --per-species treats each species as one. Each unit gets a role profile from
its moves, item, ability and base stats: support tools (Fake Out, redirection, Intimidate,
screens, ...), speed control and speed tier, weather and terrain (setter vs user), offense (setup, priority, spread,
physical / special) and bulk. A team's result (wins per game above its player's skill, as in signals.py) is
modelled as

    individual strength of each member
    + for every pair on the team: role x role synergy (e.g. Fake Out x setup, Trick Room x slow, sun x Chlorophyll)
      + type fit (weaknesses they share, weaknesses one covers for the other)
      + optionally a learned "style" match (latent factors, as in recommender systems; off by default because it
        did not generalise to held-out pairs)

fitted by weighted least squares (weights = games) with ridge penalties. Because synergy comes from roles rather
than a table of observed pairs, the model can score pairs it has rarely or never seen. Validation (README) shows
most of the gain comes from scoring sets rather than species; the role synergy adds only a little on top.

Usage:
    python analysis/synergy.py validate      # does it predict held-out pairs and later events? (~15 min)
    python analysis/synergy.py sets          # every set's individual strength: the under-used strong sets
    python analysis/synergy.py suggest       # pairs ranked by how sure it is of their synergy

Results go to analysis/output/synergy_*.csv (not committed).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from signals import player_skill

DB = Path(__file__).parent / "data" / "meta.sqlite"
OUT = Path(__file__).parent / "output"
REGULATION = "M-C"
RECENT = "2026-09-26"

# ---------------------------------------------------------------------------------------------------- roles

TYPES = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic",
         "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]
# attacking type -> {defending type: multiplier} (only the entries that aren't 1)
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
    "Bug": {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5, "Flying": .5, "Psychic": 2, "Ghost": .5, "Dark": 2,
            "Steel": .5, "Fairy": .5},
    "Rock": {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5, "Flying": 2, "Bug": 2, "Steel": .5},
    "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon": {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark": {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5, "Fairy": .5},
    "Steel": {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2, "Rock": 2, "Steel": .5, "Fairy": 2},
    "Fairy": {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2, "Dark": 2, "Steel": .5},
}
ROLES = ["fake_out", "redirection", "intimidate", "tailwind", "trick_room", "screens", "sleep", "pivot", "wide_guard",
         "helping_hand", "sun", "rain", "sand", "snow", "sun_user", "rain_user", "sand_user", "snow_user",
         "psychic_terrain", "psychic_terrain_user", "grassy_terrain", "grassy_terrain_user", "electric_terrain",
         "electric_terrain_user", "setup", "priority", "spread", "physical", "special", "fast", "slow", "bulky",
         "protect",
         "mega", "choice"]
MOVE_ROLES = {
    "fake_out": {"Fake Out"}, "redirection": {"Follow Me", "Rage Powder"}, "tailwind": {"Tailwind"},
    "trick_room": {"Trick Room"}, "screens": {"Reflect", "Light Screen", "Aurora Veil"},
    "sleep": {"Spore", "Sleep Powder", "Hypnosis", "Yawn"},
    "pivot": {"Parting Shot", "U-turn", "Volt Switch", "Flip Turn"},
    "wide_guard": {"Wide Guard", "Quick Guard"}, "helping_hand": {"Helping Hand"},
    "sun_user": {"Solar Beam", "Solar Blade"}, "rain_user": {"Electro Shot", "Hurricane", "Thunder"},
    "snow_user": {"Blizzard", "Aurora Veil"},
    "psychic_terrain_user": {"Expanding Force"}, "grassy_terrain_user": {"Grassy Glide"},
    "electric_terrain_user": {"Rising Voltage"},
    "setup": {"Swords Dance", "Nasty Plot", "Dragon Dance", "Calm Mind", "Bulk Up", "Quiver Dance", "Shell Smash",
              "Belly Drum", "Coil", "Victory Dance", "Tidy Up", "Clangorous Soul", "Growth", "Agility", "Iron Defense"},
    "protect": {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap", "Burning Bulwark"},
}
WEATHER = {"Drought": "sun", "Drizzle": "rain", "Sand Stream": "sand", "Snow Warning": "snow"}
WEATHER_USERS = {"Chlorophyll": "sun_user", "Swift Swim": "rain_user", "Sand Rush": "sand_user",
                 "Sand Force": "sand_user", "Slush Rush": "snow_user"}
CHOICE = {"Choice Scarf", "Choice Band", "Choice Specs"}
TERRAIN = {"Grassy Surge": "grassy_terrain", "Psychic Surge": "psychic_terrain", "Electric Surge": "electric_terrain"}


def weakness_profile(types, ability):
    out = {}
    for atk in TYPES:
        m = 1.0
        for t in types:
            m *= CHART[atk].get(t, 1.0)
        if atk == "Ground" and ability == "Levitate":
            m = 0.0
        out[atk] = m
    return out


class Roster:
    """Role profiles of every Pokemon on M-C tournament teams, from their real sets and the calc's data.

    By default each species is one unit, with the moves on at least a quarter of its sets. With per_set=True each
    distinct role profile a species is run with (e.g. Choice Scarf vs Bulk Up Annihilape) on at least `min_set`
    tournament sets is its own unit; rarer sets join the nearest common one. `names` are the units, `species_of`
    maps each to its species."""

    def __init__(self, db, per_set=False, min_set=30):
        from champcalc import CalcError, Calculator
        self.calc, self.per_set = Calculator(), per_set
        self._moves = {}
        rows = db.execute("SELECT m.pokemon, m.ability, m.item, m.moves FROM team_members m JOIN tournaments t "
                          "ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)).fetchall()
        sets = defaultdict(list)
        for name, ability, item, moves in rows:
            sets[name].append((ability or "", item or "", frozenset(json.loads(moves) if moves else [])))
        self.names, self.roles, self.weak, self.species_of = [], {}, {}, {}
        self.sp_names, self._unit = [], {}
        for name, entries in sets.items():
            try:
                sp = self.calc.species(name)
            except CalcError:  # names the calc doesn't know, e.g. entered in another language
                continue
            pre = Counter(a for a, _, _ in entries).most_common(1)[0][0]
            ability = pre if pre in sp["abilities"] or not sp["abilities"] else sp["abilities"][0]
            self.sp_names.append(name)
            self.weak[name] = weakness_profile(sp["types"], ability)
            if not per_set:
                moves = Counter(m for _, _, ms in entries for m in ms)
                common = {m for m, k in moves.items() if k >= 0.25 * len(entries)}
                choice = sum(i in CHOICE for _, i, _ in entries) >= 0.5 * len(entries)
                self._add(name, name, self._roles(name, sp, ability, pre, common, choice))
                continue
            keys = Counter(tuple(self._roles(name, sp, ability, pre, ms, i in CHOICE)) for _, i, ms in entries)
            main = keys.most_common(1)[0][0]
            kept = [k for k, c in keys.most_common() if c >= min_set or k == main]
            nearest = {k: min(kept, key=lambda v: (sum(a != b for a, b in zip(k, v)), -keys[v])) for k in keys}
            for k in kept:
                diff = " ".join(("+" if a else "-") + ROLES[n] for n, (a, b) in enumerate(zip(k, main)) if a != b)
                self._add(f"{name} [{diff}]" if diff else name, name, np.array(k))
            for k, v in nearest.items():
                self._unit[(name, k)] = self._unit[(name, v)]
            self._unit[(name, "sp")] = (sp, ability, pre)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.sp_index = {n: i for i, n in enumerate(self.sp_names)}

    def _add(self, unit, species, roles):
        self.names.append(unit)
        self.species_of[unit] = species
        self.roles[unit] = roles
        self._unit[(species, tuple(roles))] = unit
        self._unit.setdefault((species, None), unit)

    def _move(self, m):
        if m not in self._moves:
            try:
                self._moves[m] = self.calc.move(m)
            except Exception:
                self._moves[m] = None
        return self._moves[m]

    def _roles(self, name, sp, ability, pre, moves, choice):
        base = sp["base_stats"]
        r = {role: float(bool(moves & MOVE_ROLES.get(role, set()))) for role in ROLES}
        r["intimidate"] = float(ability == "Intimidate" or pre == "Intimidate")
        for ab, w in WEATHER.items():
            r[w] = max(r[w], float(ability == ab))
        if name in ("Charizard-Mega-Y",):
            r["sun"] = 1.0
        if ability in WEATHER_USERS:
            r[WEATHER_USERS[ability]] = 1.0
        if ability in TERRAIN:
            r[TERRAIN[ability]] = 1.0
        damaging = [i for i in map(self._move, moves) if i and i["category"] != "Status"]
        r["priority"] = float(any(i["priority"] > 0 and i["name"] != "Fake Out" for i in damaging))
        r["spread"] = float(any(i["spread"] for i in damaging))
        phys = sum(i["category"] == "Physical" for i in damaging)
        spec = sum(i["category"] == "Special" for i in damaging)
        r["physical"], r["special"] = float(phys > spec), float(spec > phys)
        r["fast"], r["slow"] = float(base["spe"] >= 100), float(base["spe"] <= 60)
        r["bulky"] = float((base["hp"] * base["def"] * base["spd"]) ** (1 / 3) >= 95)
        r["mega"] = float("-Mega" in name)
        r["choice"] = float(choice)
        return np.array([r[k] for k in ROLES])

    def unit(self, name, item, moves):
        """The unit a tournament set belongs to (None for species the calc doesn't know)."""
        if name not in self.weak:
            return None
        if not self.per_set:
            return name
        sp, ability, pre = self._unit[(name, "sp")]
        key = tuple(self._roles(name, sp, ability, pre, frozenset(moves), item in CHOICE))
        return self._unit.get((name, key), self._unit[(name, None)])

    def type_fit(self, a, b):
        """(weaknesses they share, weaknesses one resists for the other) over the 18 attacking types."""
        wa, wb = self.weak[self.species_of.get(a, a)], self.weak[self.species_of.get(b, b)]
        shared = sum(wa[t] > 1 and wb[t] > 1 for t in TYPES)
        covered = sum((wa[t] > 1 and wb[t] < 1) or (wb[t] > 1 and wa[t] < 1) for t in TYPES)
        return shared, covered


# ---------------------------------------------------------------------------------------------------- data

def load_teams(db, roster):
    """Teams as (unit indices, wins above skill per game, games, tournament id, date, species indices)."""
    skill = player_skill(db).set_index(["tournament", "player"])["skill"].to_dict()
    members, species, record = defaultdict(set), defaultdict(set), {}
    for t, p, mon, item, moves, w, l, date in db.execute(
            "SELECT m.tournament, m.player, m.pokemon, m.item, m.moves, m.wins, m.losses, t.date FROM team_members m "
            "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)):
        unit = roster.unit(mon, item or "", json.loads(moves) if moves else [])
        if unit is not None:
            members[(t, p)].add(roster.index[unit])
            species[(t, p)].add(roster.sp_index[mon])
        record[(t, p)] = (w or 0, l or 0, date)
    teams = []
    for key, idx in members.items():
        w, l, date = record[key]
        if w + l and key in skill and len(species[key]) >= 4:
            teams.append((sorted(idx), (w - skill[key] * (w + l)) / (w + l), w + l, key[0], date,
                          sorted(species[key])))
    return teams


class Design:
    """Per-team sums of the model's features, so fitting is fast."""

    def __init__(self, roster, teams):
        S, K = len(roster.names), len(ROLES)
        self.S, self.K = S, K
        self.iu = np.triu_indices(K)
        self.T = len(teams)
        self.y = np.array([t[1] for t in teams])
        self.g = np.array([t[2] for t in teams], float)
        R = np.array([roster.roles[n] for n in roster.names])
        self.role_pairs = np.zeros((self.T, len(self.iu[0])))
        self.typefit = np.zeros((self.T, 2))
        fit_cache = {}
        rows, cols = [], []
        for k, (idx, *_rest) in enumerate(teams):
            rows += [k] * len(idx)
            cols += list(idx)
            s = R[idx].sum(0)
            m = np.outer(s, s) - R[idx].T @ R[idx]  # sum over member pairs of both orders of r_i r_j^T
            self.role_pairs[k] = m[self.iu]
            for i, j in itertools.combinations(idx, 2):
                if (i, j) not in fit_cache:
                    fit_cache[(i, j)] = roster.type_fit(roster.names[i], roster.names[j])
                self.typefit[k] += fit_cache[(i, j)]
        self.member = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.T, S))
        # per-set units: each unit's strength is its species' strength plus its own deviation
        self.hier = roster.per_set
        to_sp = sparse.csr_matrix((np.ones(S), (range(S), [roster.sp_index[roster.species_of[n]]
                                                            for n in roster.names])), shape=(S, len(roster.sp_names)))
        self.smember = self.member @ to_sp
        self.Ssp = len(roster.sp_names) if self.hier else 0
        self.to_sp = to_sp


class Model:
    def __init__(self, k=4, lam_b=30.0, lam_d=30.0, lam_w=10000.0, lam_v=10.0, roles=True, latent=False, seed=0):
        self.k, self.lam_b, self.lam_d, self.lam_w, self.lam_v = k, lam_b, lam_d, lam_w, lam_v
        self.use_roles, self.use_latent, self.seed = roles, latent and k > 0, seed

    def _sizes(self, d):
        return d.S, d.Ssp, len(d.iu[0])

    def _unpack(self, x, d):
        """mu, unit strengths (species part + own part), role weights, type-fit weights, styles, species part."""
        S, Q, P = self._sizes(d)
        mu, b = x[0], x[1:1 + S]
        bs = x[1 + S:1 + S + Q]
        o = 1 + S + Q
        w, t = x[o:o + P], x[o + P:o + P + 2]
        V = x[o + P + 2:].reshape(S, self.k) if self.use_latent else np.zeros((S, 0))
        return mu, b + (d.to_sp @ bs if Q else 0), w, t, V, bs

    def _predict(self, x, d):
        mu, b, w, t, V, _ = self._unpack(x, d)
        pred = mu + d.member @ b
        if self.use_roles:
            pred = pred + d.role_pairs @ w + d.typefit @ t
        if self.use_latent:
            s = d.member @ V
            pred = pred + 0.5 * ((s ** 2).sum(1) - d.member @ (V ** 2).sum(1))
        return pred

    def fit(self, d):
        S, Q, P = self._sizes(d)
        o = 1 + S + Q
        n = o + P + 2 + (S * self.k if self.use_latent else 0)
        x0 = np.zeros(n)
        if self.use_latent:
            x0[o + P + 2:] = np.random.default_rng(self.seed).normal(0, 0.1, S * self.k)
        x0[0] = np.average(d.y, weights=d.g)
        wsum = d.g.sum()
        # with species parts, a unit's own part is shrunk toward its species (lam_d), the species toward 0 (lam_b)
        lam_unit = self.lam_d if Q else self.lam_b

        def loss(x):
            _, _, w, t, V, bs = self._unpack(x, d)
            bu = x[1:1 + S]
            r = self._predict(x, d) - d.y
            L = (d.g * r ** 2).sum() / wsum
            L += (lam_unit * (bu ** 2).sum() + self.lam_b * (bs ** 2).sum()
                  + self.lam_w * ((w ** 2).sum() + (t ** 2).sum()) + self.lam_v * (V ** 2).sum()) / wsum
            gr = 2 * d.g * r / wsum
            grad = np.zeros_like(x)
            grad[0] = gr.sum()
            grad[1:1 + S] = d.member.T @ gr + 2 * lam_unit * bu / wsum
            if Q:
                grad[1 + S:o] = d.smember.T @ gr + 2 * self.lam_b * bs / wsum
            if self.use_roles:
                grad[o:o + P] = d.role_pairs.T @ gr + 2 * self.lam_w * w / wsum
                grad[o + P:o + P + 2] = d.typefit.T @ gr + 2 * self.lam_w * t / wsum
            if self.use_latent:
                s = d.member @ V  # T x k
                gV = d.member.T @ (gr[:, None] * s) - (d.member.T @ gr)[:, None] * V
                grad[o + P + 2:] = (gV + 2 * self.lam_v * V / wsum).ravel()
            return L, grad

        res = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.x, self.d = res.x, d
        return self

    def predict(self, d):
        return self._predict(self.x, d)

    def pair_synergy(self, roster, a, b):
        """The model's synergy for one pair: (role part, type-fit part, style part), in wins per game."""
        mu, bb, w, t, V, _ = self._unpack(self.x, self.d)
        i, j = roster.index[a], roster.index[b]
        ri, rj = roster.roles[a], roster.roles[b]
        m = np.outer(ri, rj) + np.outer(rj, ri)
        role = (m[self.d.iu] * w).sum() if self.use_roles else 0.0
        tf = np.dot(roster.type_fit(a, b), t) if self.use_roles else 0.0
        style = float(V[i] @ V[j]) if self.use_latent else 0.0
        return role, tf, style

    def strength(self, roster, a):
        return self._unpack(self.x, self.d)[1][roster.index[a]]

    def top_role_terms(self, roster, a, b, n=3):
        mu, bb, w, t, V, _ = self._unpack(self.x, self.d)
        ri, rj = roster.roles[a], roster.roles[b]
        m = (np.outer(ri, rj) + np.outer(rj, ri))[self.d.iu] * w
        order = np.argsort(-np.abs(m))[:n]
        return [(f"{ROLES[self.d.iu[0][k]]} x {ROLES[self.d.iu[1][k]]}", m[k]) for k in order if m[k] != 0]


# ---------------------------------------------------------------------------------------------------- validation

def pair_stats(teams, min_games):
    """Species pairs on at least min_games games: (their teams' mean result, games, team positions)."""
    agg = defaultdict(lambda: [0.0, 0.0, []])
    for k, t in enumerate(teams):
        for i, j in itertools.combinations(t[5], 2):
            a = agg[(i, j)]
            a[0] += t[1] * t[2]
            a[1] += t[2]
            a[2].append(k)
    return {p: (a[0] / a[1], a[1], a[2]) for p, a in agg.items() if a[1] >= min_games}


def _score(data, configs, train_of, tests):
    """Fit every config on its data's training teams and predict each test group (team positions)."""
    out = {}
    for name, (key, model) in configs.items():
        roster, teams = data[key]
        m = model().fit(Design(roster, train_of(teams)))
        out[name] = [np.average(m.predict(d := Design(roster, [teams[k] for k in ks])), weights=d.g)
                     for ks in tests]
    return out


def held_out_pairs(data, configs, folds=8, min_games=500, budget=0.15, seed=0):
    """Hide a fold of well-played pairs (drop every team carrying any of them), fit, and predict how the teams
    carrying each hidden pair did. Each fold hides at most `budget` of all teams, so the fit still sees most of the
    data; pairs on more teams than that on their own are never hidden. `data` maps a name to (roster, teams), with
    the same teams in the same order. Returns, per config, (predicted, actual, games) over all hidden pairs."""
    teams = next(iter(data.values()))[1]
    pairs = pair_stats(teams, min_games)
    keys = [p for p in pairs if len(pairs[p][2]) <= budget * len(teams)]
    random.Random(seed).shuffle(keys)
    out = {name: ([], [], []) for name in configs}
    hidden = 0
    for f in range(folds):
        test, test_teams = [], set()
        for p in keys:
            more = test_teams | set(pairs[p][2])
            if len(more) <= budget * len(teams):
                test.append(p)
                test_teams = more
        keys = [p for p in keys if p not in test]
        if not test:
            break
        hidden += len(test)
        train = lambda ts: [t for k, t in enumerate(ts) if k not in test_teams]
        preds = _score(data, configs, train, [pairs[p][2] for p in test])
        for name in configs:
            out[name][0].extend(preds[name])
            out[name][1].extend(pairs[p][0] for p in test)
            out[name][2].extend(pairs[p][1] for p in test)
        print(f"  fold {f + 1}/{folds}: {len(test)} hidden pairs, trained on {len(teams) - len(test_teams)} teams",
              flush=True)
    return out, hidden


def later_events(data, configs, min_games=300):
    """Train on events before RECENT, predict the pairs' results at later events."""
    teams = next(iter(data.values()))[1]
    late = [k for k, t in enumerate(teams) if t[4] >= RECENT]
    pairs = pair_stats([teams[k] for k in late], min_games)
    tests = [[late[k] for k in ks] for _, _, ks in pairs.values()]
    preds = _score(data, configs, lambda ts: [t for t in ts if t[4] < RECENT], tests)
    act, games = [y for y, _, _ in pairs.values()], [g for _, g, _ in pairs.values()]
    return {name: (preds[name], act, games) for name in configs}, len(pairs)


def report(title, results, n):
    print(f"\n{title} ({n} pairs): correlation of predicted vs actual result of the teams carrying each pair")
    base = None
    for name, (pred, act, games) in results.items():
        pred, act, w = np.array(pred), np.array(act), np.array(games, float)
        r = np.corrcoef(pred, act)[0, 1]
        mx, my = np.average(pred, weights=w), np.average(act, weights=w)
        rw = np.average((pred - mx) * (act - my), weights=w) / np.sqrt(
            np.average((pred - mx) ** 2, weights=w) * np.average((act - my) ** 2, weights=w))
        base = base if base is not None else (pred, act)
        line = f"  {name:34s} r = {r:+.3f}   weighted {rw:+.3f}"
        if name != list(results)[0]:
            # does the model add to the individual-only baseline? correlation of its extra prediction with what
            # the baseline missed
            extra, missed = pred - base[0], act - base[0]
            line += f"   synergy part vs what the baseline missed: r = {np.corrcoef(extra, missed)[0, 1]:+.3f}"
        print(line)


CONFIGS = {  # name: (which roster, model); role weights shrunk more per set, where strengths absorb more
    "per species: strength only": ("species", lambda: Model(roles=False)),
    "per species: + role and type synergy": ("species", lambda: Model(lam_w=300)),
    "per set: strength only": ("set", lambda: Model(roles=False)),
    "per set: + role and type synergy": ("set", lambda: Model()),
}


# ---------------------------------------------------------------------------------------------------- suggest

def suggest(roster, teams, top=25, min_teams=150, max_played=150, boots=12):
    """Pairs ranked by how sure the model is of their synergy (synergy minus its bootstrap spread), among
    Pokemon on enough teams for their individual strength to be known and at least average on their own."""
    counts = Counter(i for idx, *_ in teams for i in idx)
    names = [n for n in roster.names if counts[roster.index[n]] >= min_teams]
    played = defaultdict(float)
    for idx, y, g, *_ in teams:
        for i, j in itertools.combinations(idx, 2):
            played[(i, j)] += g
    full = Model().fit(Design(roster, teams))
    rng = random.Random(1)
    boot_models = []
    for _ in range(boots):
        sample = [teams[rng.randrange(len(teams))] for _ in teams]
        boot_models.append(Model(seed=rng.randrange(1000)).fit(Design(roster, sample)))
    rows = []
    for a, b in itertools.combinations(names, 2):
        sa_, sb_ = roster.species_of[a], roster.species_of[b]
        if sa_ == sb_ or ("-Mega" in sa_ and "-Mega" in sb_):  # species clause; one Mega per team
            continue
        ra, rb = dict(zip(ROLES, roster.roles[a])), dict(zip(ROLES, roster.roles[b]))
        if any(ra[w] for w in WEATHER.values()) and any(rb[w] for w in WEATHER.values()):
            continue  # two weather setters undo or duplicate each other; the model has no term for that
        i, j = sorted((roster.index[a], roster.index[b]))
        role, tf, style = full.pair_synergy(roster, a, b)
        sa, sb = full.strength(roster, a), full.strength(roster, b)
        boot = [sum(m.pair_synergy(roster, a, b)) for m in boot_models]
        rows.append({"pair": f"{a} / {b}", "games": int(played.get((i, j), 0.0)),
                     "predicted": 100 * (sa + sb + role + tf + style), "synergy": 100 * (role + tf + style),
                     "synergy_sd": 100 * float(np.std(boot)), "roles": 100 * role, "types": 100 * tf,
                     "style": 100 * style, "strength_a": 100 * sa, "strength_b": 100 * sb,
                     "why": "; ".join(f"{k} {100 * v:+.2f}" for k, v in full.top_role_terms(roster, a, b))})
    OUT.mkdir(exist_ok=True)
    rows.sort(key=lambda r: -(r["synergy"] - r["synergy_sd"]))
    with open(OUT / "synergy_pairs.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    solid = [r for r in rows if min(r["strength_a"], r["strength_b"]) >= 0]
    for label, part in ((f"Under-played pairs (< {max_played} games)", [r for r in solid if r["games"] < max_played]),
                        ("Well-played pairs, for comparison", [r for r in solid if r["games"] >= max_played])):
        print(f"\n{label} with the surest synergy (wins above skill per game, %), "
              f"both Pokemon on {min_teams}+ teams and at least average alone:")
        print(f"  {'synergy':>7} {'+/-':>5} {'pred':>6} {'games':>5}  pair  |  top role terms")
        for r in part[:top]:
            print(f"  {r['synergy']:+7.2f} {r['synergy_sd']:5.2f} {r['predicted']:+6.2f} {r['games']:5d}  {r['pair']}"
                  f"  |  {r['why']}")
    w = full._unpack(full.x, full.d)[2]
    order = np.argsort(-np.abs(w))[:15]
    print("\nRole combinations the model learned matter most (+ helps, - hurts), per pair on a team:")
    for k in order:
        print(f"  {100 * w[k]:+.2f}  {ROLES[full.d.iu[0][k]]} x {ROLES[full.d.iu[1][k]]}")


def sets(roster, teams, boots=8, min_teams=40, top=40):
    """Every set's individual strength (wins above skill per game, %), with its bootstrap spread and how much of
    its species' play it gets: strong sets with a small share are the under-used ones."""
    full = Model(roles=False).fit(Design(roster, teams))
    rng = random.Random(2)
    boot = [Model(roles=False).fit(Design(roster, [teams[rng.randrange(len(teams))] for _ in teams]))
            for _ in range(boots)]
    count = Counter(i for t in teams for i in t[0])
    species = Counter(roster.species_of[roster.names[i]] for t in teams for i in t[0])
    rows = []
    for n in roster.names:
        if count[roster.index[n]] >= min_teams:
            st, sd = 100 * full.strength(roster, n), 100 * float(np.std([m.strength(roster, n) for m in boot]))
            rows.append({"set": n, "strength": st, "sd": sd, "teams": count[roster.index[n]],
                         "share": count[roster.index[n]] / species[roster.species_of[n]]})
    rows.sort(key=lambda r: -(r["strength"] - r["sd"]))
    OUT.mkdir(exist_ok=True)
    with open(OUT / "synergy_sets.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    print(f"\nSets by individual strength minus its spread (on {min_teams}+ teams; share = of its species' teams):")
    for r in rows[:top]:
        print(f"  {r['strength']:+6.2f} +/- {r['sd']:4.2f}  {r['teams']:5d} teams {100 * r['share']:4.0f}%  {r['set']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=("validate", "suggest", "sets"))
    p.add_argument("--per-species", action="store_true", help="suggest with one unit per species, not per set")
    args = p.parse_args(argv)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    data = {}
    for key in ("species", "set") if args.cmd == "validate" else ("species" if args.per_species else "set",):
        roster = Roster(db, per_set=key == "set")
        data[key] = (roster, load_teams(db, roster))
        print(f"{key}: {len(roster.names)} units, {len(data[key][1])} teams", flush=True)
    if args.cmd == "validate":
        res, n = held_out_pairs(data, CONFIGS)
        report("Held-out pairs (every team carrying them hidden while fitting)", res, n)
        res, n = later_events(data, CONFIGS)
        report(f"Later events (fit before {RECENT}, predict after)", res, n)
    elif args.cmd == "sets":
        sets(*next(iter(data.values())))
    else:
        suggest(*next(iter(data.values())))


if __name__ == "__main__":
    main()
