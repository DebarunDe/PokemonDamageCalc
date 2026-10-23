"""Synergy model: which Pokemon work well together, learned from tournament results, so it can suggest cores
that haven't been played much (or at all).

Each Pokemon gets a role profile from its real sets and stats: support tools (Fake Out, redirection, Intimidate,
screens, ...), speed control and speed tier, weather and terrain (setter vs user), offense (setup, priority, spread,
physical / special) and bulk. A team's result (wins per game above its player's skill, as in signals.py) is
modelled as

    individual strength of each member
    + for every pair on the team: role x role synergy (e.g. Fake Out x setup, Trick Room x slow, sun x Chlorophyll)
      + type fit (weaknesses they share, weaknesses one covers for the other)
      + a learned "style" match (latent factors, as in recommender systems)

fitted by weighted least squares (weights = games) with ridge penalties. Because synergy comes from roles and
styles rather than a table of observed pairs, the model can score pairs it has rarely or never seen.

Usage:
    python analysis/synergy.py validate      # does it predict held-out pairs and later events? (~5 min)
    python analysis/synergy.py suggest       # under-played cores it rates highly, with what drives them

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
         "helping_hand", "sun", "rain", "sand", "snow", "sun_user", "rain_user", "sand_user", "snow_user", "terrain",
         "terrain_user", "setup", "priority", "spread", "physical", "special", "fast", "slow", "bulky", "protect",
         "mega"]
MOVE_ROLES = {
    "fake_out": {"Fake Out"}, "redirection": {"Follow Me", "Rage Powder"}, "tailwind": {"Tailwind"},
    "trick_room": {"Trick Room"}, "screens": {"Reflect", "Light Screen", "Aurora Veil"},
    "sleep": {"Spore", "Sleep Powder", "Hypnosis", "Yawn"},
    "pivot": {"Parting Shot", "U-turn", "Volt Switch", "Flip Turn"},
    "wide_guard": {"Wide Guard", "Quick Guard"}, "helping_hand": {"Helping Hand"},
    "sun_user": {"Solar Beam", "Solar Blade"}, "rain_user": {"Electro Shot", "Hurricane", "Thunder"},
    "snow_user": {"Blizzard", "Aurora Veil"},
    "terrain_user": {"Grassy Glide", "Expanding Force", "Steel Roller", "Rising Voltage"},
    "setup": {"Swords Dance", "Nasty Plot", "Dragon Dance", "Calm Mind", "Bulk Up", "Quiver Dance", "Shell Smash",
              "Belly Drum", "Coil", "Victory Dance", "Tidy Up", "Clangorous Soul", "Growth", "Agility", "Iron Defense"},
    "protect": {"Protect", "Detect", "Spiky Shield", "King's Shield", "Baneful Bunker", "Silk Trap", "Burning Bulwark"},
}
WEATHER = {"Drought": "sun", "Drizzle": "rain", "Sand Stream": "sand", "Snow Warning": "snow"}
WEATHER_USERS = {"Chlorophyll": "sun_user", "Swift Swim": "rain_user", "Sand Rush": "sand_user",
                 "Sand Force": "sand_user", "Slush Rush": "snow_user"}
TERRAIN = {"Grassy Surge", "Psychic Surge", "Electric Surge", "Misty Surge"}


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
    """Role profiles of every Pokemon on M-C tournament teams, from their real sets and the calc's data."""

    def __init__(self, db):
        from champcalc import CalcError, Calculator
        calc = Calculator()
        rows = db.execute("SELECT m.pokemon, m.ability, m.moves FROM team_members m JOIN tournaments t "
                          "ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)).fetchall()
        sets = defaultdict(list)
        for name, ability, moves in rows:
            sets[name].append((ability or "", json.loads(moves) if moves else []))
        self.names, self.roles, self.weak = [], {}, {}
        for name, entries in sets.items():
            try:
                sp = calc.species(name)
            except CalcError:  # names the calc doesn't know, e.g. entered in another language
                continue
            moves = Counter(m for _, ms in entries for m in ms)
            common = {m for m, k in moves.items() if k >= 0.25 * len(entries)}
            pre = Counter(a for a, _ in entries).most_common(1)[0][0]
            ability = pre if pre in sp["abilities"] or not sp["abilities"] else sp["abilities"][0]
            base = sp["base_stats"]
            r = {role: float(bool(common & MOVE_ROLES.get(role, set()))) for role in ROLES}
            r["intimidate"] = float(ability == "Intimidate" or pre == "Intimidate")
            for ab, w in WEATHER.items():
                r[w] = max(r[w], float(ability == ab))
            if name in ("Charizard-Mega-Y",):
                r["sun"] = 1.0
            if ability in WEATHER_USERS:
                r[WEATHER_USERS[ability]] = 1.0
            r["terrain"] = float(ability in TERRAIN)
            damaging = []
            for m in common:
                try:
                    info = calc.move(m)
                except Exception:
                    continue
                if info["category"] != "Status":
                    damaging.append(info)
            r["priority"] = float(any(i["priority"] > 0 and i["name"] != "Fake Out" for i in damaging))
            r["spread"] = float(any(i["spread"] for i in damaging))
            phys = sum(i["category"] == "Physical" for i in damaging)
            spec = sum(i["category"] == "Special" for i in damaging)
            r["physical"], r["special"] = float(phys > spec), float(spec > phys)
            r["fast"], r["slow"] = float(base["spe"] >= 100), float(base["spe"] <= 60)
            r["bulky"] = float((base["hp"] * base["def"] * base["spd"]) ** (1 / 3) >= 95)
            r["mega"] = float("-Mega" in name)
            self.names.append(name)
            self.roles[name] = np.array([r[k] for k in ROLES])
            self.weak[name] = weakness_profile(sp["types"], ability)
        self.index = {n: i for i, n in enumerate(self.names)}

    def type_fit(self, a, b):
        """(weaknesses they share, weaknesses one resists for the other) over the 18 attacking types."""
        wa, wb = self.weak[a], self.weak[b]
        shared = sum(wa[t] > 1 and wb[t] > 1 for t in TYPES)
        covered = sum((wa[t] > 1 and wb[t] < 1) or (wb[t] > 1 and wa[t] < 1) for t in TYPES)
        return shared, covered


# ---------------------------------------------------------------------------------------------------- data

def load_teams(db, roster):
    """Teams as (member indices, wins above skill per game, games, tournament id, date)."""
    skill = player_skill(db).set_index(["tournament", "player"])["skill"].to_dict()
    members, record = defaultdict(set), {}
    for t, p, mon, w, l, date in db.execute(
            "SELECT m.tournament, m.player, m.pokemon, m.wins, m.losses, t.date FROM team_members m "
            "JOIN tournaments t ON t.id = m.tournament WHERE t.regulation = ?", (REGULATION,)):
        if mon in roster.index:
            members[(t, p)].add(roster.index[mon])
        record[(t, p)] = (w or 0, l or 0, date)
    teams = []
    for key, idx in members.items():
        w, l, date = record[key]
        if w + l and key in skill and len(idx) >= 4:
            teams.append((sorted(idx), (w - skill[key] * (w + l)) / (w + l), w + l, key[0], date))
    return teams


class Design:
    """Per-team sums of the model's features, so fitting is fast."""

    def __init__(self, roster, teams, latent=True):
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
        self.members_idx = [t[0] for t in teams]
        self.R = R
        self.latent = latent


class Model:
    def __init__(self, k=4, lam_b=30.0, lam_w=300.0, lam_v=300.0, roles=True, latent=True, seed=0):
        self.k, self.lam_b, self.lam_w, self.lam_v = k, lam_b, lam_w, lam_v
        self.use_roles, self.use_latent, self.seed = roles, latent and k > 0, seed

    def _unpack(self, x, d):
        S, P = d.S, len(d.iu[0])
        mu, b = x[0], x[1:1 + S]
        w = x[1 + S:1 + S + P]
        t = x[1 + S + P:3 + S + P]
        V = x[3 + S + P:].reshape(S, self.k) if self.use_latent else np.zeros((S, 0))
        return mu, b, w, t, V

    def _predict(self, x, d):
        mu, b, w, t, V = self._unpack(x, d)
        pred = mu + d.member @ b
        if self.use_roles:
            pred = pred + d.role_pairs @ w + d.typefit @ t
        if self.use_latent:
            s = d.member @ V
            pred = pred + 0.5 * ((s ** 2).sum(1) - d.member @ (V ** 2).sum(1))
        return pred

    def fit(self, d):
        S, P = d.S, len(d.iu[0])
        n = 3 + S + P + (S * self.k if self.use_latent else 0)
        x0 = np.zeros(n)
        if self.use_latent:
            x0[3 + S + P:] = np.random.default_rng(self.seed).normal(0, 0.01, S * self.k)
        x0[0] = np.average(d.y, weights=d.g)
        wsum = d.g.sum()

        def loss(x):
            mu, b, w, t, V = self._unpack(x, d)
            pred = self._predict(x, d)
            r = pred - d.y
            L = (d.g * r ** 2).sum() / wsum
            L += (self.lam_b * (b ** 2).sum() + self.lam_w * ((w ** 2).sum() + (t ** 2).sum())
                  + self.lam_v * (V ** 2).sum()) / wsum
            gr = 2 * d.g * r / wsum
            grad = np.zeros_like(x)
            grad[0] = gr.sum()
            grad[1:1 + S] = d.member.T @ gr + 2 * self.lam_b * b / wsum
            if self.use_roles:
                grad[1 + S:1 + S + P] = d.role_pairs.T @ gr + 2 * self.lam_w * w / wsum
                grad[1 + S + P:3 + S + P] = d.typefit.T @ gr + 2 * self.lam_w * t / wsum
            if self.use_latent:
                s = d.member @ V  # T x k
                gV = d.member.T @ (gr[:, None] * s) - (d.member.T @ gr)[:, None] * V
                grad[3 + S + P:] = (gV + 2 * self.lam_v * V / wsum).ravel()
            return L, grad

        res = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.x, self.d = res.x, d
        return self

    def predict(self, d):
        return self._predict(self.x, d)

    def pair_synergy(self, roster, a, b):
        """The model's synergy for one pair: (role part, type-fit part, style part), in wins per game."""
        mu, bb, w, t, V = self._unpack(self.x, self.d)
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
        mu, bb, w, t, V = self._unpack(self.x, self.d)
        ri, rj = roster.roles[a], roster.roles[b]
        m = (np.outer(ri, rj) + np.outer(rj, ri))[self.d.iu] * w
        order = np.argsort(-np.abs(m))[:n]
        return [(f"{ROLES[self.d.iu[0][k]]} x {ROLES[self.d.iu[1][k]]}", m[k]) for k in order if m[k] != 0]


# ---------------------------------------------------------------------------------------------------- validation

def pair_stats(teams, roster, min_games):
    agg = defaultdict(lambda: [0.0, 0.0, []])
    for k, (idx, y, g, *_rest) in enumerate(teams):
        for i, j in itertools.combinations(idx, 2):
            a = agg[(i, j)]
            a[0] += y * g
            a[1] += g
            a[2].append(k)
    return {p: (a[0] / a[1], a[1], a[2]) for p, a in agg.items() if a[1] >= min_games}


def held_out_pairs(roster, teams, configs, folds=5, min_games=500, seed=0):
    """Hide a fold of well-played pairs (drop every team carrying any of them), fit, and predict how the teams
    carrying each hidden pair did. Returns, per config, (predicted, actual, games) over all hidden pairs."""
    pairs = pair_stats(teams, roster, min_games)
    keys = list(pairs)
    random.Random(seed).shuffle(keys)
    out = {name: ([], [], []) for name in configs}
    for f in range(folds):
        test = set(keys[f::folds])
        test_teams = set(k for p in test for k in pairs[p][2])
        train = [t for k, t in enumerate(teams) if k not in test_teams]
        d_train = Design(roster, train)
        d_test = {p: Design(roster, [teams[k] for k in pairs[p][2]]) for p in test}
        for name, model in configs.items():
            m = model().fit(d_train)
            for p in test:
                out[name][0].append(np.average(m.predict(d_test[p]), weights=d_test[p].g))
                out[name][1].append(pairs[p][0])
                out[name][2].append(pairs[p][1])
        print(f"  fold {f + 1}/{folds}: {len(test)} hidden pairs, trained on {len(train)} teams", flush=True)
    return out, len(keys)


def later_events(roster, teams, configs, min_games=300):
    """Train on events before RECENT, predict the pairs' results at later events."""
    early = [t for t in teams if t[4] < RECENT]
    late = [t for t in teams if t[4] >= RECENT]
    pairs = pair_stats(late, roster, min_games)
    out = {}
    d_early = Design(roster, early)
    d_test = {p: Design(roster, [late[k] for k in ks]) for p, (_, _, ks) in pairs.items()}
    for name, model in configs.items():
        m = model().fit(d_early)
        pred, act, games = [], [], []
        for p, (y, g, ks) in pairs.items():
            pred.append(np.average(m.predict(d_test[p]), weights=d_test[p].g))
            act.append(y)
            games.append(g)
        out[name] = (pred, act, games)
    return out, len(pairs)


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


CONFIGS = {
    "individual strength only": lambda: Model(roles=False, latent=False),
    "+ role and type synergy": lambda: Model(roles=True, latent=False),
    "+ learned styles": lambda: Model(roles=False, latent=True),
    "+ roles, types and styles": lambda: Model(roles=True, latent=True),
}


# ---------------------------------------------------------------------------------------------------- suggest

def suggest(roster, teams, top=25, min_teams=40, max_played=150, boots=12):
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
        if "-Mega" in a and "-Mega" in b:
            continue
        i, j = sorted((roster.index[a], roster.index[b]))
        games = played.get((i, j), 0.0)
        role, tf, style = full.pair_synergy(roster, a, b)
        strength = full.strength(roster, a) + full.strength(roster, b)
        boot = [sum(m.pair_synergy(roster, a, b)) + m.strength(roster, a) + m.strength(roster, b) for m in boot_models]
        rows.append({"pair": f"{a} / {b}", "games": int(games), "predicted": 100 * (strength + role + tf + style),
                     "synergy": 100 * (role + tf + style), "roles": 100 * role, "types": 100 * tf, "style": 100 * style,
                     "strength": 100 * strength, "sd": 100 * float(np.std(boot)),
                     "why": "; ".join(f"{k} {100 * v:+.2f}" for k, v in full.top_role_terms(roster, a, b))})
    OUT.mkdir(exist_ok=True)
    rows.sort(key=lambda r: -r["predicted"])
    with open(OUT / "synergy_pairs.csv", "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    fresh = [r for r in rows if r["games"] < max_played and r["predicted"] - r["sd"] > 0]
    print(f"\nUnder-played pairs (< {max_played} games) the model rates highly "
          "(predicted wins above skill per game, %):")
    print(f"  {'pred':>6} {'+/-':>5} {'synergy':>7} {'games':>5}  pair  |  top role terms")
    for r in fresh[:top]:
        print(f"  {r['predicted']:+6.2f} {r['sd']:5.2f} {r['synergy']:+7.2f} {r['games']:5d}  {r['pair']}"
              f"  |  {r['why']}")
    print("\nStrongest synergies overall (any sample size):")
    for r in sorted(rows, key=lambda r: -r["synergy"])[:12]:
        print(f"  synergy {r['synergy']:+5.2f} (roles {r['roles']:+.2f}, types {r['types']:+.2f}, "
              f"style {r['style']:+.2f}), games {r['games']:5d}  {r['pair']}")
    w = full._unpack(full.x, full.d)[2]
    order = np.argsort(-np.abs(w))[:15]
    print("\nRole combinations the model learned matter most (+ helps, - hurts), per pair on a team:")
    for k in order:
        print(f"  {100 * w[k]:+.2f}  {ROLES[full.d.iu[0][k]]} x {ROLES[full.d.iu[1][k]]}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=("validate", "suggest"))
    args = p.parse_args(argv)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    roster = Roster(db)
    teams = load_teams(db, roster)
    print(f"{len(roster.names)} Pokemon, {len(teams)} teams, {sum(t[2] for t in teams)} games", flush=True)
    if args.cmd == "validate":
        res, n = held_out_pairs(roster, teams, CONFIGS)
        report("Held-out pairs (every team carrying them hidden while fitting)", res, n)
        res, n = later_events(roster, teams, CONFIGS)
        report(f"Later events (fit before {RECENT}, predict after)", res, n)
    else:
        suggest(roster, teams)


if __name__ == "__main__":
    main()
