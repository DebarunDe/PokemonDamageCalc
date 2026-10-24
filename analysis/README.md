# OHKO analysis

`ohko.py` answers: **which Pokémon one-hit KO the most of the field?**

```bash
python analysis/ohko.py                    # all legal Pokémon, singles (~50 min on 4 cores)
python analysis/ohko.py --roster megas     # Megas only (~1 min)
python analysis/ohko.py --doubles          # spread moves take 0.75x
python analysis/ohko.py --exclude recharge # also drop Hyper Beam, Giga Impact, ...
python analysis/ohko.py --items none       # non-Megas hold nothing (Megas keep their stones)
python analysis/ohko.py --regulation M-B   # an earlier regulation (default: the current one, M-C)
```

The results go to `analysis/output/` (not committed):
- `ohko_<regulation>_<roster>_<mode>_matchups.csv.gz`: one row per attacker, defender and baseline pair.
  Each row has the attacker's ability and item, the defender's worst-case ability, the best move,
  the best reliable move and the best reliable move without a recharge turn, damage percentages,
  OHKO flags, weather, terrain and notes.
- `ohko_<regulation>_<roster>_<mode>_ranking.csv`: per Pokémon, how many others it OHKOs at each baseline pair,
  the first-strike count, its Speed (base and with 32 SP), and with which ability.

## Method

**Roster:** `Calculator(regulation).legal_species`. That's every Pokémon and Mega with a tier in
Showdown's data for that regulation, plus non-battle-only formes of legal species, minus duplicates
that play identically. M-A has 278 entries, M-B 314 and M-C 344.

**Baselines:** 3 attacker × 3 defender.

| Attacker | | Defender | |
|---|---|---|---|
| `max+` | 32 SP in the attacking stat, boosting nature | `none` | no investment |
| `max` | 32 SP in the attacking stat, neutral nature | `hp` | 32 HP |
| `none` | no investment | `bulk` | 32 HP + 32 in the defensive stat the move hits |

Investment follows the stat a move really uses: Body Press attacks with Defense, Psyshock hits
Defense, and Foul Play uses the target's Attack.

**Items:** Megas hold their Mega Stone. Every other Pokémon attacks with Life Orb, the strongest
damage item in Champions, which has no Choice Band or Specs, or with nothing when run with
`--items none`. Defenders hold no item. A Focus Sash
would stop any OHKO from full HP.

**Speed and first-strike OHKOs:** attackers at `max+` and `max` also put 32 SP in Speed (64 of the
66). Defenders at `hp` put 32 in Speed, and at `bulk` the 2 SP left over. A **first-strike OHKO** is
a reliable OHKO landed before the target can move: either the attacker is strictly faster (speed
ties don't count), or the move has priority. Swift Swim, Chlorophyll, Sand Rush, Slush Rush and
Surge Surfer double Speed under their weather or terrain. Gale Wings gives Flying moves priority,
and Grassy Glide has priority in Grassy Terrain. Trick Room, Tailwind and Choice Scarf are not
modelled.

**Abilities:** every attacker/defender ability pairing is calculated. A KO counts only if it works
against every ability the defender could have. Each attacker is then ranked with the single ability
that OHKOs the most.

**Matchup rules**
- **Weather and terrain** from either side's ability apply. When both set weather, the slower
  Pokémon's wins, because abilities activate in speed order with no Speed investment.
- **Intimidate** drops a physical attacker's Attack by 1. Inner Focus, Scrappy, Clear Body and
  similar abilities block it, and Contrary, Defiant and Guard Dog turn it into +1.
- **Trace** copies the opponent's ability.
- **Sturdy** stops a single-hit OHKO from full HP, but multi-hit moves and Parental Bond get past it.
  **Disguise** stops any OHKO. Mold Breaker ignores both. The calc does not model either ability, so
  the script applies them.
- **Charge moves:** Solar Beam and Solar Blade count only in sun or with Mega Sol, and Electro Shot
  only in rain. Steel Roller counts only with terrain.
- **Left out:** OHKO moves, fixed or reactive damage, self-KO moves, two-turn moves and moves that
  fail without setup. See `EXCLUDED_MOVES`.
- **Reliable** means at least 90% accuracy after weather, No Guard, Hustle and Compound Eyes.
  Population Bomb and Triple Axel roll accuracy for every hit, so they need all their hits to land.
- **Guaranteed OHKO** means the lowest damage roll KOs. **Possible OHKO** means the highest roll does.

**Not modelled:** crits, speed order and who moves first, Sucker Punch or Payback conditions
(Payback assumes the slower user moves second), and items other than those above. Multi-hit
moves with 2–5 hits assume 3 hits, or 5 with Skill Link.

# Meta modelling (finding undervalued Pokémon)

```bash
python analysis/ingest.py          # download Showdown ladder stats + Limitless tournaments (first run ~1 h)
python analysis/signals.py         # usage vs tournament win rate, top-cut conversion, rating lift; backtests
python analysis/meta_matchups.py   # meta-weighted doubles matchup features for every legal Pokémon
```

All three take `--regulation` (`M-A`, `M-B`, `M-C`; default the current one) where it applies.

- **`ingest.py`** fills `analysis/data/meta.sqlite` (not committed) with Showdown monthly stats for each
  Champions VGC regulation (best-of-1 and best-of-3 ladders, all ratings and 1760+): usage, moves,
  items, abilities, Stat Point spreads, teammates and checks/counters. It also stores Limitless
  tournaments with 8+ players, with every published team and its win-loss record. Official events
  (Regionals, Internationals, Worlds) come from RK9 Labs via `rk9.py`: every Masters team list, final
  standing and match record, found automatically and assigned to the regulation their teams are legal
  in (`python analysis/ingest.py rk9`). Raw downloads are cached under `analysis/data/raw`.
- **`signals.py`** scores each Pokémon on tournament win rate (shrunk toward the field average), top-cut
  conversion (actual vs expected top-cut appearances, shrunk toward 1) and ladder lift (1760+ usage vs
  all-rating usage). The undervalued score is performance minus what usage alone predicts. It also
  backtests whether one regulation's scores predict usage gains in the next.
- **`meta_matchups.py`** takes the meta (every Pokémon with 1%+ usage at 1760+, weighted by usage) with
  each Pokémon's most common set (ability, item, nature + SP, moves). It scores every legal Pokémon on
  its own common set, or a synthesized one without ladder data, in doubles. Expected OHKO/2HKO shares
  are computed both ways, attacking and defending, with the common moves and with any legal move.
  It also reports Speed against the meta (with no speed control, under Tailwind and under Trick Room)
  and access to utility (Fake Out, Tailwind, Trick Room, redirection, Intimidate, weather, ...).
- **`model.py`** fits what the meta rewards: a ridge model predicts log ladder usage at 1760+ from the
  matchup features plus is-Mega, base stat total and counter pressure (how often the meta beats it in
  ladder games). It's cross-validated, so each Pokémon is scored by a model that never saw it. It
  writes two lists:
  - **proven**: Pokémon under 10% ladder usage, ranked by tournament win rate shrunk toward the field
    average by sample size (empirical Bayes), each with its skill-adjusted edge and the model's top reasons.
  - **potential**: Pokémon the model rates highly but with too little tournament data to judge. This
    is a speculative watchlist.

  It also backtests each score: M-A (May) → M-B (July) and M-B (August) → M-C (September).
  ```bash
  python analysis/meta_matchups.py --regulation M-A --month 2026-05   # features for each snapshot
  python analysis/meta_matchups.py --regulation M-B --month 2026-08
  python analysis/meta_matchups.py --regulation M-C --month 2026-09
  python analysis/model.py
  ```
  **Findings so far:** the model explains about 43% of usage variation (cross-validated R²), but its
  usage gap does not predict future usage. Tournament performance does, with Spearman +0.13 and +0.17
  against usage change beyond mean reversion. The features miss mechanics like Unburden, Armor Tail
  and Last Respects, which is why the ranking leans on performance and uses the model to explain it.

## Longer backtest (month by month, Champions and Scarlet/Violet)

```bash
python analysis/ingest.py history     # Scarlet/Violet VGC 2022-12 to 2026-05: Showdown usage + Limitless (~1.5 h)
python analysis/backtest.py           # writes analysis/output/backtest_monthly.csv
python analysis/backtest.py --series champions
```

For every month with 8+ tournaments in its main format, each Pokémon with 50+ tournament games gets
three scores: skill-adjusted edge (player skill from tournaments up to that month only), raw win rate,
and win rate shrunk the same way as edge. Each score is compared (Spearman) with the Pokémon's change in
1760+ usage on the matching Showdown ladder the next month, beyond mean reversion. Limitless formats map to
Showdown ladders in `SERIES`; a transition into a different tournament format counts as a new regulation.

**Findings (46 transitions: 5 Champions, 41 Scarlet/Violet):**

| | Edge | Win rate | Shrunk win rate |
|---|---|---|---|
| All transitions: mean Spearman (positive / total, t) | +0.064 (32/46, t = 2.9) | +0.090 (35/46, t = 4.4) | +0.085 (34/46, t = 4.0) |
| Within a regulation (32) | +0.091 | +0.108 | +0.108 |
| Into a new regulation (14) | +0.000 | +0.051 | +0.031 |

- Tournament results lead the ladder by about a month, and the signal is small but consistent across
  three and a half years and two games.
- Adjusting for player skill does not help: plain win rate predicts as well or better. Strong players'
  choices seem to be part of the signal the ladder follows, not noise to remove.
- Results do not carry over into a new regulation (Scarlet/Violet: about zero over 12 changes), so read
  each regulation's first month on its own data.
- The 20 Pokémon with the highest edge beat mean reversion 59% of the time (66% in Champions).
- So the report ranks by shrunk win rate and shows edge alongside it. A full run (`--series all`) also
  writes `analysis/backtest_summary.json`, which is committed so the monthly refresh can show the track
  record without the Scarlet/Violet download.

## Monthly refresh

```bash
python analysis/refresh.py                 # ingest new data, rebuild features, model, report page (~1 h from scratch)
python analysis/refresh.py --skip-ingest   # rebuild from the data already downloaded
```

This writes `analysis/output/report.json` and `analysis/output/report.html`, which is
`analysis/report_template.html` with the data embedded. The page ranks underused Pokémon by
shrunk tournament win rate (with skill-adjusted edge beside it), lists popular Pokémon that are losing ground, lists untested leads from the
usage model, and shows the backtests.

A new regulation needs three things first: its Showdown commit and mod in `js/package.json`
(`config.regulations`, then `npm run build`), its Showdown format id in `ingest.py`
(`SHOWDOWN_FORMATS`), and its backtest snapshots in `model.py`.

# Doubles battle simulator

`doubles_sim.py` plays doubles battles turn by turn, with @smogon/calc damage, random rolls, accuracy and
critical hits, to test how Pokémon do *together* rather than one-on-one.

```bash
python analysis/doubles_sim.py battle "Metagross-Mega,Arcanine-Hisui" "Incineroar,Gholdengo"   # one logged battle
python analysis/doubles_sim.py core "Rillaboom,Metagross-Mega,Arcanine-Hisui" --teams 400      # win rate vs real teams
python analysis/core_search.py                                                                 # best 2-, 3-, 4-cores
```

- **Sets:** each Pokémon's most common complete tournament set (item, ability, nature, moves) from
  `data/meta.sqlite`, with its most common ladder spread for that nature.
- **Opponents:** real teams, every official Regional top 32 plus a sample of recent online top 8s. Against a core of
  *n*, the opponent brings *n* of its six at random. Each battle is played twice with fixed seeds, so every core
  meets the same opponents and the same dice.
- **AI:** the same heuristic for both sides. It takes KOs and focus fires, Protects when about to be KO'd, sets up
  when it's safe (counting its partner's Fake Out or redirection), Fake Outs the biggest threat on its first turn,
  and sets Tailwind or Trick Room when slower. It doesn't predict, switch (other than Parting Shot) or Encore.
- **Mechanics:** speed order with priority, Tailwind and Trick Room; Fake Out and its immunities (Inner Focus,
  Ghost, Armor Tail, Psychic Terrain); Protect (a repeat works 1 time in 3), Wide Guard and Quick Guard; Follow Me,
  Rage Powder, Helping Hand and screens; Intimidate, weather and terrain on entry; a Mega's pre-Mega ability on entry
  (Metagross's Clear Body blocks Intimidate); setup moves, sleep, burn and paralysis; Focus Sash, Sitrus Berry,
  Leftovers, Life Orb, recoil and drain; Sucker Punch, Rage Fist, Last Respects, Electro Shot and Steel Roller;
  spread moves (Earthquake also hits the ally) and replacements from the back.

`core_search.py` screens every pair of the most used Pokémon (at most one Mega per core) on a few teams, plays the
best again on many more, then extends the best pairs to trios and the best trios to quads the same way. The best
pair of a trio or quad leads. Results go to `output/cores_<size>.csv`.

**How far to trust it.** `sim_validation.py` checks the simulator against reality: for the 70 most played real pairs
it plays the real teams carrying each pair (4v4, both sides picking their four at team preview) and correlates the
simulated win rate with how those teams actually did, beyond their players' skill. As of October 2026 the
correlation is about zero (−0.10 to +0.05 across versions, including the October AI that values setup moves by
the damage they add and sets up behind a partner's Fake Out or redirection), although the real numbers are
reliable (split by tournament, the two halves agree at r ≈ 0.57; reliability ≈ 0.73). The simulator overrates high-powered attackers
(Mega Garchomp Z, Mega Salamence, Gholdengo, Volcarona) and underrates bulky, support and late-game Pokémon
(Grimmsnarl, Kingambit, Archaludon, Mega Charizard Y). Use it to compare sets within a core or to check specific
interactions, not to rank cores; rank those from tournament results. Seed battles with `doubles_sim.seed_of(...)`
rather than `hash()` of strings, which Python randomises per process, so runs repeat exactly.

# Core ranking from tournament results

`core_ranking.py` ranks 2-, 3- and 4-Pokémon cores by how the teams carrying them did in M-C tournaments: wins per
game above what the players' skill predicts, shrunk toward the average by sample size (the empirical-Bayes prior
from `backtest.py`). Next to each score it shows how many players used the core and the biggest one's share of its
games, and the score at official Regionals vs online events and before vs after the current meta settled.

```bash
python analysis/core_ranking.py                      # pairs, trios and fours (~30 s)
python analysis/core_ranking.py --size 4 --top 40
```

Results go to `output/core_ranking_<size>.csv`.

# Synergy model (ideas for new cores)

Core ranking only scores cores people already play. `synergy.py` learns *why* Pokémon work together so it can score
pairs that have rarely or never been played. Each set a Pokémon is commonly run with is its own unit (Choice Scarf
vs Bulk Up Annihilape, Dragon Dance vs Low Kick Mega Tyranitar; sets on fewer than 30 teams join the nearest common
one), with its strength shrunk toward its species'. Each unit gets a role profile from its moves, item, ability and
base stats (Fake Out, redirection, Intimidate, Tailwind, Trick Room, screens, sleep, pivoting, Wide Guard, Helping
Hand, weather setters and users, Psychic / Grassy / Electric terrain setters and users, setup, priority, spread
moves, physical / special, Choice item, fast / slow, bulky, Protect, Mega). A team's result (wins per game above
its player's skill) is modelled as each member's individual strength plus, for every pair on the team, a learned
weight for each role × role combination and a type-fit term (weaknesses shared, weaknesses covered), fitted by
weighted ridge regression.

```bash
python analysis/synergy.py validate    # per species vs per set, with and without synergy (~25 min)
python analysis/synergy.py sets        # every set's individual strength and share of its species (~3 min)
python analysis/synergy.py suggest     # pairs ranked by how sure the model is of their synergy (~5 min)
```

**Validation (October 2026).** Correlation between predicted and actual results of the teams carrying each pair,
weighted by games:

| test | per species | per species + synergy | per set | per set + synergy |
| --- | --- | --- | --- | --- |
| held-out pairs (69 pairs, 500+ games; their teams hidden while fitting) | +0.24 | +0.30 | +0.39 | **+0.40** |
| later events (fit before 26 Sep, predict 512 pairs after) | +0.16 | +0.28 | +0.49 | **+0.52** |

Scoring sets instead of species is the big gain. Once sets have their own strength, role synergy adds little, and
only with its weights shrunk hard (the default); a lot of what looked like synergy per species was set choice.
Splitting terrain by type (so Electric Terrain doesn't count as helping Expanding Force) and adding a Choice role
made the per-species synergy validate worse (+0.45 to +0.30 on held-out pairs), a sign of how fragile the role
terms are. Learned latent "styles" (`Model(latent=True)`) did not generalise to unseen pairs, so they are off. The
shrinkage was tuned on the later-events split, so the held-out-pairs test is the independent check. `suggest` skips
same-species pairs, two Megas, and two weather setters (the model has no term for weather clashes).

Treat `sets` as the most reliable output and `suggest` as a source of ideas to check with damage calcs. Results go
to `output/synergy_sets.csv` and `output/synergy_pairs.csv`.
