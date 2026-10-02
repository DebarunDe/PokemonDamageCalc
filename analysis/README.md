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
  tournaments with 8+ players, with every published team and its win-loss record. Raw downloads are
  cached under `analysis/data/raw`.
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
  - **proven**: Pokémon under 10% ladder usage, ranked by tournament performance, each with the
    model's top reasons.
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

## Monthly refresh

```bash
python analysis/refresh.py                 # ingest new data, rebuild features, model, report page (~1 h from scratch)
python analysis/refresh.py --skip-ingest   # rebuild from the data already downloaded
```

This writes `analysis/output/report.json` and `analysis/output/report.html`, which is
`analysis/report_template.html` with the data embedded. The page ranks underused Pokémon by
skill-adjusted edge, lists popular Pokémon that are losing ground, lists untested leads from the
usage model, and shows the backtests.

A new regulation needs three things first: its Showdown commit and mod in `js/package.json`
(`config.regulations`, then `npm run build`), its Showdown format id in `ingest.py`
(`SHOWDOWN_FORMATS`), and its backtest snapshots in `model.py`.
