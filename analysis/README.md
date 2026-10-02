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
