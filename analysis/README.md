# Mega vs. Mega OHKO analysis

`mega_ohko.py` answers: **which Mega one-hit KOs the most other Megas?**

```bash
python analysis/mega_ohko.py                      # singles
python analysis/mega_ohko.py --doubles            # spread moves take 0.75x
python analysis/mega_ohko.py --exclude recharge   # without Hyper Beam, Giga Impact, Blast Burn, ...
```

It takes about a minute on 4 cores. The results go to `analysis/output/` (not committed):
- `mega_ohko_<mode>_matchups.csv`: one row per attacker, defender and baseline, with the best move,
  the best move with at least 90% accuracy, damage percentages, OHKO flags, weather, terrain and notes.
- `mega_ohko_<mode>_ranking.csv`: per Mega, how many of the other 81 it OHKOs at each baseline.

## Method

All 82 Megas attack all 81 others with every damaging move they can learn in Champions.
That's 279,126 attacker-move-defender combinations, each run at 9 baselines.

| Attacker | | Defender | |
|---|---|---|---|
| `max+` | 32 SP in the attacking stat, boosting nature | `none` | no investment |
| `max` | 32 SP in the attacking stat, neutral nature | `hp` | 32 HP |
| `none` | no investment | `bulk` | 32 HP + 32 in the defensive stat the move hits |

Investment always follows the stat a move really uses: Body Press attacks with Defense, Psyshock
hits Defense, and Foul Play uses the target's Attack.

**Matchup rules**
- Every Mega holds its Mega Stone, which matters for Acrobatics and Poltergeist.
- **Weather and terrain** from either Mega's ability apply. When both set weather, the slower Mega's
  weather wins, because abilities activate in speed order with no Speed investment. Those rows are
  marked "contested".
- **Intimidate** drops a physical attacker's Attack by 1. Inner Focus and Scrappy block it, and
  Contrary and Defiant turn it into +1.
- **Trace** copies the opponent's ability.
- Solar Beam and Solar Blade count only in sun or with Mega Sol. Steel Roller counts only with terrain.
- **Left out:** OHKO moves, fixed or reactive damage, self-KO moves, two-turn moves and moves that
  fail without setup. See `EXCLUDED_MOVES`.
- **Reliable** means at least 90% accuracy after weather (for example, Blizzard in snow) and No Guard.
- **Guaranteed OHKO** means the lowest damage roll KOs. **Possible OHKO** means the highest roll does.

**Not modelled:** crits, speed order and who moves first, Sucker Punch or Payback conditions
(Payback assumes the slower user moves second), and items other than Mega Stones. Multi-hit
moves with 2–5 hits assume 3 hits, or 5 with Skill Link.
