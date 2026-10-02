# PokemonDamageCalc

A Pokémon Champions damage calculator for Python.

Damage comes from [`@smogon/calc`](https://github.com/smogon/damage-calc), the engine behind the
Showdown calculator, which supports Champions natively. It is bundled into one JS file and run
inside Python through an embedded V8 ([`mini-racer`](https://pypi.org/project/mini-racer/)), so
**Node is not needed to use the package**. Movepools and ability lists come from
[Pokémon Showdown's `champions` mod](https://github.com/smogon/pokemon-showdown/tree/master/data/mods/champions).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

## Command line

```console
$ champcalc calc Garchomp Earthquake Incineroar --a-nature Jolly --a-sp "32 Atk / 32 Spe" --d-sp "32 HP / 2 Def"
32 Atk Garchomp Earthquake vs. 32 HP / 2 Def Incineroar: 186-218 (92 - 107.9%) -- 43.8% chance to OHKO

$ champcalc info incineroar
Incineroar  [Fire / Dark]
Abilities: Blaze, Intimidate
Base stats: HP 95  ATK 115  DEF 90  SPA 80  SPD 90  SPE 60  (BST 530)

$ champcalc moves garchomp-mega
```

`champcalc calc ATTACKER MOVE DEFENDER` takes these options:

| Attacker | Defender | Meaning |
|---|---|---|
| `--a-nature` | `--d-nature` | e.g. `Jolly` |
| `--a-ability` | `--d-ability` | defaults to the Pokémon's first ability |
| `--a-item` | `--d-item` | e.g. `"Life Orb"` |
| `--a-sp` | `--d-sp` | Stat Points: `"32 Atk / 32 Spe"` or `atk=32,spe=32` |
| `--a-atk` `--a-def` `--a-spa` `--a-spd` `--a-spe` | `--d-atk` `--d-def` `--d-spa` `--d-spd` `--d-spe` | one stat stage, -6 to +6, e.g. `--d-def +2` (`--a-spatk`/`--a-spdef` also work) |
| `--a-boosts` | `--d-boosts` | several stages at once: `"+1 Atk / -1 Def"` or `def=-1` |
| `--a-status` | `--d-status` | `brn`, `par`, `psn`, `tox`, `slp`, `frz` |
| `--a-hp` | `--d-hp` | current HP as a percentage |

Field options: `--crit`, `--weather {Sun,Rain,Sand,Snow}`, `--terrain {Electric,Grassy,Psychic,Misty}`,
`--reflect`, `--light-screen`, `--aurora-veil` (screens are on the defender's side). Add `--rolls` to
print all 16 damage rolls or `--json` for the full result.

### Doubles

```console
$ champcalc calc Garchomp Earthquake Incineroar --a-nature Jolly --a-sp "32 Atk" --d-sp "32 HP / 2 Def" --doubles
32 Atk Garchomp Earthquake vs. 32 HP / 2 Def Incineroar: 138-164 (68.3 - 81.1%) -- guaranteed 2HKO (spread)
```

| Option | Meaning |
|---|---|
| `--doubles` | doubles battle: moves that hit several targets take 0.75x damage, screens cut damage to about 2/3 instead of 1/2 |
| `--single-target` | a spread move has only one target left, so no spread penalty |
| `--helping-hand` | the attacker's ally used Helping Hand |
| `--friend-guard` | the defender's ally has Friend Guard |

Spread moves are marked `(spread)` in the output. Steely Spirit from an ally is not offered:
`@smogon/calc`'s Champions mechanics do not apply it yet. Battery, Power Spot and Flower Gift are not
in Champions.

Battles are at level 50, which Champions always uses. Stat Points are capped at 32 per stat
and 66 in total. Unknown or non-Champions Pokémon, moves, items, abilities and natures are errors.
A move the attacker cannot learn or an ability it cannot have is a warning.

## Python

```python
from champcalc import Calculator, Field, PokemonSet

calc = Calculator()  # loads the engine once (~0.2 s); reuse it
result = calc.calculate(
    PokemonSet("Garchomp", nature="Jolly", sp={"atk": 32, "spe": 32}),
    PokemonSet("Incineroar", sp={"hp": 32, "def": 2}),
    "Earthquake",
    Field(weather="sand"),
)
# Doubles: Field(doubles=True, helping_hand=True, friend_guard=True, aurora_veil=True)
# A spread move with one target left: calc.calculate(..., spread=False)
result.min, result.max            # 186, 218
result.min_percent, result.max_percent
result.rolls                      # all 16 rolls
result.ko.text                    # '43.8% chance to OHKO (81.3% ... after sandstorm damage)'
result.warnings                   # e.g. illegal move or ability

calc.species("Incineroar")        # types, base stats, abilities, formes
calc.learnset("Garchomp-Mega")    # Champions movepool
calc.all_species, calc.all_moves, calc.all_items, calc.all_abilities
```

## Updating the data

The embedded engine and data live in `src/champcalc/data/` and are generated from `js/`:

```bash
cd js
npm install
npm run build   # rewrites src/champcalc/data/calc.js and champions.json
```

Each regulation (M-A, M-B, M-C, ...) is read from a pinned Pokémon Showdown commit and mod, listed
under `config.regulations` in `js/package.json`; `Calculator("M-B")` picks one, and the default is
`config.defaultRegulation`. To pick up a new regulation or balance patch, add or bump an entry there,
update `@smogon/calc`, rebuild, then run `pytest`.
The build prints a warning for any movepool entry or ability the calc does not know about.
