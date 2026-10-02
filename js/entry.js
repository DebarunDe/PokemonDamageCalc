// Thin wrapper around @smogon/calc, bundled into a single script that the
// Python package evaluates in an embedded V8 (mini-racer). Every function
// takes and returns JSON strings so the Python side never handles JS objects.
import {calculate, Field, Generations, Move, Pokemon} from '@smogon/calc';

// In @smogon/calc, generation 0 is Pokemon Champions.
const gen = Generations.get(0);

const SPREAD_TARGETS = ['allAdjacent', 'allAdjacentFoes'];

class CalcError extends Error {}

function toID(text) {
  return ('' + text).toLowerCase().replace(/[^a-z0-9]+/g, '');
}

// The calc silently accepts unknown names, so resolve everything against the
// Champions data first and fail loudly on typos or non-Champions entries.
function resolve(table, kind, name) {
  const entry = table.get(toID(name));
  if (!entry) throw new CalcError(`Unknown ${kind} in Pokemon Champions: '${name}'`);
  return entry.name;
}

function makePokemon(set) {
  const species = resolve(gen.species, 'Pokemon', set.species);
  const options = {
    nature: set.nature ? resolve(gen.natures, 'nature', set.nature) : undefined,
    ability: set.ability ? resolve(gen.abilities, 'ability', set.ability) : undefined,
    item: set.item ? resolve(gen.items, 'item', set.item) : undefined,
    // Champions Stat Points are passed through the calc's `evs` field.
    evs: set.sp || {},
    boosts: set.boosts || {},
    status: set.status || '',
  };
  if (set.cur_hp_percent != null) {
    const max = new Pokemon(gen, species, options).maxHP();
    options.curHP = Math.max(1, Math.floor((max * set.cur_hp_percent) / 100));
  }
  return new Pokemon(gen, species, options);
}

function describePokemon(pokemon) {
  return {
    species: pokemon.name,
    types: pokemon.types,
    ability: pokemon.ability || null,
    item: pokemon.item || null,
    nature: pokemon.nature,
    stats: pokemon.stats,
    max_hp: pokemon.maxHP(),
    cur_hp: pokemon.curHP(),
  };
}

// damage is a number, a list of 16 rolls, or one list per hit (multi-hit,
// Parental Bond). Collapse it to one list of totals per roll.
function flattenRolls(damage) {
  if (typeof damage === 'number') return [damage];
  if (damage.length && Array.isArray(damage[0])) {
    return damage[0].map((_, i) => damage.reduce((sum, hit) => sum + hit[i], 0));
  }
  return damage.slice();
}

// Spread moves (Earthquake, Heat Wave, ...) take 0.75x damage in doubles when
// they hit more than one target. `spread: false` models a spread move with only
// one target left; `spread: true` on a single-target move is a mistake. The
// override goes through `overrides` because the calc clones the move.
function makeMove(name, {doubles = false, spread = null, isCrit = false} = {}) {
  const moveName = resolve(gen.moves, 'move', name);
  const isSpreadMove = SPREAD_TARGETS.includes(gen.moves.get(toID(moveName)).target);
  if (spread === true && !(doubles && isSpreadMove)) {
    throw new CalcError(doubles ?
      `'${moveName}' only hits one target` :
      `Spread damage only applies in doubles`);
  }
  const move = new Move(gen, moveName, {
    isCrit,
    overrides: spread === false && isSpreadMove ? {target: 'normal'} : undefined,
  });
  if (move.category === 'Status') {
    throw new CalcError(`'${moveName}' is a status move and deals no direct damage`);
  }
  return move;
}

function makeField(f = {}) {
  return new Field({
    gameType: f.doubles ? 'Doubles' : 'Singles',
    weather: f.weather || undefined,
    terrain: f.terrain || undefined,
    attackerSide: {isHelpingHand: !!f.helping_hand},
    defenderSide: {
      isReflect: !!f.reflect,
      isLightScreen: !!f.light_screen,
      isAuroraVeil: !!f.aurora_veil,
      isFriendGuard: !!f.friend_guard,
    },
  });
}

function percent(damage, hp) {
  return Math.round((damage / hp) * 1000) / 10;
}

function runCalc(request) {
  const attacker = makePokemon(request.attacker);
  const defender = makePokemon(request.defender);
  const f = request.field || {};
  const move = makeMove(request.move, {doubles: !!f.doubles, spread: request.spread, isCrit: !!request.is_crit});
  const result = calculate(gen, attacker, defender, move, makeField(f));
  const [min, max] = result.range();
  const rolls = flattenRolls(result.damage);
  const hp = defender.maxHP();
  let ko = null;
  let description;
  if (max > 0) {
    const k = result.kochance();
    ko = {chance: k.chance ?? null, n: k.n, text: k.text};
    description = result.desc();
  } else {
    description = `${attacker.name} ${move.name} vs. ${defender.name}: 0 damage`;
  }
  return {
    description,
    move: {
      name: move.name,
      type: move.type,
      category: move.category,
      base_power: move.bp,
      spread: !!f.doubles && SPREAD_TARGETS.includes(move.target),
    },
    attacker: describePokemon(attacker),
    defender: describePokemon(defender),
    rolls,
    min,
    max,
    min_percent: percent(min, hp),
    max_percent: percent(max, hp),
    ko,
  };
}

// Many moves from one attacker into one defender, without per-move JSON
// round trips or KO-chance text. Used for bulk analysis.
function runMany(request) {
  const attacker = makePokemon(request.attacker);
  const defender = makePokemon(request.defender);
  const f = request.field || {};
  const field = makeField(f);
  const hp = defender.maxHP();
  const results = request.moves.map(name => {
    const result = calculate(gen, attacker, defender, makeMove(name, {doubles: !!f.doubles}), field);
    const [min, max] = result.range();
    return {
      move: result.move.name,
      // After ability and weather changes, e.g. Pixilate or Weather Ball.
      type: result.move.type,
      category: result.move.category,
      min,
      max,
      min_percent: percent(min, hp),
      max_percent: percent(max, hp),
    };
  });
  return {
    attacker: describePokemon(attacker),
    defender: describePokemon(defender),
    results,
  };
}

// Bulk analysis: many attacker/defender/field jobs in one call. `moves` is
// shared and each job lists the indices it uses. Returns [min, max] per move.
function runBatch({moves, jobs}) {
  const cache = new Map();
  const moveFor = (index, doubles) => {
    const key = `${index}|${doubles}`;
    if (!cache.has(key)) cache.set(key, makeMove(moves[index], {doubles}));
    return cache.get(key);
  };
  return jobs.map(job => {
    const attacker = makePokemon(job.attacker);
    const defender = makePokemon(job.defender);
    const f = job.field || {};
    const field = makeField(f);
    return {
      hp: defender.maxHP(),
      damage: job.moves.map(i => calculate(gen, attacker, defender, moveFor(i, !!f.doubles), field).range()),
    };
  });
}

// Mega Stone item name for each Mega forme, e.g. 'Charizard-Mega-Y' -> 'Charizardite Y'.
const MEGA_STONES = {};
for (const item of gen.items) {
  for (const forme of Object.values(item.megaStone || {})) MEGA_STONES[forme] = item.name;
}

function wrap(fn) {
  return (json) => {
    try {
      return JSON.stringify({ok: true, value: fn(JSON.parse(json))});
    } catch (e) {
      const known = e instanceof CalcError;
      return JSON.stringify({ok: false, user_error: known, error: known ? e.message : String(e)});
    }
  };
}

globalThis.champcalc = {
  calculate: wrap(runCalc),
  calculateMany: wrap(runMany),
  calculateBatch: wrap(runBatch),
  species: wrap(({name}) => {
    const s = gen.species.get(toID(resolve(gen.species, 'Pokemon', name)));
    return {
      name: s.name,
      types: s.types,
      base_stats: s.baseStats,
      weight_kg: s.weightkg,
      other_formes: s.otherFormes || [],
      mega_stone: MEGA_STONES[s.name] || null,
    };
  }),
  move: wrap(({name}) => {
    const m = gen.moves.get(toID(resolve(gen.moves, 'move', name)));
    const full = new Move(gen, m.name);
    // Raw move data leaves out the category of status moves.
    const category = m.category || 'Status';
    const statFor = {Physical: ['atk', 'def'], Special: ['spa', 'spd']}[category] || [null, null];
    return {
      name: m.name,
      type: m.type,
      category,
      base_power: m.basePower || 0,
      priority: m.priority || 0,
      spread: SPREAD_TARGETS.includes(m.target),
      // The stats the move really uses: Body Press attacks with Def, Psyshock
      // hits Def, Foul Play uses the target's Attack.
      offensive_stat: full.overrideOffensiveStat || statFor[0],
      defensive_stat: full.overrideDefensiveStat || statFor[1],
      uses_target_attack: full.overrideOffensivePokemon === 'target',
      contact: !!full.flags?.contact,
      multi_hit: full.hits > 1 || !!m.multihit,
    };
  }),
  list: wrap(({kind}) => {
    const table = {species: gen.species, moves: gen.moves, items: gen.items, abilities: gen.abilities, natures: gen.natures}[kind];
    if (!table) throw new CalcError(`Unknown list kind '${kind}'`);
    return [...table].map(x => x.name).sort();
  }),
};
