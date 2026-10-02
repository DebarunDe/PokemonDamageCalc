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

function runCalc(request) {
  const attacker = makePokemon(request.attacker);
  const defender = makePokemon(request.defender);
  const moveName = resolve(gen.moves, 'move', request.move);
  const f = request.field || {};
  const doubles = !!f.doubles;

  // Spread moves (Earthquake, Heat Wave, ...) take 0.75x damage in doubles when
  // they hit more than one target. `spread: false` models a spread move with only
  // one target left; `spread: true` on a single-target move is a mistake. The
  // override goes through `overrides` because the calc clones the move.
  const isSpreadMove = SPREAD_TARGETS.includes(gen.moves.get(toID(moveName)).target);
  if (request.spread === true && !(doubles && isSpreadMove)) {
    throw new CalcError(doubles ?
      `'${moveName}' only hits one target` :
      `Spread damage only applies in doubles`);
  }
  const move = new Move(gen, moveName, {
    isCrit: !!request.is_crit,
    overrides: request.spread === false && isSpreadMove ? {target: 'normal'} : undefined,
  });
  if (move.category === 'Status') {
    throw new CalcError(`'${moveName}' is a status move and deals no direct damage`);
  }

  const field = new Field({
    gameType: doubles ? 'Doubles' : 'Singles',
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

  const result = calculate(gen, attacker, defender, move, field);
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
    description = `${attacker.name} ${moveName} vs. ${defender.name}: 0 damage`;
  }
  return {
    description,
    move: {
      name: move.name,
      type: move.type,
      category: move.category,
      base_power: move.bp,
      spread: doubles && SPREAD_TARGETS.includes(move.target),
    },
    attacker: describePokemon(attacker),
    defender: describePokemon(defender),
    rolls,
    min,
    max,
    min_percent: Math.round((min / hp) * 1000) / 10,
    max_percent: Math.round((max / hp) * 1000) / 10,
    ko,
  };
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
  species: wrap(({name}) => {
    const s = gen.species.get(toID(resolve(gen.species, 'Pokemon', name)));
    return {
      name: s.name,
      types: s.types,
      base_stats: s.baseStats,
      weight_kg: s.weightkg,
      other_formes: s.otherFormes || [],
    };
  }),
  move: wrap(({name}) => {
    const m = gen.moves.get(toID(resolve(gen.moves, 'move', name)));
    return {
      name: m.name,
      type: m.type,
      category: m.category,
      base_power: m.bp,
      priority: m.priority || 0,
      spread: SPREAD_TARGETS.includes(m.target),
    };
  }),
  list: wrap(({kind}) => {
    const table = {species: gen.species, moves: gen.moves, items: gen.items, abilities: gen.abilities, natures: gen.natures}[kind];
    if (!table) throw new CalcError(`Unknown list kind '${kind}'`);
    return [...table].map(x => x.name).sort();
  }),
};
