// Produces the two files embedded in the Python package:
//   src/champcalc/data/calc.js        - entry.js + @smogon/calc bundled into one script
//   src/champcalc/data/champions.json - per regulation: legal roster, movepools and ability lists
//                                       (keyed by the calc's species names) and move accuracies
//
// Each regulation is read from Pokemon Showdown at a pinned commit and mod
// (package.json -> config.regulations). A mod without its own file for some
// data falls back to the `champions` mod at the same commit, as Showdown does.
// Ability lists come from the pokedex, because @smogon/calc only stores one
// default ability per species. To pick up a new regulation or balance patch,
// add or bump an entry there, update `@smogon/calc`, then rebuild.
import {readFileSync, writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {Generations} from '@smogon/calc';

const pkg = JSON.parse(readFileSync(new URL('package.json', import.meta.url)));
const OUT = new URL('../src/champcalc/data/', import.meta.url);
const {regulations: REGULATIONS, defaultRegulation} = pkg.config;

await build({
  entryPoints: [new URL('entry.js', import.meta.url).pathname],
  bundle: true,
  format: 'iife',
  platform: 'neutral',
  mainFields: ['module', 'main'],
  target: 'es2020',
  outfile: new URL('calc.js', OUT).pathname,
  logLevel: 'warning',
});

const toID = text => ('' + text).toLowerCase().replace(/[^a-z0-9]+/g, '');

const fileCache = new Map();
async function fetchShowdownFile(commit, path, {optional = false} = {}) {
  const url = `https://raw.githubusercontent.com/smogon/pokemon-showdown/${commit}/${path}`;
  if (!fileCache.has(url)) {
    const response = await fetch(url);
    if (response.status === 404 && optional) {
      fileCache.set(url, null);
    } else if (!response.ok) {
      throw new Error(`Fetching ${url} failed: ${response.status}`);
    } else {
      fileCache.set(url, await response.text());
    }
  }
  return fileCache.get(url);
}

// Showdown's data files are a single object literal behind a type annotation.
function parseTable(source) {
  const literal = source.slice(source.indexOf('= {') + 2).trim().replace(/;$/, '');
  return new Function(`return (${literal});`)();
}

// A mod's own file, else the `champions` mod's at the same commit.
async function modTable(commit, mod, file) {
  const own = await fetchShowdownFile(commit, `data/mods/${mod}/${file}`, {optional: true});
  return parseTable(own ?? await fetchShowdownFile(commit, `data/mods/champions/${file}`));
}

// moves.ts holds TypeScript callbacks, so read each move's top-level
// `accuracy:` line instead of evaluating the file. `true` means it never misses.
function parseAccuracies(source) {
  const accuracies = {};
  if (!source) return accuracies;
  const entries = [...source.matchAll(/^\t(\w+): \{$/gm)];
  entries.forEach((entry, i) => {
    const block = source.slice(entry.index, entries[i + 1]?.index ?? source.length);
    const match = block.match(/^\t\taccuracy: (true|\d+),$/m);
    if (match) accuracies[entry[1]] = match[1] === 'true' ? true : Number(match[1]);
  });
  return accuracies;
}

// Formes without their own entry (Megas, Aegislash-Blade, Gourgeist sizes, ...)
// use the nearest named parent (Meowstic-F-Mega -> Meowstic-F -> Meowstic),
// falling back to the calc's base species (Floette-Mega -> Floette-Eternal).
function findEntry(species, lookup) {
  for (const name of [species.name, species.baseSpecies].filter(Boolean)) {
    for (let parts = name.split('-'); parts.length; parts.pop()) {
      const found = lookup(toID(parts.join('-')));
      if (found) return found;
    }
  }
  return null;
}

const gen = Generations.get(0);
// @smogon/calc has no plain Aegislash; its Aegislash-Both attacks as Blade and
// defends as Shield, which is how Stance Change plays out.
const CALC_NAME = {aegislash: 'Aegislash-Both'};
const sorted = table => Object.fromEntries(Object.entries(table).sort(([a], [b]) => a.localeCompare(b)));

async function buildRegulation(name, {commit, mod}) {
  const problems = [];
  const upstreamLearnsets = await modTable(commit, mod, 'learnsets.ts');
  const upstreamFormats = await modTable(commit, mod, 'formats-data.ts');
  const upstreamPokedex = parseTable(await fetchShowdownFile(commit, 'data/pokedex.ts'));
  const accuracyById = {
    ...parseAccuracies(await fetchShowdownFile(commit, 'data/moves.ts')),
    // Champions changes some accuracies, e.g. Crabhammer is 95%.
    ...parseAccuracies(await fetchShowdownFile(commit, 'data/mods/champions/moves.ts')),
    ...(mod === 'champions' ? {} :
      parseAccuracies(await fetchShowdownFile(commit, `data/mods/${mod}/moves.ts`, {optional: true}))),
  };

  const learnsets = {};
  const abilities = {};
  for (const species of gen.species) {
    const learnset = findEntry(species, id => {
      const l = upstreamLearnsets[id]?.learnset;
      return l && Object.keys(l).length ? l : null;
    });
    if (learnset) {
      const moves = [];
      for (const id of Object.keys(learnset)) {
        const move = gen.moves.get(id);
        if (move) moves.push(move.name);
        else problems.push(`${species.name}: move '${id}' is unknown to @smogon/calc`);
      }
      learnsets[species.name] = moves.sort();
    }

    const entry = findEntry(species, id => upstreamPokedex[id]);
    if (entry) {
      abilities[species.name] = Object.values(entry.abilities).filter(a => {
        if (gen.abilities.get(toID(a))) return true;
        problems.push(`${species.name}: ability '${a}' is not in Champions, skipped`);
        return false;
      });
    } else {
      problems.push(`no pokedex entry for ${species.name}`);
    }
  }

  const accuracy = {};
  for (const move of gen.moves) {
    if (move.id === 'nomove') continue;
    if (move.id in accuracyById) accuracy[move.name] = accuracyById[move.id];
    else problems.push(`no accuracy for ${move.name}`);
  }

  // Legal roster. A forme is legal when the formats data gives it a tier, or
  // when it is not battle-only (Mega formes count as tiered) and its base
  // species is legal. Formes identical in battle to an earlier one (Vivillon
  // patterns, Squawkabilly colours sharing abilities, ...) are dropped.
  const isLegalId = id => !!upstreamFormats[id]?.tier && upstreamFormats[id].tier !== 'Illegal';
  const legal = [];
  const seen = new Set();
  for (const species of [...gen.species].sort((a, b) => a.name.localeCompare(b.name))) {
    const id = toID(species.name);
    const dex = upstreamPokedex[id];
    const aliased = Object.entries(CALC_NAME).find(([, n]) => n === species.name)?.[0];
    let ok = isLegalId(id) || (aliased && isLegalId(aliased));
    if (!ok && dex && !dex.battleOnly && species.baseSpecies) ok = isLegalId(toID(species.baseSpecies));
    if (!ok) continue;
    const signature = JSON.stringify([
      species.types, species.baseStats, abilities[species.name], learnsets[species.name],
    ]);
    if (seen.has(signature)) continue;
    seen.add(signature);
    legal.push(species.name);
  }
  for (const n of legal) {
    if (!learnsets[n]) problems.push(`no learnset for legal ${n}`);
  }
  for (const id of Object.keys(upstreamFormats)) {
    if (!isLegalId(id)) continue;
    const calcName = CALC_NAME[id] || gen.species.get(id)?.name;
    if (!calcName || !gen.species.get(toID(calcName))) problems.push(`legal '${id}' is unknown to @smogon/calc`);
  }

  for (const p of problems) console.warn(`warning (${name}): ${p}`);
  console.log(`${name}: ${legal.length} legal Pokemon (Showdown ${commit.slice(0, 7)}, mod ${mod})`);
  return {
    showdown_commit: commit,
    showdown_mod: mod,
    legal,
    learnsets: sorted(learnsets),
    abilities: sorted(abilities),
    accuracy: sorted(accuracy),
  };
}

const regulations = {};
for (const [name, source] of Object.entries(REGULATIONS)) {
  regulations[name] = await buildRegulation(name, source);
}
writeFileSync(
  new URL('champions.json', OUT),
  JSON.stringify({default_regulation: defaultRegulation, regulations}, null, 1) + '\n'
);
console.log(`Wrote calc.js and champions.json (default regulation ${defaultRegulation})`);
