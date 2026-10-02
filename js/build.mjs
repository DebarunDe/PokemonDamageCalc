// Produces the two files embedded in the Python package:
//   src/champcalc/data/calc.js        - entry.js + @smogon/calc bundled into one script
//   src/champcalc/data/champions.json - movepools and ability lists (keyed by the calc's species
//                                       names) and move accuracies
//
// Movepools come from Pokemon Showdown's `champions` mod and ability lists from
// its pokedex, both at a pinned commit (package.json -> config.showdownCommit).
// @smogon/calc only stores one default ability per species, hence the pokedex.
// To pick up a balance patch, bump that commit and `@smogon/calc`, then rebuild.
import {readFileSync, writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {Generations} from '@smogon/calc';

const pkg = JSON.parse(readFileSync(new URL('package.json', import.meta.url)));
const OUT = new URL('../src/champcalc/data/', import.meta.url);
const commit = pkg.config.showdownCommit;

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

async function fetchShowdownFile(path) {
  const url = `https://raw.githubusercontent.com/smogon/pokemon-showdown/${commit}/${path}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fetching ${url} failed: ${response.status}`);
  return response.text();
}

// Showdown's data files are a single object literal behind a type annotation.
async function fetchShowdownTable(path) {
  const source = await fetchShowdownFile(path);
  const literal = source.slice(source.indexOf('= {') + 2).trim().replace(/;$/, '');
  return new Function(`return (${literal});`)();
}

const upstreamLearnsets = await fetchShowdownTable('data/mods/champions/learnsets.ts');
const upstreamPokedex = await fetchShowdownTable('data/pokedex.ts');

// moves.ts holds TypeScript callbacks, so read each move's top-level
// `accuracy:` line instead of evaluating the file. `true` means it never misses.
function parseAccuracies(source) {
  const accuracies = {};
  const entries = [...source.matchAll(/^\t(\w+): \{$/gm)];
  entries.forEach((entry, i) => {
    const block = source.slice(entry.index, entries[i + 1]?.index ?? source.length);
    const match = block.match(/^\t\taccuracy: (true|\d+),$/m);
    if (match) accuracies[entry[1]] = match[1] === 'true' ? true : Number(match[1]);
  });
  return accuracies;
}
const accuracyById = {
  ...parseAccuracies(await fetchShowdownFile('data/moves.ts')),
  // Champions changes some accuracies, e.g. Crabhammer is 95%.
  ...parseAccuracies(await fetchShowdownFile('data/mods/champions/moves.ts')),
};

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
const learnsets = {};
const abilities = {};
const problems = [];
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
  } else {
    problems.push(`no learnset for ${species.name}`);
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

const sorted = table => Object.fromEntries(Object.entries(table).sort(([a], [b]) => a.localeCompare(b)));
writeFileSync(
  new URL('champions.json', OUT),
  JSON.stringify(
    {showdown_commit: commit, learnsets: sorted(learnsets), abilities: sorted(abilities), accuracy: sorted(accuracy)},
    null,
    1
  ) + '\n'
);
for (const p of problems) console.warn(`warning: ${p}`);
console.log(`Wrote calc.js and data for ${Object.keys(learnsets).length} Pokemon (Showdown ${commit.slice(0, 7)})`);
