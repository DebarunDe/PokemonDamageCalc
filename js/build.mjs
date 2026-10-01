// Produces the two files embedded in the Python package:
//   src/champcalc/data/calc.js        - entry.js + @smogon/calc bundled into one script
//   src/champcalc/data/champions.json - movepools and ability lists, keyed by the calc's species names
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

// Showdown's data files are a single object literal behind a type annotation.
async function fetchShowdownTable(path) {
  const url = `https://raw.githubusercontent.com/smogon/pokemon-showdown/${commit}/${path}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fetching ${url} failed: ${response.status}`);
  const source = await response.text();
  const literal = source.slice(source.indexOf('= {') + 2).trim().replace(/;$/, '');
  return new Function(`return (${literal});`)();
}

const upstreamLearnsets = await fetchShowdownTable('data/mods/champions/learnsets.ts');
const upstreamPokedex = await fetchShowdownTable('data/pokedex.ts');

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

const sorted = table => Object.fromEntries(Object.entries(table).sort(([a], [b]) => a.localeCompare(b)));
writeFileSync(
  new URL('champions.json', OUT),
  JSON.stringify({showdown_commit: commit, learnsets: sorted(learnsets), abilities: sorted(abilities)}, null, 1) + '\n'
);
for (const p of problems) console.warn(`warning: ${p}`);
console.log(`Wrote calc.js and data for ${Object.keys(learnsets).length} Pokemon (Showdown ${commit.slice(0, 7)})`);
