import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const links = JSON.parse(await readFile(new URL('../../links.json', import.meta.url), 'utf8'));
const firstURL = (link) => link.url ?? link.links[0].url;
async function adapter() {
  const { linkEntries } = await import('../entries.js');
  assert.equal(typeof linkEntries, 'function', 'search/entries.js must export linkEntries');
  return linkEntries;
}

test('linkEntries preserves source order, titles, public URLs, and the entry contract', async () => {
  const linkEntries = await adapter();
  const entries = linkEntries(structuredClone(links));
  assert.ok(Array.isArray(entries));
  assert.equal(entries.length, links.length, 'Each source record must remain searchable');
  assert.deepEqual(entries.map(({ id }) => id), links.map(firstURL));
  for (let index = 0; index < links.length; index += 1) {
    const source = links[index];
    const entry = entries[index];
    for (const field of ['id', 'title', 'text', 'group', 'url']) {
      assert.equal(typeof entry[field], 'string', `Entry ${index}: missing string field ${field}`);
      assert.ok(entry[field].trim(), `Entry ${index}: empty field ${field}`);
    }
    assert.equal(entry.id, firstURL(source));
    assert.equal(entry.url, firstURL(source));
    assert.equal(entry.title, source.title);
    assert.ok(entry.text.toLowerCase().includes(source.title.toLowerCase()), 'Search text lost the source title');
    assert.ok(!entry.text.includes('[object Object]'), 'Search text contains object coercion instead of source text');
  }
});

test('the existing multi-link article uses its first public URL as its stable id', async () => {
  const linkEntries = await adapter();
  const source = links.find((link) => Array.isArray(link.links) && link.links.length > 1);
  assert.ok(source, 'The corpus fixture must retain the existing multi-link article');
  const [entry] = linkEntries([structuredClone(source)]);
  assert.equal(entry.id, source.links[0].url);
  assert.equal(entry.url, source.links[0].url);
  assert.equal(entry.title, source.title);
});

test('reordered or shortened source cannot retain positional ids from the original source', async () => {
  const linkEntries = await adapter();
  const changed = [...links].reverse().filter((_, index) => index % 2 === 0);
  assert.deepEqual(linkEntries(structuredClone(changed)).map(({ id }) => id), changed.map(firstURL));
  assert.deepEqual(linkEntries([]), []);
});

test('paper text keeps existing author and venue keywords searchable', async () => {
  const linkEntries = await adapter();
  const papers = links.filter((link) => link.type === 'paper');
  const entries = linkEntries(structuredClone(papers));
  for (let index = 0; index < papers.length; index += 1) {
    const source = papers[index];
    const text = entries[index].text.toLowerCase();
    const author = source.meta.split(' et al.')[0].toLowerCase();
    assert.ok(text.includes(author), `Missing author keyword for ${source.title}`);
    assert.ok(text.includes(source.venue.toLowerCase()), `Missing venue keyword for ${source.title}`);
  }
});
