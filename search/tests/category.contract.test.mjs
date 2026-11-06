import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createSearch } from '../engine.js';
import { linkEntries } from '../entries.js';

const links = JSON.parse(await readFile(new URL('../../links.json', import.meta.url), 'utf8'));
const sourceURL = (record) => record.url ?? record.links[0].url;

// Literal title matches are required candidates, regardless of relevance order.
const titleMatches = links.filter((record) => /\bxlstm\b/iu.test(record.title));
const visionPapers = titleMatches.filter((record) => record.model === 'vision-lstm' && record.type === 'paper');

function familyCounts(records, familyKey) {
  const counts = {};
  for (const record of records) counts[record[familyKey]] = (counts[record[familyKey]] ?? 0) + 1;
  return counts;
}

test('keyword candidates preserve the Vision-LSTM category despite matches in other families', async (t) => {
  assert.ok(visionPapers.length > 0, 'links.json must contain the Vision-LSTM paper with xLSTM in its title');
  assert.ok(new Set(titleMatches.map((record) => record.model)).size > 2, 'The source must exercise multiple matching families');
  assert.ok(titleMatches.filter((record) => record.model !== 'vision-lstm').length > visionPapers.length,
    'Other families must supply more literal title matches than Vision-LSTM');

  const requests = [];
  t.mock.method(globalThis, 'fetch', (input) => {
    requests.push(String(input));
    return Promise.reject(new Error('Network is disabled in the category contract test'));
  });
  let latest = [];
  const engine = createSearch(linkEntries(structuredClone(links)), {
    onResults(results) { latest = structuredClone(results); },
    onStatus() {},
  });
  t.after(() => engine.dispose());
  await engine.setSemantic(false);
  await engine.search('xLSTM');

  t.diagnostic(JSON.stringify({
    requiredVisionPapers: visionPapers.map(({ title, url }) => ({ title, url })),
    sourceTitleMatchesByFamily: familyCounts(titleMatches, 'model'),
    candidatesByFamily: familyCounts(latest, 'group'),
  }));

  // The UI can filter only candidates supplied through this public callback.
  for (const record of [...visionPapers, ...titleMatches.filter((item) => !visionPapers.includes(item))]) {
    assert.ok(latest.some((result) => result.id === sourceURL(record) && result.group === record.model),
      `Keyword candidates omitted ${record.model}: ${record.title} (${sourceURL(record)})`);
  }
  assert.ok(latest.every((result) => result.match === 'keyword'), 'Semantic search must remain disabled');
  assert.deepEqual(requests, [], 'Keyword category filtering must not request model or index assets');

  await engine.search('');
  assert.deepEqual(latest, [], 'Clearing the query must remove the search candidate restriction');
  t.diagnostic('Cleared query: []; network requests: 0');
});
