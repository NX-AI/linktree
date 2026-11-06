import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { combine, createKeywordIndex, keywordSearch } from '../core.js';
import { createSearch } from '../engine.js';

// Contract: keyword labels as a match annotation.
// - combine(entries, keyword, semantic, labels): labels change only the `match` field.
// - engine: when no entry contains every query word, the semantic list alone ranks the
//   results, and entries that contain at least one query word show match 'both'.

const ids = (results) => results.map(({ id }) => id);
const matches = (results) => Object.fromEntries(results.map(({ id, match }) => [id, match]));
const ranking = (results) => results.map(({ id, score }) => ({ id, score }));

function entry(id, title, notes) {
  return { id, title, text: `Notes: ${notes}`, group: 'g', url: `https://example.test/${id}` };
}

// ---------------------------------------------------------------------------
// combine(): the labels argument
// ---------------------------------------------------------------------------

const plain = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5'].map((id) => entry(id, id, 'content'));
// Index 0: keyword only. Index 1: semantic only. Index 2: keyword and semantic.
// Index 3: semantic only. Index 4: semantic only. Index 5: in no list.
const keywordList = [{ id: 0, score: 3 }, { id: 2, score: 1 }];
const semanticList = [{ id: 1, score: 0.9 }, { id: 2, score: 0.5 }, { id: 3, score: 0.3 }, { id: 4, score: 0.1 }];

test('combine: labels do not change membership, order or score values', () => {
  const base = combine(plain, keywordList, semanticList);
  for (const labels of [
    [{ id: 3, score: 5 }, { id: 5, score: 4 }, { id: 0, score: 1 }],
    [{ id: 0, score: 9 }, { id: 5, score: 2 }, { id: 3, score: 1 }],
    [{ id: 4, score: 1 }, { id: 1, score: 0.5 }],
  ]) {
    const labelled = combine(plain, keywordList, semanticList, labels);
    assert.deepEqual(ranking(labelled), ranking(base));
  }
});

test('combine: a labelled entry in the semantic list gets match both', () => {
  const labels = [{ id: 3, score: 2 }, { id: 0, score: 1 }, { id: 5, score: 0.5 }];
  const result = combine(plain, keywordList, semanticList, labels);
  assert.deepEqual(matches(result), {
    p0: 'keyword', // labelled, keyword list only
    p1: 'semantic', // not labelled
    p2: 'both', // keyword and semantic
    p3: 'both', // labelled and semantic
    p4: 'semantic', // not labelled
  });
  assert.ok(!ids(result).includes('p5'), 'A label must not add an entry that is in no ranked list');
});

test('combine: with an empty keyword list, labels mark semantic entries both and keep semantic order', () => {
  const labels = [{ id: 4, score: 3 }, { id: 2, score: 2 }, { id: 5, score: 1 }];
  const result = combine(plain, [], semanticList, labels);
  assert.deepEqual(ids(result), ['p1', 'p2', 'p3', 'p4']);
  assert.deepEqual(ranking(result), ranking(combine(plain, [], semanticList)));
  assert.deepEqual(matches(result), { p1: 'semantic', p2: 'both', p3: 'semantic', p4: 'both' });
});

test('combine: omitted or empty labels give output identical to the three-argument call', () => {
  for (const [keyword, semantic] of [
    [keywordList, semanticList],
    [[], semanticList],
    [keywordList, []],
  ]) {
    const base = combine(plain, keyword, semantic);
    assert.deepEqual(combine(plain, keyword, semantic, []), base);
    assert.deepEqual(combine(plain, keyword, semantic, undefined), base);
  }
  assert.deepEqual(combine(plain, keywordList), combine(plain, keywordList, [], []));
});

// ---------------------------------------------------------------------------
// engine: semantic results with and without an every-word keyword match
// ---------------------------------------------------------------------------

// Each entry has a unit vector along its own dimension, so a query vector component sets
// that entry's cosine score. "please" and "zzz" occur in no entry.
const corpus = [
  entry('alpha', 'Alpha', 'tirex forecasting'), // 0: tirex
  entry('beta', 'Beta', 'documentation guide'), // 1: documentation, guide
  entry('gamma', 'Gamma', 'tirex documentation'), // 2: tirex, documentation
  entry('delta', 'Delta', 'unrelated cobalt'), // 3: no query word
  entry('epsilon', 'Epsilon', 'tirex'), // 4: tirex; a semantic hit only in the single-hit boundary test
];
// Semantic order: beta (0.4), delta (0.3), gamma (0.2), alpha (0.1). Epsilon is negative.
const components = [0.1, 0.4, 0.2, 0.3, -0.5];
const semanticOrder = [{ id: 1, score: 0.4 }, { id: 3, score: 0.3 }, { id: 2, score: 0.2 }, { id: 0, score: 0.1 }];

function queryVector(values) {
  const vector = new Float32Array(512);
  values.forEach((value, i) => { vector[i] = value; });
  return vector;
}

async function eventually(predicate, message) {
  const deadline = Date.now() + 1500;
  do {
    if (predicate()) return;
    await delay(5);
  } while (Date.now() < deadline);
  assert.fail(message);
}

// Same mock Worker and 512-dimensional asset format as ranking.contract.test.mjs.
async function withEngine(body, entries = corpus) {
  const saved = new Map();
  const replace = (key, value) => {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  };
  const workers = [];
  const results = [];
  const meta = {
    version: 1, model: 'onnx-community/embeddinggemma-2-ONNX',
    revision: 'daa72c51243991dfcaf9f9137d2c573d8f7790c0',
    dtype: 'q8', dim: 512, query_prefix: 'task: search result | query: ', entries,
  };
  const buffer = new ArrayBuffer(entries.length * (4 + 512));
  new Float32Array(buffer, 0, entries.length).fill(1);
  const codes = new Int8Array(buffer, entries.length * 4);
  entries.forEach((_, i) => { codes[i * 512 + i] = 1; });
  class MockWorker extends EventTarget {
    constructor() { super(); this.messages = []; workers.push(this); }
    postMessage(message) { this.messages.push(structuredClone(message)); }
    terminate() {}
    emit(data) {
      const event = new MessageEvent('message', { data });
      this.dispatchEvent(event);
      this.onmessage?.(event);
    }
  }
  replace('Worker', MockWorker);
  replace('localStorage', { getItem: () => null, setItem() {} });
  replace('navigator', { hardwareConcurrency: 2 });
  replace('fetch', async (input) => {
    const pathname = new URL(String(input)).pathname;
    if (pathname.endsWith('/meta.json')) return new Response(JSON.stringify(meta));
    if (pathname.endsWith('/vectors.bin')) return new Response(buffer.slice(0));
    throw new Error(`Unexpected network request: ${input}`);
  });
  let engine;
  try {
    engine = createSearch(entries, { onResults: (value) => results.push(structuredClone(value)), onStatus() {} });
    await body({ engine, workers, results });
  } finally {
    engine?.dispose();
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
}

/** Enable semantic search, run one query, answer it with `vector`, and return the final results. */
async function semanticResults(query, vector) {
  let final;
  await withEngine(async ({ engine, workers, results }) => {
    engine.setSemantic(true);
    await eventually(() => workers.length === 1, 'Semantic worker did not start');
    const worker = workers[0];
    worker.emit({ type: 'ready', backend: 'WASM' });
    engine.search(query);
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'Worker received no query');
    const request = worker.messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, query);
    const before = results.length;
    worker.emit({ type: 'vector', id: request.id, vector });
    await eventually(() => results.length > before, 'Semantic response produced no results');
    final = results.at(-1);
  });
  return final;
}

test('engine 2a: an every-word keyword match keeps RRF of every-word keyword list and semantic list', async () => {
  const query = 'tirex documentation';
  const all = keywordSearch(createKeywordIndex(corpus), query, { requireAll: true });
  assert.deepEqual(all.map(({ id }) => id), [2], 'Fixture: only gamma contains every query word');
  const result = await semanticResults(query, queryVector(components));
  // Gamma has keyword rank 1 plus semantic rank 3, so it is first; the rest keep semantic order.
  assert.deepEqual(ids(result), ['gamma', 'beta', 'delta', 'alpha']);
  assert.deepEqual(result, combine(corpus, all, semanticOrder));
  assert.deepEqual(matches(result), { gamma: 'both', beta: 'semantic', delta: 'semantic', alpha: 'semantic' });
});

test('engine 2b: a query word in no entry ranks by semantic list alone and labels any-word matches both', async () => {
  const query = 'tirex documentation please';
  assert.deepEqual(keywordSearch(createKeywordIndex(corpus), query, { requireAll: true }), [],
    'Fixture: no entry contains "please"');
  const result = await semanticResults(query, queryVector(components));
  assert.deepEqual(ids(result), ['beta', 'delta', 'gamma', 'alpha'], 'Order must be the semantic order');
  assert.ok(!ids(result).includes('epsilon'), 'A keyword-only entry must not join the semantic ranking');
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], semanticOrder)));
  assert.deepEqual(matches(result), { beta: 'both', delta: 'semantic', gamma: 'both', alpha: 'both' });
});

test('engine 2b: words split across entries rank by semantic list alone and label any-word matches both', async () => {
  const query = 'tirex guide';
  assert.deepEqual(keywordSearch(createKeywordIndex(corpus), query, { requireAll: true }), [],
    'Fixture: no entry contains both "tirex" and "guide"');
  const result = await semanticResults(query, queryVector(components));
  assert.deepEqual(ids(result), ['beta', 'delta', 'gamma', 'alpha']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], semanticOrder)));
  assert.deepEqual(matches(result), { beta: 'both', delta: 'semantic', gamma: 'both', alpha: 'both' });
});

test('engine 2b: semantic entries that contain no query word stay semantic', async () => {
  const query = 'zzz please';
  const result = await semanticResults(query, queryVector(components));
  assert.deepEqual(ids(result), ['beta', 'delta', 'gamma', 'alpha']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], semanticOrder)));
  assert.ok(result.every(({ match }) => match === 'semantic'));
});

test('engine 3: an empty semantic list keeps the any-word keyword ranking', async () => {
  const query = 'tirex documentation please';
  const result = await semanticResults(query, queryVector([-0.1, -0.4, -0.2, -0.3, -0.5]));
  const any = keywordSearch(createKeywordIndex(corpus), query);
  assert.ok(any.length >= 4, 'Fixture: several entries contain one query word');
  assert.deepEqual(result, combine(corpus, any));
  assert.ok(result.every(({ match }) => match === 'keyword'));
});

test('engine 3: keyword phase before the semantic result keeps the any-word keyword ranking', async () => {
  const query = 'tirex documentation please';
  await withEngine(async ({ engine, workers, results }) => {
    engine.setSemantic(true);
    await eventually(() => workers.length === 1, 'Semantic worker did not start');
    engine.search(query);
    await eventually(() => results.at(-1)?.length > 0, 'No keyword results before the semantic result');
    const any = keywordSearch(createKeywordIndex(corpus), query);
    assert.deepEqual(results.at(-1), combine(corpus, any));
    assert.ok(results.at(-1).every(({ match }) => match === 'keyword'));
  });
});

test('engine 3: disabled semantic search keeps the any-word keyword ranking', async () => {
  const query = 'tirex documentation please';
  await withEngine(async ({ engine, workers, results }) => {
    engine.setSemantic(false);
    engine.search(query);
    await eventually(() => results.at(-1)?.length > 0, 'No keyword results with semantic search off');
    const any = keywordSearch(createKeywordIndex(corpus), query);
    assert.deepEqual(results.at(-1), combine(corpus, any));
    assert.ok(results.at(-1).every(({ match }) => match === 'keyword'));
    assert.equal(workers.length, 0);
  });
});

// ---------------------------------------------------------------------------
// engine: boundary between a semantic list of one entry and an empty semantic list
// ---------------------------------------------------------------------------

// A semantic list with exactly one entry follows the non-empty rules (2a/2b), not rule 3.
// In each fixture, rule 3 (any-word keyword ranking) gives a different output.

const anyWord = (query) => keywordSearch(createKeywordIndex(corpus), query);
const everyWord = (query) => keywordSearch(createKeywordIndex(corpus), query, { requireAll: true });

function assertNotEmptyRule(result, query) {
  assert.notDeepEqual(result, combine(corpus, anyWord(query)),
    'A semantic list of one entry must not use the any-word keyword ranking');
}

test('engine boundary 2b: one semantic entry with no query word is the only result, match semantic', async () => {
  const query = 'tirex documentation please';
  assert.deepEqual(everyWord(query), [], 'Fixture: no entry contains every query word');
  assert.ok(anyWord(query).length >= 4, 'Fixture: several entries contain one query word');
  // Only delta (index 3, no query word) has a positive similarity.
  const result = await semanticResults(query, queryVector([-0.25, -0.5, -0.125, 0.25, -0.5]));
  assert.deepEqual(ids(result), ['delta']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], [{ id: 3, score: 0.25 }])));
  assert.deepEqual(matches(result), { delta: 'semantic' });
  assertNotEmptyRule(result, query);
});

test('engine boundary 2b: one semantic entry with a query word is the only result, match both', async () => {
  const query = 'tirex documentation please';
  // Only beta (index 1, contains "documentation") has a positive similarity.
  const result = await semanticResults(query, queryVector([-0.25, 0.5, -0.125, -0.25, -0.5]));
  assert.deepEqual(ids(result), ['beta']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], [{ id: 1, score: 0.5 }])));
  assert.deepEqual(matches(result), { beta: 'both' });
  for (const id of ['alpha', 'gamma', 'epsilon']) {
    assert.ok(!ids(result).includes(id), `Any-word match ${id} is not in the semantic list and must not join`);
  }
  assertNotEmptyRule(result, query);
});

test('engine boundary 2b: with query words split across entries, one semantic entry is the only result, match both', async () => {
  const query = 'tirex guide';
  assert.deepEqual(everyWord(query), [], 'Fixture: no entry contains both "tirex" and "guide"');
  // Only epsilon (index 4, contains "tirex") has a positive similarity.
  const result = await semanticResults(query, queryVector([-0.25, -0.5, -0.125, -0.25, 0.125]));
  assert.deepEqual(ids(result), ['epsilon']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], [{ id: 4, score: 0.125 }])));
  assert.deepEqual(matches(result), { epsilon: 'both' });
  assertNotEmptyRule(result, query);
});

test('engine boundary 2a: one semantic entry that is not the every-word match combines with the every-word list', async () => {
  const query = 'tirex documentation';
  const all = everyWord(query);
  assert.deepEqual(all.map(({ id }) => id), [2], 'Fixture: only gamma contains every query word');
  // Only delta (index 3) has a positive similarity.
  const result = await semanticResults(query, queryVector([-0.25, -0.5, -0.125, 0.25, -0.5]));
  assert.deepEqual(result, combine(corpus, all, [{ id: 3, score: 0.25 }]));
  assert.deepEqual(new Set(ids(result)), new Set(['gamma', 'delta']));
  assert.deepEqual(matches(result), { gamma: 'keyword', delta: 'semantic' });
  for (const id of ['alpha', 'beta', 'epsilon']) {
    assert.ok(!ids(result).includes(id), `Any-word-only match ${id} must not join the RRF ranking`);
  }
  assertNotEmptyRule(result, query);
});

test('engine boundary 2a: one semantic entry that is also the every-word match gives match both', async () => {
  const query = 'tirex documentation';
  const all = everyWord(query);
  // Only gamma (index 2) has a positive similarity.
  const result = await semanticResults(query, queryVector([-0.25, -0.5, 0.125, -0.25, -0.5]));
  assert.deepEqual(result, combine(corpus, all, [{ id: 2, score: 0.125 }]));
  assert.deepEqual(ids(result), ['gamma']);
  assert.deepEqual(matches(result), { gamma: 'both' });
  assertNotEmptyRule(result, query);
});

test('engine boundary 2a: one semantic entry that is an any-word match keeps match semantic, not both', async () => {
  const query = 'tirex documentation';
  const all = everyWord(query);
  // Only alpha (index 0, contains "tirex" but not "documentation") has a positive similarity.
  const result = await semanticResults(query, queryVector([0.25, -0.5, -0.125, -0.25, -0.5]));
  assert.deepEqual(result, combine(corpus, all, [{ id: 0, score: 0.25 }]));
  assert.deepEqual(new Set(ids(result)), new Set(['gamma', 'alpha']));
  assert.deepEqual(matches(result), { gamma: 'keyword', alpha: 'semantic' });
  assertNotEmptyRule(result, query);
});

test('engine boundary: a zero similarity is not positive, so zeros with negatives give rule 3', async () => {
  const query = 'tirex documentation please';
  const result = await semanticResults(query, queryVector([0, -0.5, 0, -0.25, 0]));
  assert.deepEqual(result, combine(corpus, anyWord(query)));
  assert.ok(result.every(({ match }) => match === 'keyword'));
});

test('engine boundary: zero similarities do not join a semantic list of one entry', async () => {
  const query = 'tirex documentation please';
  // Delta is the only positive entry; alpha, beta, gamma and epsilon have similarity zero.
  const result = await semanticResults(query, queryVector([0, 0, 0, 0.25, 0]));
  assert.deepEqual(ids(result), ['delta']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], [{ id: 3, score: 0.25 }])));
  assert.deepEqual(matches(result), { delta: 'semantic' });
});

test('engine 3: an empty semantic list uses the any-word ranking even when an every-word match exists', async () => {
  const query = 'tirex documentation';
  assert.deepEqual(everyWord(query).map(({ id }) => id), [2], 'Fixture: gamma contains every query word');
  const any = anyWord(query);
  assert.ok(any.length > 1, 'Fixture: any-word matches exist that are not every-word matches');
  const result = await semanticResults(query, queryVector([-0.25, -0.5, -0.125, -0.25, -0.5]));
  assert.deepEqual(result, combine(corpus, any));
  assert.ok(result.every(({ match }) => match === 'keyword'));
});

test('engine boundary: two semantic entries follow rule 2b like one entry does', async () => {
  const query = 'tirex documentation please';
  // Delta and alpha are positive; beta, gamma and epsilon are negative.
  const semantic = [{ id: 3, score: 0.5 }, { id: 0, score: 0.25 }];
  const result = await semanticResults(query, queryVector([0.25, -0.5, -0.125, 0.5, -0.5]));
  assert.deepEqual(ids(result), ['delta', 'alpha']);
  assert.deepEqual(ranking(result), ranking(combine(corpus, [], semantic)));
  assert.deepEqual(matches(result), { delta: 'semantic', alpha: 'both' });
});
