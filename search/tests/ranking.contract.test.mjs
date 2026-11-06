import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { createKeywordIndex, keywordSearch } from '../core.js';
import { createSearch } from '../engine.js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { stageSite } from '../stage.mjs';
import { attachSearch, MATCH_LABELS, RESULT_LIMIT } from '../ui.js';

test('linktree UI exports its integration API without a standalone results renderer', async () => {
  const source = fs.readFileSync(new URL('../ui.js', import.meta.url), 'utf8');
  assert.ok(!source.includes('search-entries'), 'UI still references the standalone entry payload');
  assert.equal(typeof attachSearch, 'function');
  assert.deepEqual(MATCH_LABELS, { keyword: 'Keyword', semantic: 'Semantic', both: 'Both' });
  assert.ok(Number.isInteger(RESULT_LIMIT) && RESULT_LIMIT > 0);

  const nodes = new Map();
  for (const id of ['search', 'semantic-toggle', 'search-status', 'search-progress',
    'search-download', 'search-count', 'search-settings']) {
    nodes.set(id, { value: '', textContent: '', dataset: {}, addEventListener() {},
      closest: () => ({ hidden: false }) });
  }
  const saved = new Map();
  const replace = (key, value) => {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  };
  replace('document', {
    getElementById(id) {
      assert.ok(nodes.has(id), `Unexpected standalone UI target: ${id}`);
      return nodes.get(id);
    },
    querySelector(selector) {
      assert.equal(selector, '.semantic-search');
      return { hidden: true };
    },
    createElement() { assert.fail('attachSearch must not create standalone result elements'); },
  });
  replace('window', { addEventListener() {} });
  replace('localStorage', { getItem: () => 'off', setItem() {} });
  replace('fetch', async () => { throw new Error('No network in UI contract'); });
  let engine;
  let latest;
  try {
    engine = attachSearch(rankingEntries, (results) => { latest = results; });
    nodes.get('search').value = 'amber violet';
    engine.search('amber violet');
    await eventually(() => latest?.length === 3, 'UI did not forward candidates to the existing page');
    assert.ok(latest.every(({ match }) => match === 'keyword'));
  } finally {
    engine?.dispose();
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});

test('linktree staging omits search/entries.json', () => {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), 'linktree-ranking-contract-'));
  try {
    const { entries, searchDir } = stageSite(output);
    assert.ok(entries.length > 0, 'Staging must still return entries for embedding');
    assert.equal(searchDir, path.join(output, 'search'));
    assert.ok(fs.existsSync(path.join(output, 'index.html')), 'Staging did not produce a site');
    assert.ok(fs.existsSync(path.join(searchDir, 'ui.js')), 'Staging omitted the browser UI');
    assert.ok(!fs.existsSync(path.join(searchDir, 'entries.json')), 'Staging emitted a standalone entry payload');
  } finally {
    fs.rmSync(output, { recursive: true, force: true });
  }
});

test('linktree CSS removes standalone result rules and retains match badges', () => {
  const css = fs.readFileSync(new URL('../ui.css', import.meta.url), 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '');
  assert.doesNotMatch(css, /\.search-results?(?![\w-])/);
  assert.match(css, /\.search-match(?![\w-])[^{}]*\{/);
});

function entry(id, fields = {}) {
  return { id, title: 'Foo', text: 'Plain content', group: 'xlstm', url: 'https://example.org/item', ...fields };
}

const labelled = 'Model family: xlstm architecture | Resource type: code | Title: Foo | Details: bar | Description: baz';
const labelWords = ['model', 'family', 'resource', 'type', 'title', 'details', 'description'];

for (const word of labelWords) {
  test(`label-only word "${word}" is not searchable`, () => {
    const index = createKeywordIndex([entry('labelled', { text: labelled })]);
    assert.deepEqual(keywordSearch(index, word), []);
    assert.deepEqual(keywordSearch(index, word, { requireAll: true }), []);
  });

  for (const field of ['text', 'title', 'group', 'url']) {
    test(`label word "${word}" remains searchable in ${field} content`, () => {
      const value = field === 'text' ? `${labelled} | Notes: ${word}`
        : field === 'url' ? `https://example.org/${word}` : word;
      const index = createKeywordIndex([entry('content', { text: labelled, [field]: value })]);
      assert.deepEqual(keywordSearch(index, word).map(({ id }) => id), [0]);
      assert.deepEqual(keywordSearch(index, word, { requireAll: true }).map(({ id }) => id), [0]);
    });
  }
}

test('stripping labels preserves every segment value and inline colon text', () => {
  const index = createKeywordIndex([entry('values', {
    text: `${labelled} | Notes: supports custom model: forecasting`,
  })]);
  for (const word of ['xlstm', 'architecture', 'code', 'foo', 'bar', 'baz', 'supports', 'custom', 'model', 'forecasting']) {
    assert.deepEqual(keywordSearch(index, word).map(({ id }) => id), [0], word);
  }
});

const rankingEntries = [
  entry('partial', { text: 'Title: amber' }),
  entry('complete', { text: 'Title: amber violet' }),
  entry('other-partial', { text: 'Title: violet' }),
  entry('unrelated', { text: 'Title: cobalt' }),
];

test('default and explicit false use OR, while requireAll uses AND', () => {
  const index = createKeywordIndex(rankingEntries);
  const any = keywordSearch(index, 'amber violet');
  assert.deepEqual(any.map(({ id }) => id).sort(), [0, 1, 2]);
  assert.deepEqual(keywordSearch(index, 'amber violet', { requireAll: false }), any);
  const all = keywordSearch(index, 'amber violet', { requireAll: true });
  assert.deepEqual(all.map(({ id }) => id), [1]);
  assert.equal(all[0].score, any.find(({ id }) => id === 1).score);
  assert.deepEqual(keywordSearch(index, 'amber missing', { requireAll: true }), []);
});

test('requireAll applies existing case, punctuation, plural and stop-word tokenization', () => {
  const index = createKeywordIndex([
    entry('both', { text: 'Title: kernel forecast' }),
    entry('one', { text: 'Title: kernel' }),
  ]);
  assert.deepEqual(keywordSearch(index, 'THE KERNELS, and forecasts!', { requireAll: true }).map(({ id }) => id), [0]);
  assert.deepEqual(keywordSearch(index, 'kernel kernel forecast', { requireAll: true }),
    keywordSearch(index, 'kernel forecast', { requireAll: true }));
});

test('requireAll can find its terms across independent entry fields', () => {
  const index = createKeywordIndex([
    entry('split', { title: 'amber', text: 'Details: neutral', group: 'violet' }),
    entry('partial', { title: 'amber', text: 'Details: neutral' }),
  ]);
  assert.deepEqual(keywordSearch(index, 'amber violet', { requireAll: true }).map(({ id }) => id), [0]);
});

test('empty and stop-word-only queries return no results in either mode', () => {
  const index = createKeywordIndex(rankingEntries);
  for (const query of ['', ' ', 'the and for with', 'THE, and!']) {
    assert.deepEqual(keywordSearch(index, query), []);
    assert.deepEqual(keywordSearch(index, query, { requireAll: true }), []);
  }
});

test('results contain only id and finite positive score, sorted descending with ascending id ties', () => {
  const index = createKeywordIndex([
    entry('z', { text: 'amber violet' }),
    entry('a', { text: 'amber violet' }),
    entry('m', { title: 'amber violet', text: 'amber violet' }),
  ]);
  for (const options of [{}, { requireAll: true }]) {
    const results = keywordSearch(index, 'amber violet', options);
    assert.equal(results.length, 3);
    for (const result of results) {
      assert.deepEqual(Object.keys(result).sort(), ['id', 'score']);
      assert.ok(Number.isInteger(result.id));
      assert.ok(Number.isFinite(result.score) && result.score > 0);
    }
    assert.deepEqual(results, [...results].sort((a, b) => b.score - a.score || a.id - b.id));
    assert.equal(results.find(({ id }) => id === 0).score, results.find(({ id }) => id === 1).score);
    assert.ok(results.findIndex(({ id }) => id === 0) < results.findIndex(({ id }) => id === 1));
    assert.ok(new Set(results.map(({ score }) => score)).size > 1, 'Fixture must exercise unequal scores too');
  }
});

async function eventually(predicate, message) {
  const deadline = Date.now() + 1500;
  do {
    if (predicate()) return;
    await delay(5);
  } while (Date.now() < deadline);
  assert.fail(message);
}

// Same EventTarget worker and fixed 512-dimensional asset format as engine.contract.test.mjs.
async function withEngine(body, entries = rankingEntries.slice(0, 2)) {
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

test('semantic fusion marks partial keyword matches semantic and complete matches both', async () => {
  await withEngine(async ({ engine, workers, results }) => {
    engine.setSemantic(true);
    engine.search('amber violet');
    await eventually(() => workers.length === 1, 'Semantic worker did not start');
    const worker = workers[0];
    worker.emit({ type: 'ready', backend: 'WASM' });
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'Worker received no query');
    const request = worker.messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, 'amber violet');
    const vector = new Float32Array(512);
    vector[0] = vector[1] = Math.SQRT1_2;
    const before = results.length;
    worker.emit({ type: 'vector', id: request.id, vector });
    await eventually(() => results.length > before, 'Semantic response produced no results');
    const matches = new Map(results.at(-1).map(({ id, match }) => [id, match]));
    assert.equal(matches.get('partial'), 'semantic');
    assert.equal(matches.get('complete'), 'both');
  });
});

test('semantic search with no positive cosine hits retains keyword OR matches', async () => {
  await withEngine(async ({ engine, workers, results }) => {
    engine.setSemantic(true);
    await eventually(() => workers.length === 1, 'Semantic worker did not start');
    const worker = workers[0];
    worker.emit({ type: 'ready', backend: 'WASM' });
    engine.search('forecasting unknown');
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'Worker received no query');
    const request = worker.messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, 'forecasting unknown');
    const vector = new Float32Array(512);
    vector[0] = -1;
    const before = results.length;
    worker.emit({ type: 'vector', id: request.id, vector });
    await eventually(() => results.length > before, 'Semantic response produced no results');
    assert.ok(results.at(-1).some(({ id, match }) => id === 'a' && match === 'keyword'),
      'No semantic hits must preserve partial keyword matches');
  }, [{ id: 'a', title: 'Alpha', text: 'Description: forecasting', group: 'g', url: 'https://example.test' }]);
});

test('keyword-only engine retains partial OR matches', async () => {
  await withEngine(async ({ engine, workers, results }) => {
    engine.setSemantic(false);
    engine.search('amber violet');
    await eventually(() => results.at(-1)?.length === 2, 'Keyword results omitted a partial match');
    assert.deepEqual(new Map(results.at(-1).map(({ id, match }) => [id, match])),
      new Map([['partial', 'keyword'], ['complete', 'keyword']]));
    assert.equal(workers.length, 0);
  });
});

