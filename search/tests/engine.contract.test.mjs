import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';

// Corpus facts come from projects.yml, not from the search implementation.
// The optional "snippet" field is the item's projects.yml description; result objects must keep it.
const corpus = [
  {
    id: 'https://github.com/NX-AI/flashrnn',
    title: 'FlashRNN',
    text: 'FlashRNN CUDA and Triton kernels for LSTM, GRU, Elman networks and sLSTM, optimised for modern GPU hardware. ICLR 2025',
    group: 'efficiency',
    url: 'https://github.com/NX-AI/flashrnn',
    snippet: 'CUDA and Triton kernels for LSTM, GRU, Elman networks and sLSTM, optimised for modern GPU hardware.',
  },
  {
    id: 'https://github.com/NX-AI/tirex-2',
    title: 'TiRex-2',
    text: 'TiRex-2 Pretrained multivariate time series forecasting model with past and future-known covariates and quantile forecasts, released for zero-shot use. NeurIPS 2026',
    group: 'time-series',
    url: 'https://github.com/NX-AI/tirex-2',
    snippet: 'Pretrained multivariate time series forecasting model with past and future-known covariates and quantile forecasts, released for zero-shot use.',
  },
  {
    id: 'https://github.com/NX-AI/vision-lstm',
    title: 'Vision-LSTM',
    text: 'Vision-LSTM (ViL) adapts xLSTM as a generic backbone for computer vision. Includes pretrained models and a full training pipeline. ICLR 2025',
    group: 'xlstm',
    url: 'https://github.com/NX-AI/vision-lstm',
    snippet: 'Vision-LSTM (ViL) adapts xLSTM as a generic backbone for computer vision. Includes pretrained models and a full training pipeline.',
  },
];
// 'consent' is no longer a valid state: semantic search has no connection-based consent gate.
const states = new Set(['idle', 'loading', 'searching', 'ready', 'off', 'error']);
const PREFERENCE = 'nxai-semantic-search';
// These constants preserve the existing model contract, independently of core.js.
const model = Object.freeze({
  model: 'onnx-community/embeddinggemma-2-ONNX',
  revision: 'daa72c51243991dfcaf9f9137d2c573d8f7790c0',
  dtype: 'q8',
  dim: 512,
  query_prefix: 'task: search result | query: ',
});
const weightsURL = (url) => /\.(?:onnx(?:_data)?|safetensors)(?:$|[?#])|\/pytorch_model[^/]*\.bin(?:$|[?#])/i.test(url);
let importSequence = 0;

async function eventually(predicate, message, timeout = 1200) {
  const end = Date.now() + timeout;
  do {
    if (predicate()) return;
    await delay(5);
  } while (Date.now() < end);
  assert.fail(message);
}

async function settled(promises) {
  let timer;
  try {
    await Promise.race([
      Promise.all(promises),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Public API promises did not settle after cancellation/disposal')), 1200);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function semanticAssets(entries) {
  const meta = { version: 1, ...model, entries: structuredClone(entries) };
  const buffer = new ArrayBuffer(entries.length * (4 + model.dim));
  new Float32Array(buffer, 0, entries.length).fill(1);
  const codes = new Int8Array(buffer, entries.length * 4);
  // Entry i is the unit vector along dimension i.
  entries.forEach((_, i) => { codes[i * model.dim + i] = 1; });
  return { meta, buffer };
}

function unitVector(index) {
  const vector = new Float32Array(model.dim);
  vector[index] = 1;
  return vector;
}

// options.storage: undefined = empty Storage; 'absent' = no localStorage global;
// 'getter-throws' = reading localStorage throws; 'methods-throw' = getItem/setItem throw;
// any string = value stored under the preference key before construction.
// options.connection: navigator.connection value ('none' = no connection object).
// options.fetch: 'reject' (default), 'pending', 'assets' (serves meta.json and vectors.bin),
// or 'fail-once' (the first fetch rejects, later fetches behave like 'assets').
async function withEngine(options, body) {
  const saved = new Map();
  const remember = (key) => {
    if (!saved.has(key)) saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  };
  const replace = (key, value) => {
    remember(key);
    Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  };
  const requests = [];
  const workers = [];
  const pendingFetches = [];
  const returns = [];
  const results = [];
  const statuses = [];
  const writes = [];
  const invalidStates = [];
  const fetchAborts = { count: 0 };
  const fetchFailed = { done: false };
  const entries = structuredClone(options.entries ?? corpus);
  const assets = semanticAssets(entries);
  const observeReturn = (value) => {
    if (value && typeof value.then === 'function') {
      // Attach both handlers immediately. Cancellation may reject a public promise.
      returns.push(Promise.resolve(value).then(() => ({ settled: true }), (error) => ({ settled: true, error: String(error) })));
    }
    return value;
  };
  class MockWorker extends EventTarget {
    constructor(url, workerOptions) {
      super();
      if (options.worker === 'constructor-error') throw new Error('Worker initialization denied by test');
      this.url = String(url);
      this.options = workerOptions;
      this.messages = [];
      this.terminated = false;
      this.fetchedBefore = [...requests];
      workers.push(this);
    }
    postMessage(message) {
      this.messages.push(structuredClone(message));
      if (options.worker === 'runtime-error') queueMicrotask(() => this.fail());
    }
    terminate() { this.terminated = true; }
    emit(data) {
      const event = new MessageEvent('message', { data });
      this.dispatchEvent(event);
      this.onmessage?.(event);
    }
    fail() {
      const event = new Event('error', { cancelable: true });
      Object.defineProperties(event, {
        message: { value: 'Worker failed in deterministic test' },
        error: { value: new Error('Worker failed in deterministic test') },
      });
      this.dispatchEvent(event);
      this.onerror?.(event);
    }
  }
  replace('Worker', options.worker === 'unavailable' ? undefined : MockWorker);
  const connection = options.connection ?? { saveData: false, effectiveType: '4g' };
  replace('navigator', connection === 'none' ? { hardwareConcurrency: 2 } : { connection, hardwareConcurrency: 2 });
  const stored = new Map();
  if (typeof options.storage === 'string' && !['absent', 'getter-throws', 'methods-throw'].includes(options.storage)) {
    stored.set(PREFERENCE, options.storage);
  }
  if (options.storage === 'absent') replace('localStorage', undefined);
  else if (options.storage === 'getter-throws') {
    remember('localStorage');
    Object.defineProperty(globalThis, 'localStorage', {
      configurable: true,
      get() { throw new DOMException('Storage is blocked in this test', 'SecurityError'); },
    });
  } else if (options.storage === 'methods-throw') {
    replace('localStorage', {
      getItem() { throw new DOMException('Storage is blocked in this test', 'SecurityError'); },
      setItem(key, value) { writes.push([key, String(value)]); throw new DOMException('Storage quota exceeded in this test', 'QuotaExceededError'); },
      removeItem() { throw new DOMException('Storage is blocked in this test', 'SecurityError'); },
    });
  } else {
    replace('localStorage', {
      getItem: (key) => (stored.has(key) ? stored.get(key) : null),
      setItem: (key, value) => { writes.push([key, String(value)]); stored.set(key, String(value)); },
      removeItem: (key) => { stored.delete(key); },
    });
  }
  replace('fetch', (input, init = {}) => {
    const url = String(input?.url ?? input);
    requests.push({ url, method: init.method ?? input?.method ?? 'GET' });
    if (options.fetch === 'pending') {
      return new Promise((resolve, reject) => {
        const abort = () => reject(new DOMException('Fetch cancelled in test', 'AbortError'));
        pendingFetches.push(abort);
        const signal = init.signal ?? input?.signal;
        if (signal?.aborted) { fetchAborts.count++; abort(); }
        else signal?.addEventListener('abort', () => { fetchAborts.count++; abort(); }, { once: true });
      });
    }
    if (options.fetch === 'fail-once' && !fetchFailed.done) {
      fetchFailed.done = true;
      return Promise.reject(new TypeError('First index download fails in this test'));
    }
    if (options.fetch === 'assets' || options.fetch === 'fail-once') {
      const { pathname } = new URL(url);
      if (pathname.endsWith('/meta.json')) return Promise.resolve(new Response(JSON.stringify(assets.meta), { status: 200 }));
      if (pathname.endsWith('/vectors.bin')) return Promise.resolve(new Response(assets.buffer.slice(0), { status: 200 }));
    }
    return Promise.reject(new TypeError('Network and model downloads are disabled in Node tests'));
  });
  let engine;
  try {
    const moduleURL = new URL('../engine.js', import.meta.url);
    moduleURL.searchParams.set('contract-test', String(++importSequence));
    const { createSearch } = await import(moduleURL.href);
    assert.equal(typeof createSearch, 'function', 'search/engine.js must export createSearch');
    engine = createSearch(entries, {
      onResults(value) {
        assert.ok(Array.isArray(value), 'onResults must receive an array');
        const snapshot = structuredClone(value);
        for (const item of snapshot) {
          const source = entries.find((entry) => entry.id === item.id);
          assert.ok(source, `Unknown result id: ${item.id}`);
          // Every source field, including the optional snippet, reaches the result unchanged.
          for (const field of Object.keys(source)) assert.equal(item[field], source[field], `Result changed source field ${field}`);
          assert.ok(['keyword', 'semantic', 'both'].includes(item.match), `Invalid match: ${item.match}`);
          assert.ok(Number.isFinite(item.score), 'Result score must be finite');
        }
        results.push(snapshot);
      },
      onStatus(value) {
        assert.ok(value && typeof value === 'object', 'onStatus must receive an object');
        for (const key of ['state', 'enabled', 'loaded', 'total', 'backend', 'message']) {
          assert.ok(Object.hasOwn(value, key), `Missing status key: ${key}`);
        }
        statuses.push(structuredClone(value));
        // Recorded, not thrown: a throw from a timer-driven report would escape the test that caused it.
        if (!states.has(value.state)) invalidStates.push(value.state);
        assert.equal(typeof value.enabled, 'boolean');
        assert.equal(typeof value.message, 'string');
      },
    });
    for (const name of ['search', 'setSemantic', 'dispose']) assert.equal(typeof engine[name], 'function', `Missing method ${name}`);
    await body({
      entries, results, statuses, requests, workers, returns, writes, fetchAborts,
      search: (query) => observeReturn(engine.search(query)),
      semantic: (enabled) => observeReturn(engine.setSemantic(enabled)),
      dispose: () => { const value = observeReturn(engine.dispose()); engine = null; return value; },
    });
    assert.deepEqual(invalidStates, [], `Invalid status states: ${invalidStates.join(', ')}`);
  } finally {
    if (engine) observeReturn(engine.dispose());
    for (const abort of pendingFetches) abort();
    await delay(0);
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
}

function hasKeyword(results, id) {
  return results.some((items) => items.some((item) => item.id === id && ['keyword', 'both'].includes(item.match)));
}

function keywordOnly(items) {
  assert.ok(items.every((item) => item.match === 'keyword'), 'Disabled semantic search returned a semantic match');
}

test('construction is lazy and exposes the documented public API', async () => {
  // An earlier explicit opt-in is stored; construction must still download nothing.
  await withEngine({ worker: 'silent', storage: 'on' }, async ({ requests, workers, statuses, search }) => {
    assert.equal(statuses[0]?.state, 'idle', 'A stored opt-in did not start in the idle state');
    assert.equal(statuses[0].enabled, true, 'A stored opt-in did not enable semantic search');
    await delay(25);
    const modelRequests = requests.filter(({ url }) => weightsURL(url));
    assert.deepEqual(modelRequests, [], 'The heavy model was fetched before the first search');
    search('');
    await delay(350);
    assert.deepEqual(requests, [], 'Construction or an empty query fetched search assets');
    assert.equal(workers.length, 0, 'Construction or an empty query started a Worker');
  });
});

test('disabled semantic search preserves keyword search and source fields without a model', async () => {
  await withEngine({ worker: 'unavailable' }, async ({ semantic, search, results, requests, statuses }) => {
    semantic(false);
    const before = results.length;
    search('fLaShRnN');
    await eventually(() => hasKeyword(results.slice(before), corpus[0].id), 'Case-insensitive title keyword did not find FlashRNN');
    keywordOnly(results.at(-1));
    assert.ok(results.at(-1).every(({ id }) => id === corpus[0].id), 'Disabled keyword search included an unrelated entry');
    assert.deepEqual(requests.filter(({ url }) => weightsURL(url)), [], 'Disabled semantic search requested model weights');
    if (statuses.length) assert.equal(statuses.at(-1).enabled, false);
    const next = results.length;
    search('quantile');
    await eventually(() => hasKeyword(results.slice(next), corpus[1].id), 'Keyword in entry text did not find TiRex-2');
    keywordOnly(results.at(-1));
    assert.ok(results.at(-1).every(({ id }) => id === corpus[1].id), 'Text keyword search included an unrelated entry');
  });
});

for (const worker of ['unavailable', 'constructor-error', 'runtime-error']) {
  test(`keywords work when Worker is ${worker} and all fetches fail`, async (t) => {
    await withEngine({ worker }, async ({ semantic, search, results, requests, workers, returns, dispose }) => {
      semantic(true);
      search('FlashRNN');
      await eventually(() => hasKeyword(results, corpus[0].id), 'Keyword results waited for the unavailable semantic backend');
      semantic(false);
      dispose();
      await settled(returns);
      t.diagnostic(`Intercepted fetches=${requests.length}; constructed workers=${workers.length}; no real network allowed`);
    });
  });
}

test('pending model/index IO does not block keywords; disabling settles public promises', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending' }, async ({ semantic, search, results, returns, statuses }) => {
    semantic(true);
    search('quantile');
    await eventually(() => hasKeyword(results, corpus[1].id), 'Keyword result was blocked by pending IO');
    semantic(false);
    await settled(returns);
    assert.equal(statuses.at(-1).enabled, false, 'Semantic mode remained enabled after cancellation');
    keywordOnly(results.at(-1));
  });
});

test('rapid replacement and empty queries cannot restore old semantic answers', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending' }, async ({ semantic, search, results, workers, returns, dispose }) => {
    semantic(true);
    search('FlashRNN');
    search('quantile');
    search('');
    semantic(false);
    const before = results.length;
    search('Vision-LSTM');
    await eventually(() => hasKeyword(results.slice(before), corpus[2].id), 'Newest query did not produce keyword results');
    const current = results.length;
    for (const worker of workers) worker.fail();
    await delay(30);
    for (const items of results.slice(current)) {
      assert.ok(items.every((item) => item.id === corpus[2].id), 'An older query replaced the newest keyword results');
      keywordOnly(items);
    }
    const empty = results.length;
    search('');
    await eventually(() => results.length > empty, 'Clearing the query did not update results');
    keywordOnly(results.at(-1));
    dispose();
    await settled(returns);
  });
});

test('empty source remains empty on keyword search', async () => {
  await withEngine({ worker: 'unavailable', entries: [] }, async ({ semantic, search, results }) => {
    semantic(false);
    search('TiRex');
    await eventually(() => results.length > 0, 'Empty corpus did not settle with results');
    assert.deepEqual(results.at(-1), []);
  });
});

test('disposal settles pending public calls and ignores later worker errors', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending' }, async ({ semantic, search, dispose, results, statuses, workers, returns }) => {
    semantic(true);
    search('FlashRNN');
    await eventually(() => hasKeyword(results, corpus[0].id), 'Keyword search did not run before disposal');
    dispose();
    await settled(returns);
    const counts = [results.length, statuses.length];
    for (const worker of workers) worker.fail();
    await delay(30);
    assert.deepEqual([results.length, statuses.length], counts, 'Disposed engine emitted late callbacks');
  });
});

const isAsset = (name) => ({ url }) => new URL(url).pathname.endsWith(`/${name}`);
const metered = { saveData: true, metered: true, type: 'cellular', effectiveType: '2g' };

for (const storage of ['absent', 'getter-throws', 'methods-throw', undefined, 'off', '', 'true', 'ON', '1', 'yes']) {
  const label = storage === undefined ? 'no stored value' : ['absent', 'getter-throws', 'methods-throw'].includes(storage)
    ? `localStorage ${storage}` : `stored value ${JSON.stringify(storage)}`;
  test(`semantic search is off by default with ${label}`, async () => {
    await withEngine({ worker: 'silent', storage }, async ({ statuses, requests, workers }) => {
      assert.ok(statuses.length > 0, 'Construction reported no initial status');
      assert.equal(statuses[0].state, 'off', 'The first status was not off');
      assert.equal(statuses[0].enabled, false, 'Semantic search was enabled without an explicit opt-in');
      await delay(25);
      assert.deepEqual(requests, [], 'Construction fetched search assets');
      assert.equal(workers.length, 0, 'Construction started a Worker');
    });
  });
}

for (const [label, connection] of [
  ['a fast connection', { saveData: false, effectiveType: '4g' }],
  ['Save-Data', { saveData: true, effectiveType: '4g' }],
  ['a cellular connection', { saveData: false, type: 'cellular', effectiveType: '4g' }],
  ['a 2g connection', { saveData: false, effectiveType: '2g' }],
  ['a metered connection', metered],
  ['no connection information', 'none'],
]) {
  test(`disabled semantic search never fetches or starts a Worker on ${label}`, async () => {
    await withEngine({ worker: 'silent', fetch: 'assets', connection }, async ({ search, results, requests, workers, statuses }) => {
      for (const query of ['FlashRNN', 'quantile', 'GPU kernels for recurrent networks', '', 'Vision-LSTM']) search(query);
      await eventually(() => hasKeyword(results, corpus[2].id), 'The newest query produced no keyword results');
      await delay(400);
      search('time series forecasting');
      await delay(400);
      assert.deepEqual(requests, [], 'Disabled semantic search fetched meta.json, vectors.bin or model files');
      assert.equal(workers.length, 0, 'Disabled semantic search constructed a Worker');
      assert.ok(statuses.length > 0);
      for (const status of statuses) {
        assert.equal(status.state, 'off', `Disabled semantic search reported state ${status.state}`);
        assert.equal(status.enabled, false);
      }
      for (const items of results) keywordOnly(items);
      assert.ok(hasKeyword(results.slice(-1), corpus[1].id), 'The last query did not return its keyword result');
    });
  });
}

test('a stored opt-in starts loading at the first non-empty query', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending', storage: 'on' }, async ({ search, requests, statuses }) => {
    assert.equal(statuses[0].state, 'idle');
    search('FlashRNN');
    await eventually(() => requests.some(isAsset('meta.json')), 'A stored opt-in did not load the index for a query');
    assert.equal(statuses.at(-1).state, 'loading');
    assert.equal(statuses.at(-1).enabled, true);
  });
});

test('setSemantic(true) persists the opt-in and loads at once with an empty query', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, search, results, requests, workers, statuses, writes, entries }) => {
    const first = statuses.length;
    semantic(true);
    assert.deepEqual(writes.at(-1), [PREFERENCE, 'on'], 'The explicit opt-in was not persisted');
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker without a query');
    const worker = workers[0];
    assert.ok(worker.fetchedBefore.some(isAsset('meta.json')), 'The Worker started before meta.json was fetched');
    assert.ok(worker.fetchedBefore.some(isAsset('vectors.bin')), 'The Worker started before vectors.bin was fetched');
    assert.equal(worker.messages[0]?.type, 'load', 'The Worker did not receive the load request');
    assert.equal(worker.messages[0].forceWasm, false);
    assert.ok(statuses.length > first, 'Enabling semantic search reported no status');
    for (const status of statuses.slice(first)) {
      assert.equal(status.state, 'loading', `Enabling semantic search reported ${status.state} before the model was ready`);
      assert.equal(status.enabled, true);
    }
    assert.ok(!requests.some(({ url }) => weightsURL(url)), 'The page fetched model weights itself');

    // A query typed during loading is answered when the Worker is ready.
    search('zzqx');
    await delay(300);
    assert.equal(statuses.at(-1).state, 'loading');
    worker.emit({ type: 'ready', backend: 'WebGPU' });
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'The query typed during loading was not sent after ready');
    const request = worker.messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, 'zzqx');
    worker.emit({ type: 'vector', id: request.id, vector: unitVector(2) });
    await eventually(() => results.at(-1)?.some(({ id, match }) => id === corpus[2].id && match === 'semantic'), 'No semantic result after ready');
    const hit = results.at(-1).find(({ id }) => id === corpus[2].id);
    assert.equal(hit.snippet, entries[2].snippet, 'A semantic result lost the snippet field');
    assert.equal(statuses.at(-1).state, 'ready');
  });
});

test('setSemantic(true) answers a query that was typed while semantic search was off', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, search, results, requests, workers, statuses, entries }) => {
    search('quantile');
    await delay(300);
    assert.deepEqual(requests, [], 'A query fetched assets while semantic search was off');
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    assert.ok(hasKeyword(results, corpus[1].id), 'Enabling semantic search removed keyword results');
    const worker = workers[0];
    worker.emit({ type: 'ready', backend: 'WASM' });
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'The current query was not sent after ready');
    const request = worker.messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, 'quantile');
    worker.emit({ type: 'vector', id: request.id, vector: unitVector(1) });
    await eventually(() => results.at(-1)?.some(({ id, match }) => id === corpus[1].id && match === 'both'), 'No combined result after ready');
    assert.equal(results.at(-1).find(({ id }) => id === corpus[1].id).snippet, entries[1].snippet);
    assert.equal(statuses.at(-1).state, 'ready');
    assert.equal(statuses.at(-1).backend, 'WASM');
  });
});

test('setSemantic(true) still loads when storage cannot persist the opt-in', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending', storage: 'methods-throw' }, async ({ semantic, requests, statuses, writes }) => {
    assert.doesNotThrow(() => semantic(true), 'A storage failure escaped setSemantic(true)');
    assert.deepEqual(writes.at(-1), [PREFERENCE, 'on'], 'The opt-in was not offered to storage');
    await eventually(() => requests.some(isAsset('meta.json')), 'A storage failure prevented loading');
    assert.equal(statuses.at(-1).state, 'loading');
    assert.equal(statuses.at(-1).enabled, true);
  });
});

test('setSemantic(false) during index download aborts it, persists off and keeps keyword results', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending', storage: 'on' }, async ({ semantic, search, results, requests, workers, statuses, writes, fetchAborts }) => {
    search('quantile');
    await eventually(() => requests.some(isAsset('meta.json')), 'The stored opt-in did not start loading');
    semantic(false);
    assert.deepEqual(writes.at(-1), [PREFERENCE, 'off'], 'Disabling was not persisted');
    assert.ok(fetchAborts.count >= 1, 'Disabling did not abort the pending index download');
    assert.equal(statuses.at(-1).state, 'off');
    assert.equal(statuses.at(-1).enabled, false);
    assert.ok(hasKeyword(results.slice(-1), corpus[1].id), 'Disabling removed the keyword results');
    keywordOnly(results.at(-1));
    const before = requests.length;
    search('FlashRNN');
    await delay(400);
    assert.equal(requests.length, before, 'Searching after disabling fetched again');
    assert.equal(workers.length, 0, 'Searching after disabling started a Worker');
    assert.equal(statuses.at(-1).state, 'off');
  });
});

test('setSemantic(false) terminates a loading Worker and ignores its later messages', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, search, results, workers, statuses, writes }) => {
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    search('FlashRNN');
    semantic(false);
    assert.deepEqual(writes.at(-1), [PREFERENCE, 'off']);
    assert.ok(workers[0].terminated, 'Disabling did not terminate the Worker');
    assert.equal(statuses.at(-1).state, 'off');
    assert.ok(hasKeyword(results.slice(-1), corpus[0].id), 'Disabling removed the keyword results');
    const counts = [results.length, statuses.length];
    workers[0].emit({ type: 'ready', backend: 'WebGPU' });
    await delay(300);
    assert.deepEqual([results.length, statuses.length], counts, 'A terminated Worker changed results or status');
    assert.equal(workers.length, 1);
  });
});

for (const start of ['stored opt-in', 'setSemantic(true)']) {
  test(`metered connections do not gate loading after ${start}`, async () => {
    await withEngine({ worker: 'silent', fetch: 'pending', connection: metered, storage: start === 'stored opt-in' ? 'on' : undefined },
      async ({ semantic, search, requests, statuses }) => {
        if (start === 'setSemantic(true)') semantic(true);
        search('FlashRNN');
        await eventually(() => requests.some(isAsset('meta.json')), 'An enabled engine did not load on a metered connection');
        assert.ok(statuses.every(({ state }) => state !== 'consent'), 'The engine reported the removed consent state');
        assert.equal(statuses.at(-1).state, 'loading');
      });
  });
}

test('keyword results keep the snippet field unchanged', async () => {
  await withEngine({ worker: 'unavailable' }, async ({ search, results, entries }) => {
    search('quantile');
    await eventually(() => hasKeyword(results, corpus[1].id), 'Keyword search did not find TiRex-2');
    const hit = results.at(-1).find(({ id }) => id === corpus[1].id);
    assert.equal(hit.snippet, entries[1].snippet, 'A keyword result lost the snippet field');
    assert.equal(hit.text, entries[1].text, 'A keyword result changed the text field');
  });
});

const assetCount = (requests, name) => requests.filter(isAsset(name)).length;
const loadMessages = (worker) => worker.messages.filter(({ type }) => type === 'load').length;
const persistedOn = (writes) => writes.some(([key, value]) => key === PREFERENCE && value === 'on');

test('setSemantic(true) on a stored opt-in that is still idle loads at once with an empty query', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets', storage: 'on' }, async ({ semantic, requests, workers, statuses, writes }) => {
    assert.equal(statuses[0].state, 'idle');
    const mark = statuses.length;
    semantic(true);
    assert.ok(persistedOn(writes), 'The explicit opt-in was not persisted');
    await eventually(() => workers.length === 1, 'An explicit opt-in on an idle engine did not start the Worker without a query');
    assert.ok(workers[0].fetchedBefore.some(isAsset('meta.json')), 'The Worker started before meta.json was fetched');
    assert.ok(workers[0].fetchedBefore.some(isAsset('vectors.bin')), 'The Worker started before vectors.bin was fetched');
    assert.equal(workers[0].messages[0]?.type, 'load');
    assert.ok(statuses.length > mark, 'The explicit opt-in reported no status');
    for (const status of statuses.slice(mark)) {
      assert.equal(status.state, 'loading', `An explicit opt-in on an idle engine reported ${status.state}`);
      assert.equal(status.enabled, true);
    }
  });
});

test('setSemantic(true) while the index download is pending does not restart it', async () => {
  await withEngine({ worker: 'silent', fetch: 'pending' }, async ({ semantic, requests, workers, statuses, writes, fetchAborts }) => {
    semantic(true);
    await eventually(() => requests.some(isAsset('meta.json')), 'Enabling semantic search did not fetch meta.json');
    await delay(50);
    const counts = [requests.length, assetCount(requests, 'meta.json'), assetCount(requests, 'vectors.bin')];
    const mark = statuses.length;
    const written = writes.length;
    semantic(true);
    assert.ok(persistedOn(writes.slice(written)), 'The repeated opt-in was not persisted');
    await delay(300);
    assert.deepEqual([requests.length, assetCount(requests, 'meta.json'), assetCount(requests, 'vectors.bin')], counts,
      'Re-enabling during the index download fetched search assets again');
    assert.equal(fetchAborts.count, 0, 'Re-enabling aborted the healthy index download');
    assert.equal(workers.length, 0, 'Re-enabling during the index download started a Worker');
    for (const status of statuses.slice(mark)) {
      assert.equal(status.state, 'loading', `Re-enabling during the index download reported ${status.state}`);
      assert.equal(status.enabled, true);
    }
    assert.equal(statuses.at(-1).state, 'loading');
    assert.equal(statuses.at(-1).enabled, true);
  });
});

test('setSemantic(true) while the Worker loads the model keeps that Worker and answers after ready', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, search, results, requests, workers, statuses, writes }) => {
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    const worker = workers[0];
    search('zzqx');
    await delay(300);
    const fetched = [assetCount(requests, 'meta.json'), assetCount(requests, 'vectors.bin')];
    assert.deepEqual(fetched, [1, 1]);
    const mark = statuses.length;
    const written = writes.length;
    semantic(true);
    assert.ok(persistedOn(writes.slice(written)), 'The repeated opt-in was not persisted');
    await delay(300);
    assert.equal(workers.length, 1, 'Re-enabling during model load constructed a new Worker');
    assert.equal(worker.terminated, false, 'Re-enabling during model load terminated the loading Worker');
    assert.equal(loadMessages(worker), 1, 'Re-enabling during model load sent another load message');
    assert.deepEqual([assetCount(requests, 'meta.json'), assetCount(requests, 'vectors.bin')], fetched,
      'Re-enabling during model load fetched search assets again');
    for (const status of statuses.slice(mark)) {
      assert.equal(status.state, 'loading', `Re-enabling during model load reported ${status.state}`);
      assert.equal(status.enabled, true);
    }
    assert.equal(statuses.at(-1).state, 'loading');
    worker.emit({ type: 'ready', backend: 'WebGPU' });
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'The query typed during loading was not sent after ready');
    const request = worker.messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, 'zzqx');
    worker.emit({ type: 'vector', id: request.id, vector: unitVector(0) });
    await eventually(() => results.at(-1)?.some(({ id, match }) => id === corpus[0].id && match === 'semantic'), 'No semantic result after ready');
    assert.equal(statuses.at(-1).state, 'ready');
    assert.equal(workers.length, 1);
  });
});

test('setSemantic(true) on a ready engine with an empty query keeps it ready without reloading', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, requests, workers, statuses, writes }) => {
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    const worker = workers[0];
    worker.emit({ type: 'ready', backend: 'WASM' });
    await eventually(() => statuses.at(-1)?.state === 'ready', 'The Worker ready message did not make the engine ready');
    const fetched = requests.length;
    const mark = statuses.length;
    const written = writes.length;
    semantic(true);
    assert.ok(persistedOn(writes.slice(written)), 'The repeated opt-in was not persisted');
    await delay(300);
    assert.equal(requests.length, fetched, 'Re-enabling a ready engine fetched search assets again');
    assert.equal(workers.length, 1, 'Re-enabling a ready engine constructed a new Worker');
    assert.equal(worker.terminated, false, 'Re-enabling a ready engine terminated its Worker');
    assert.equal(loadMessages(worker), 1, 'Re-enabling a ready engine sent another load message');
    for (const status of statuses.slice(mark)) {
      assert.equal(status.state, 'ready', `Re-enabling a ready engine reported ${status.state}`);
      assert.equal(status.enabled, true);
    }
    assert.equal(statuses.at(-1).state, 'ready');
    assert.equal(statuses.at(-1).backend, 'WASM');
  });
});

test('setSemantic(true) on a ready engine with a query answers it on the same Worker', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, search, results, requests, workers, statuses }) => {
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    const worker = workers[0];
    worker.emit({ type: 'ready', backend: 'WebGPU' });
    await eventually(() => statuses.at(-1)?.state === 'ready', 'The Worker ready message did not make the engine ready');
    search('zzqx');
    await eventually(() => worker.messages.some(({ type }) => type === 'embed'), 'A query on a ready engine was not sent');
    const first = worker.messages.filter(({ type }) => type === 'embed').at(-1);
    worker.emit({ type: 'vector', id: first.id, vector: unitVector(1) });
    await eventually(() => results.at(-1)?.some(({ id, match }) => id === corpus[1].id && match === 'semantic'), 'No semantic result before re-enabling');
    const fetched = requests.length;
    const mark = statuses.length;
    const resultMark = results.length;
    semantic(true);
    await delay(300);
    assert.equal(requests.length, fetched, 'Re-enabling a ready engine fetched search assets again');
    assert.equal(workers.length, 1, 'Re-enabling a ready engine constructed a new Worker');
    assert.equal(worker.terminated, false, 'Re-enabling a ready engine terminated its Worker');
    assert.equal(loadMessages(worker), 1, 'Re-enabling a ready engine sent another load message');
    for (const status of statuses.slice(mark)) {
      assert.ok(['ready', 'searching'].includes(status.state), `Re-enabling a ready engine reported ${status.state}`);
      assert.equal(status.enabled, true);
    }
    // When the engine answers the current query again, the answer comes from the same Worker.
    const embeds = worker.messages.filter(({ type }) => type === 'embed');
    if (embeds.length > 1) {
      const again = embeds.at(-1);
      assert.equal(again.text, 'zzqx');
      worker.emit({ type: 'vector', id: again.id, vector: unitVector(1) });
      await eventually(() => results.slice(resultMark).some((items) => items.some(({ id, match }) => id === corpus[1].id && match === 'semantic')),
        'The re-sent query produced no semantic result');
      assert.equal(statuses.at(-1).state, 'ready');
    }
  });
});

test('setSemantic(true) after a failed index download retries the download and becomes ready', async () => {
  await withEngine({ worker: 'silent', fetch: 'fail-once' }, async ({ semantic, search, results, requests, workers, statuses, writes }) => {
    semantic(true);
    await eventually(() => statuses.at(-1)?.state === 'error', 'A failed index download did not report the error state');
    assert.equal(statuses.at(-1).enabled, true);
    assert.equal(workers.length, 0, 'A failed index download started a Worker');
    search('zzqx');
    const before = assetCount(requests, 'meta.json');
    const mark = statuses.length;
    const written = writes.length;
    semantic(true);
    assert.ok(persistedOn(writes.slice(written)), 'The retry did not persist the opt-in');
    await eventually(() => workers.length === 1, 'Retry after a failed index download did not start the Worker');
    assert.ok(assetCount(requests, 'meta.json') > before, 'Retry after a failed index download did not fetch meta.json again');
    assert.ok(workers[0].fetchedBefore.some(isAsset('vectors.bin')), 'The retry Worker started before vectors.bin was fetched');
    assert.equal(workers[0].messages[0]?.type, 'load');
    assert.ok(statuses.slice(mark).some(({ state }) => state === 'loading'), 'Retry did not report the loading state');
    assert.equal(statuses.at(-1).state, 'loading');
    assert.equal(statuses.at(-1).enabled, true);
    workers[0].emit({ type: 'ready', backend: 'WebGPU' });
    await eventually(() => workers[0].messages.some(({ type }) => type === 'embed'), 'Retry did not answer the current query after ready');
    const request = workers[0].messages.find(({ type }) => type === 'embed');
    assert.equal(request.text, 'zzqx');
    workers[0].emit({ type: 'vector', id: request.id, vector: unitVector(2) });
    await eventually(() => results.at(-1)?.some(({ id, match }) => id === corpus[2].id && match === 'semantic'), 'No semantic result after retry');
    assert.equal(statuses.at(-1).state, 'ready');
  });
});

test('setSemantic(true) after a Worker failure with an empty query starts a fresh Worker at once', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, workers, statuses }) => {
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    const first = workers[0];
    first.fail();
    await eventually(() => statuses.at(-1)?.state === 'error', 'A Worker failure did not report the error state');
    const mark = statuses.length;
    semantic(true);
    await eventually(() => workers.length === 2, 'Retry with an empty query did not start a fresh Worker');
    const retry = workers[1];
    assert.notEqual(retry, first);
    assert.equal(retry.messages[0]?.type, 'load', 'The retry Worker did not receive the load request');
    assert.ok(statuses.slice(mark).some(({ state }) => state === 'loading'), 'Retry did not report the loading state');
    assert.equal(statuses.at(-1).state, 'loading');
    assert.equal(statuses.at(-1).enabled, true);
    retry.emit({ type: 'ready', backend: 'WASM' });
    await eventually(() => statuses.at(-1)?.state === 'ready', 'The retry Worker did not make the engine ready');
  });
});

test('setSemantic(true) after setSemantic(false) on a ready engine loads a fresh Worker', async () => {
  await withEngine({ worker: 'silent', fetch: 'assets' }, async ({ semantic, workers, statuses, writes }) => {
    semantic(true);
    await eventually(() => workers.length === 1, 'Enabling semantic search did not start the Worker');
    workers[0].emit({ type: 'ready', backend: 'WebGPU' });
    await eventually(() => statuses.at(-1)?.state === 'ready', 'The Worker ready message did not make the engine ready');
    semantic(false);
    assert.ok(workers[0].terminated);
    assert.equal(statuses.at(-1).state, 'off');
    const mark = statuses.length;
    semantic(true);
    assert.deepEqual(writes.at(-1), [PREFERENCE, 'on']);
    await eventually(() => workers.length === 2, 'Enabling again after disabling did not start a fresh Worker');
    assert.equal(workers[1].messages[0]?.type, 'load');
    for (const status of statuses.slice(mark)) {
      assert.equal(status.state, 'loading', `Enabling again after disabling reported ${status.state} before ready`);
      assert.equal(status.enabled, true);
    }
    workers[1].emit({ type: 'ready', backend: 'WebGPU' });
    await eventually(() => statuses.at(-1)?.state === 'ready', 'The fresh Worker did not make the engine ready');
  });
});
