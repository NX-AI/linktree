import test from 'node:test';
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';

// Corpus facts come from projects.yml, not from the search implementation.
const corpus = [
  {
    id: 'https://github.com/NX-AI/flashrnn',
    title: 'FlashRNN',
    text: 'FlashRNN CUDA and Triton kernels for LSTM, GRU, Elman networks and sLSTM, optimised for modern GPU hardware. ICLR 2025',
    group: 'efficiency',
    url: 'https://github.com/NX-AI/flashrnn',
  },
  {
    id: 'https://github.com/NX-AI/tirex-2',
    title: 'TiRex-2',
    text: 'TiRex-2 Pretrained multivariate time series forecasting model with past and future-known covariates and quantile forecasts, released for zero-shot use. NeurIPS 2026',
    group: 'time-series',
    url: 'https://github.com/NX-AI/tirex-2',
  },
  {
    id: 'https://github.com/NX-AI/vision-lstm',
    title: 'Vision-LSTM',
    text: 'Vision-LSTM (ViL) adapts xLSTM as a generic backbone for computer vision. Includes pretrained models and a full training pipeline. ICLR 2025',
    group: 'xlstm',
    url: 'https://github.com/NX-AI/vision-lstm',
  },
];
const states = new Set(['idle', 'consent', 'loading', 'searching', 'ready', 'off', 'error']);
const entryFields = ['id', 'title', 'text', 'group', 'url'];
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

async function withEngine(options, body) {
  const saved = new Map();
  const replace = (key, value) => {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  };
  const requests = [];
  const workers = [];
  const pendingFetches = [];
  const returns = [];
  const results = [];
  const statuses = [];
  const entries = structuredClone(options.entries ?? corpus);
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
      workers.push(this);
    }
    postMessage(message) {
      this.messages.push(structuredClone(message));
      if (options.worker === 'runtime-error') queueMicrotask(() => this.fail());
    }
    terminate() { this.terminated = true; }
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
  replace('navigator', { connection: { saveData: false, effectiveType: '4g' }, hardwareConcurrency: 2 });
  replace('fetch', (input, init = {}) => {
    requests.push({ url: String(input?.url ?? input), method: init.method ?? input?.method ?? 'GET' });
    if (options.fetch === 'pending') {
      return new Promise((resolve, reject) => {
        const abort = () => reject(new DOMException('Fetch cancelled in test', 'AbortError'));
        pendingFetches.push(abort);
        const signal = init.signal ?? input?.signal;
        if (signal?.aborted) abort();
        else signal?.addEventListener('abort', abort, { once: true });
      });
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
          for (const field of entryFields) assert.equal(item[field], source[field], `Result changed source field ${field}`);
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
        assert.ok(states.has(value.state), `Invalid status state: ${value.state}`);
        assert.equal(typeof value.enabled, 'boolean');
        assert.equal(typeof value.message, 'string');
        statuses.push(structuredClone(value));
      },
    });
    for (const name of ['search', 'setSemantic', 'dispose']) assert.equal(typeof engine[name], 'function', `Missing method ${name}`);
    await body({
      entries, results, statuses, requests, workers, returns,
      search: (query) => observeReturn(engine.search(query)),
      semantic: (enabled) => observeReturn(engine.setSemantic(enabled)),
      dispose: () => { const value = observeReturn(engine.dispose()); engine = null; return value; },
    });
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
  await withEngine({ worker: 'silent' }, async ({ requests }) => {
    await delay(25);
    const modelRequests = requests.filter(({ url }) => weightsURL(url));
    assert.deepEqual(modelRequests, [], 'The heavy model was fetched before the first search');
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
