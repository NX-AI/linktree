import test from 'node:test';
import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { setTimeout as delay } from 'node:timers/promises';

// These constants preserve the existing model contract, independently of core.js.
const model = Object.freeze({
  model: 'onnx-community/embeddinggemma-2-ONNX',
  revision: 'daa72c51243991dfcaf9f9137d2c573d8f7790c0',
  dtype: 'q8',
  dim: 512,
  query_prefix: 'task: search result | query: ',
});
const entries = [
  { id: 'kernels', title: 'FlashRNN', text: 'Recurrent network kernels.', group: 'compute', url: 'https://example.invalid/kernels' },
  { id: 'forecast', title: 'Forecasting', text: 'Time series prediction.', group: 'models', url: 'https://example.invalid/forecast' },
];
const meta = { version: 1, ...model, entries };
let nonce = 0;

async function eventually(predicate, message, milliseconds = 1500) {
  const end = performance.now() + milliseconds;
  do {
    if (predicate()) return;
    await delay(5);
  } while (performance.now() < end);
  assert.fail(message);
}

function installGlobals(values) {
  const saved = new Map();
  for (const [name, value] of Object.entries(values)) {
    saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
  }
  return () => {
    for (const [name, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  };
}

// Event-handler properties participate in dispatch, rather than running twice.
class Scope extends EventTarget {
  handlers = new Map();
  pending = new Set();
  exceptions = [];
  wrapped = new Map();

  addEventListener(type, listener, options) {
    if (!listener) return;
    if (!this.wrapped.has(listener)) this.wrapped.set(listener, (event) => {
      try {
        const result = typeof listener === 'function' ? listener.call(this, event) : listener.handleEvent(event);
        if (result && typeof result.then === 'function') {
          const observed = Promise.resolve(result).then(
            () => this.pending.delete(observed),
            (error) => { this.pending.delete(observed); this.exceptions.push(String(error)); },
          );
          this.pending.add(observed);
        }
      } catch (error) { this.exceptions.push(String(error)); }
    });
    super.addEventListener(type, this.wrapped.get(listener), options);
  }

  removeEventListener(type, listener, options) {
    super.removeEventListener(type, this.wrapped.get(listener) ?? listener, options);
  }

  handler(type, value) {
    if (this.handlers.has(type)) this.removeEventListener(type, this.handlers.get(type));
    this.handlers.set(type, value);
    if (value) this.addEventListener(type, value);
  }

  set onmessage(value) { this.handler('message', value); }
  set onerror(value) { this.handler('error', value); }
  set onmessageerror(value) { this.handler('messageerror', value); }
  set onunhandledrejection(value) { this.handler('unhandledrejection', value); }

  emit(type, properties = {}) {
    const event = new Event(type, { cancelable: true });
    for (const [key, value] of Object.entries(properties)) Object.defineProperty(event, key, { value });
    this.dispatchEvent(event);
    return event;
  }
}

async function withWorker(options, body) {
  const scope = new Scope();
  scope.messages = [];
  scope.closed = false;
  scope.close = () => { scope.closed = true; };
  scope.postMessage = (data) => scope.messages.push(structuredClone(data));
  let adapters = 0;
  scope.navigator = options.gpu ? { gpu: { async requestAdapter() { adapters++; return {}; } } } : {};
  const calls = { config: [], pipeline: [], embed: [], fetch: [] };
  const runtime = {
    env: {
      backends: { onnx: { wasm: {} } },
      async fetch(input) { calls.fetch.push(String(input)); return new Response('{}'); },
    },
    async config(...args) {
      calls.config.push(structuredClone(args));
      if (options.configError) throw new TypeError('Model configuration download failed');
      return { vision_config: {}, audio_config: {} };
    },
    async pipeline(...args) {
      const [task, name, settings] = args;
      calls.pipeline.push({ task, name, ...settings });
      if (options.pipelineError) throw new TypeError('Model download failed');
      if (options.pending) return new Promise(() => {});
      return async (text, settings) => {
        calls.embed.push({ text, settings: structuredClone(settings) });
        return { dims: [1, model.dim], data: Float32Array.from({ length: model.dim }, (_, i) => i === 0 ? 1 : 0) };
      };
    },
  };
  const key = `__nxaiFailureRuntime${++nonce}`;
  const runtimeURL = new URL('../runtime/transformers.min.js', import.meta.url).href;
  const mockURL = `${runtimeURL}?contract=${nonce}`;
  const hooks = registerHooks({
    resolve(specifier, context, next) {
      if (specifier === runtimeURL) return { url: mockURL, shortCircuit: true };
      return next(specifier, context);
    },
    load(url, context, next) {
      if (url !== mockURL) return next(url, context);
      return {
        format: 'module', shortCircuit: true,
        source: `const fixture = globalThis[${JSON.stringify(key)}];
          export const env = fixture.env;
          export const AutoConfig = { from_pretrained: (...args) => fixture.config(...args) };
          export const pipeline = (...args) => fixture.pipeline(...args);`,
      };
    },
  });
  const restore = installGlobals({
    [key]: runtime, self: scope, navigator: scope.navigator,
    addEventListener: scope.addEventListener.bind(scope), removeEventListener: scope.removeEventListener.bind(scope),
    postMessage: scope.postMessage, close: scope.close,
    fetch: async () => { throw new Error('Real network is forbidden in this test'); },
  });
  try {
    const url = new URL('../worker.js', import.meta.url);
    url.searchParams.set('failure-contract', String(nonce));
    await import(url.href);
    await body({ scope, calls, runtime, adapters: () => adapters, load: (forceWasm = false) => scope.emit('message', { data: { type: 'load', meta: structuredClone(meta), forceWasm } }) });
    assert.deepEqual(scope.exceptions, [], 'A Worker event handler escaped without handling its failure');
  } finally {
    hooks.deregister();
    restore();
  }
}

test('model identity and document/query text contracts remain unchanged', async () => {
  const core = await import('../core.js');
  assert.deepEqual(core.MODEL, model);
  assert.deepEqual(core.embeddingTexts(entries), entries.map((entry) => `title: ${entry.title} | text: ${entry.text}`));
});

for (const gpu of [false, true]) {
  test(`healthy ${gpu ? 'WebGPU' : 'WASM'} load and embed protocol remains usable`, async () => {
    await withWorker({ gpu }, async ({ scope, calls, runtime, load }) => {
      load();
      await eventually(() => scope.messages.some(({ type }) => type === 'ready'), 'Healthy Worker never became ready');
      assert.deepEqual(scope.messages.filter(({ type }) => type === 'ready'), [{ type: 'ready', backend: gpu ? 'WebGPU' : 'WASM' }]);
      const text = 'FlashRNN — recurrent kernels?';
      scope.emit('message', { data: { type: 'embed', id: 37, text } });
      await eventually(() => scope.messages.some(({ type }) => type === 'vector'), 'Healthy embed did not finish');
      assert.deepEqual(calls.embed, [{ text: model.query_prefix + text, settings: { pooling: 'mean', normalize: true } }]);
      assert.deepEqual(calls.config, [[model.model, { revision: model.revision }]]);
      assert.equal(calls.pipeline.length, 1);
      assert.equal(calls.pipeline[0].task, 'feature-extraction');
      assert.equal(calls.pipeline[0].name, model.model);
      assert.equal(calls.pipeline[0].dtype, model.dtype);
      assert.equal(calls.pipeline[0].revision, model.revision);
      assert.equal(calls.pipeline[0].device, gpu ? 'webgpu' : 'wasm');
      const vector = scope.messages.filter(({ type }) => type === 'vector');
      assert.equal(vector.length, 1);
      assert.equal(vector[0].id, 37);
      assert.equal(vector[0].vector.length, model.dim);
      assert.ok(Array.from(vector[0].vector).every(Number.isFinite));
      assert.equal(scope.messages.filter(({ type }) => type === 'error' || type === 'fallback').length, 0);
      assert.equal(scope.closed, false, 'A healthy Worker was shut down');
      await runtime.env.fetch(`https://huggingface.co/${model.model}/resolve/main/config.json`);
      assert.deepEqual(calls.fetch, [`https://huggingface.co/${model.model}/resolve/${model.revision}/config.json`]);
    });
  });
}

test('ordinary configuration failure emits one terminal error and settles loading', async () => {
  await withWorker({ configError: true }, async ({ scope, load }) => {
    load();
    await eventually(() => scope.messages.some(({ type }) => type === 'error'), 'Configuration failure did not report an error');
    await eventually(() => scope.closed || scope.pending.size === 0, 'Configuration failure left loading pending');
    assert.equal(scope.messages.filter(({ type }) => type === 'error').length, 1);
    assert.equal(scope.messages.filter(({ type }) => type === 'ready' || type === 'vector').length, 0);
  });
});

for (const gpu of [false, true]) {
  test(`fatal detached download rejection ends pending ${gpu ? 'WebGPU' : 'WASM'} loading once`, async () => {
    await withWorker({ gpu, pending: true }, async ({ scope, calls, load }) => {
      load();
      await eventually(() => calls.pipeline.length === 1, 'Failure injection never reached the pending model loader');
      const reason = new TypeError('Model download failed (detached dependency rejection)');
      const event = scope.emit('unhandledrejection', { reason, promise: Promise.resolve() });
      // emit() returns only after every registered production listener has run.
      assert.equal(event.defaultPrevented, true, 'Fatal rejection retained its unhandled default action');
      await eventually(() => scope.messages.some(({ type }) => type === 'error'), 'Fatal rejection was swallowed instead of reporting terminal failure');
      await eventually(() => scope.closed || scope.pending.size === 0, 'Fatal rejection left the model loader permanently pending');
      if (!scope.closed) {
        const duplicate = scope.emit('unhandledrejection', { reason, promise: Promise.resolve() });
        assert.equal(duplicate.defaultPrevented, true, 'A second fatal rejection escaped the terminal handler');
      }
      await delay(20);
      const terminal = scope.messages.filter(({ type }) => ['error', 'fallback', 'ready'].includes(type));
      assert.equal(terminal.length, 1, 'One failed load emitted duplicate or conflicting terminal events');
      assert.equal(terminal[0].type, 'error', 'A model download failure did not terminate with an error');
      assert.ok(terminal[0].message, 'Terminal error omitted its diagnostic');
    });
  });
}

test('WebGPU failure still permits a fresh Worker forced to WASM', async () => {
  await withWorker({ gpu: true, pipelineError: true }, async ({ scope, load }) => {
    load();
    await eventually(() => scope.messages.some(({ type }) => type === 'fallback'), 'WebGPU load failure did not request WASM fallback');
    assert.equal(scope.messages.filter(({ type }) => ['fallback', 'ready', 'error'].includes(type)).length, 1);
  });
  await withWorker({ gpu: true }, async ({ scope, calls, adapters, load }) => {
    load(true);
    await eventually(() => scope.messages.some(({ type }) => type === 'ready'), 'Fresh forced-WASM Worker never became ready');
    assert.equal(adapters(), 0, 'Forced WASM retried WebGPU');
    assert.equal(calls.pipeline[0].device, 'wasm');
    assert.equal(scope.messages.find(({ type }) => type === 'ready').backend, 'WASM');
  });
});

async function withEngine(body) {
  const workers = [];
  const statuses = [];
  const results = [];
  const timers = new Map();
  const nativeSetTimeout = globalThis.setTimeout;
  const nativeClearTimeout = globalThis.clearTimeout;
  class Worker extends Scope {
    constructor(url, options) {
      super();
      this.url = String(url);
      this.options = options;
      this.messages = [];
      this.terminated = false;
      workers.push(this);
    }
    postMessage(data) { this.messages.push(structuredClone(data)); }
    terminate() { this.terminated = true; }
  }
  const buffer = new ArrayBuffer(entries.length * (4 + model.dim));
  new Float32Array(buffer, 0, entries.length).fill(1);
  const codes = new Int8Array(buffer, entries.length * 4);
  entries.forEach((_, i) => { codes[i * model.dim + i] = 1; });
  const restore = installGlobals({
    Worker, localStorage: { getItem: () => 'on', setItem() {} },
    navigator: { connection: { effectiveType: '4g', saveData: false } },
    fetch: async (input) => {
      const url = new URL(String(input));
      if (url.pathname.endsWith('/meta.json')) return { ok: true, json: async () => structuredClone(meta) };
      if (url.pathname.endsWith('/vectors.bin')) return { ok: true, arrayBuffer: async () => buffer.slice(0) };
      throw new Error(`Real network is forbidden: ${url}`);
    },
    setTimeout(callback, milliseconds, ...args) {
      const handle = nativeSetTimeout(() => { timers.delete(handle); callback(...args); }, milliseconds);
      timers.set(handle, milliseconds);
      return handle;
    },
    clearTimeout(handle) { timers.delete(handle); nativeClearTimeout(handle); },
  });
  let engine;
  try {
    const url = new URL('../engine.js', import.meta.url);
    url.searchParams.set('failure-contract', String(++nonce));
    const { createSearch } = await import(url.href);
    engine = createSearch(structuredClone(entries), {
      onStatus: (value) => statuses.push(structuredClone(value)),
      onResults: (value) => results.push(structuredClone(value)),
    });
    await body({ engine, workers, statuses, results, timers });
    for (const worker of workers) assert.deepEqual(worker.exceptions, [], 'Page-side Worker callback threw');
  } finally {
    engine?.dispose();
    for (const timer of timers.keys()) nativeClearTimeout(timer);
    restore();
  }
}

function keywordPresent(results, id) {
  return results.at(-1)?.some((entry) => entry.id === id && ['keyword', 'both'].includes(entry.match));
}

for (const failureType of ['message', 'error']) {
  test(`public API exits loading promptly on Worker ${failureType} failure and allows explicit retry`, async () => {
    await withEngine(async ({ engine, workers, statuses, results, timers }) => {
      engine.search('FlashRNN');
      assert.ok(keywordPresent(results, 'kernels'), 'Keywords waited for the model');
      await eventually(() => workers.length === 1, 'The engine never started a Worker');
      const first = workers[0];
      assert.deepEqual(first.messages[0], { type: 'load', meta, forceWasm: false });
      if (failureType === 'message') first.emit('message', { data: { type: 'error', message: 'Model download failed' } });
      else {
        first.emit('message', { data: { type: 'ready', backend: 'WebGPU' } });
        assert.equal(statuses.at(-1).state, 'searching');
        assert.equal(first.messages.find(({ type }) => type === 'embed').text, 'FlashRNN');
        const event = first.emit('error', { message: 'Model download failed', error: new TypeError('Model download failed') });
        assert.equal(event.defaultPrevented, true, 'Worker error retained its uncaught default action');
      }
      await eventually(() => statuses.at(-1)?.state === 'error', 'Public status did not promptly leave loading', 1000);
      assert.ok(first.terminated, 'Failed Worker was not terminated');
      assert.match(statuses.at(-1).message, /keyword/i);
      assert.ok(keywordPresent(results, 'kernels'), 'Terminal failure removed keyword results');
      assert.equal([...timers.values()].filter((milliseconds) => milliseconds >= 10000).length, 0, 'Failure left a model/search watchdog pending');
      first.emit('message', { data: { type: 'error', message: 'Second rejection from failed work' } });
      first.emit('message', { data: { type: 'ready', backend: 'WASM' } });
      assert.equal(statuses.filter(({ state }) => state === 'error').length, 1, 'Failed work produced duplicate terminal status');
      engine.search('Forecasting');
      assert.ok(keywordPresent(results, 'forecast'), 'Keywords stopped working after semantic failure');
      await delay(300);
      assert.equal(workers.length, 1, 'Typing retried a fatal model failure without explicit user action');
      engine.setSemantic(true);
      await eventually(() => workers.length === 2, 'Explicit retry could not start fresh work');
      const retry = workers[1];
      assert.notEqual(retry, first);
      retry.emit('message', { data: { type: 'ready', backend: 'WASM' } });
      const request = retry.messages.find(({ type }) => type === 'embed');
      assert.ok(request, 'Retry did not resume the current query');
      assert.equal(request.text, 'Forecasting');
      const vector = new Float32Array(model.dim);
      vector[1] = 1;
      retry.emit('message', { data: { type: 'vector', id: request.id, vector } });
      assert.equal(statuses.at(-1).state, 'ready');
      assert.ok(keywordPresent(results, 'forecast'));
    });
  });
}

test('page protocol replaces WebGPU Worker for WASM and ignores retired Worker messages', async () => {
  await withEngine(async ({ engine, workers, statuses }) => {
    engine.search('FlashRNN');
    await eventually(() => workers.length === 1, 'No initial Worker');
    const first = workers[0];
    first.emit('message', { data: { type: 'fallback' } });
    assert.equal(workers.length, 2, 'WASM reused the WebGPU Worker');
    assert.ok(first.terminated);
    const second = workers[1];
    assert.deepEqual(second.messages[0], { type: 'load', meta, forceWasm: true });
    const before = statuses.length;
    first.emit('message', { data: { type: 'error', message: 'Retired WebGPU work' } });
    assert.equal(statuses.length, before, 'Retired Worker changed public status');
    second.emit('message', { data: { type: 'ready', backend: 'WASM' } });
    const request = second.messages.find(({ type }) => type === 'embed');
    assert.ok(request);
    assert.equal(request.text, 'FlashRNN');
    const vector = new Float32Array(model.dim);
    vector[0] = 1;
    second.emit('message', { data: { type: 'vector', id: request.id, vector } });
    assert.equal(statuses.at(-1).state, 'ready');
    assert.equal(statuses.at(-1).backend, 'WASM');
    assert.equal(statuses.filter(({ state }) => state === 'error').length, 0);
  });
});
