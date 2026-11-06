import { MODEL, combine, createKeywordIndex, createVectorIndex, keywordSearch, semanticSearch } from './core.js';

const PREFERENCE = 'nxai-semantic-search';

/** Browser-only search. Keyword results never depend on the worker or semantic assets. */
export function createSearch(entries, { onResults, onStatus }) {
  const live = entries.map((entry) => ({ ...entry }));
  const keywords = createKeywordIndex(live);
  let enabled = false;
  try { enabled = globalThis.localStorage.getItem(PREFERENCE) === 'on'; } catch { /* Storage is optional. */ }
  let query = '';
  let sequence = 0;
  let desiredId = null;
  let sentId = null;
  let epoch = 0;
  let worker = null;
  let workerReady = false;
  let loading = false;
  let forceWasm = false;
  let vectorIndex = null;
  let metadata = null;
  let controller = null;
  let debounce = null;
  let deadline = null;
  let disposed = false;
  let failure = '';
  let status = null;

  function report(state, changes = {}) {
    if (disposed) return;
    const next = { state, enabled, loaded: 0, total: 0, backend: '', message: '', ...status, ...changes };
    next.state = state;
    next.enabled = enabled;
    if (!status || Object.keys(next).some((key) => next[key] !== status[key])) {
      status = next;
      onStatus({ ...next });
    }
  }

  function clearDeadline() {
    clearTimeout(deadline);
    deadline = null;
  }

  function stop() {
    epoch++;
    clearTimeout(debounce);
    debounce = null;
    clearDeadline();
    controller?.abort();
    controller = null;
    worker?.terminate();
    worker = null;
    workerReady = false;
    loading = false;
    sentId = desiredId = null;
  }

  function fail(error) {
    if (disposed || !enabled) return;
    failure = error instanceof Error ? error.message : String(error);
    stop();
    report('error', { message: `${failure} Keyword search is still available. Enable semantic search to retry.` });
  }

  function watch(milliseconds) {
    clearDeadline();
    deadline = setTimeout(() => fail(new Error('Semantic search timed out.')), milliseconds);
  }

  function sendQuery() {
    if (disposed || !enabled || !query || desiredId !== sequence || !workerReady || sentId === sequence) return;
    sentId = sequence;
    report('searching', { message: 'Searching on this device.' });
    watch(120000);
    try { worker.postMessage({ type: 'embed', id: sequence, text: query }); }
    catch (error) { fail(error); }
  }

  function startWorker(wasmOnly) {
    worker?.terminate();
    workerReady = false;
    sentId = null;
    forceWasm = wasmOnly;
    loading = true;
    watch(180000);
    const current = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    worker = current;
    current.onerror = (event) => {
      event.preventDefault();
      if (worker === current) fail(new Error(event.message || 'The semantic search worker failed.'));
    };
    current.onmessageerror = () => {
      if (worker === current) fail(new Error('The semantic search worker sent an unreadable response.'));
    };
    current.onmessage = ({ data }) => {
      if (disposed || !enabled || worker !== current) return;
      try {
        if (data.type === 'progress') {
          if (!loading || !Number.isFinite(data.loaded) || !Number.isFinite(data.total) || data.loaded < 0 || data.total < 0) return;
          if (data.loaded > status.loaded) watch(180000);
          report('loading', { loaded: data.loaded, total: data.total, message: 'Loading the text search model on this device.' });
        } else if (data.type === 'fallback') {
          if (forceWasm) throw new Error('The WASM search backend failed.');
          report('loading', { backend: 'WASM', loaded: 0, total: 0, message: 'WebGPU is unavailable. Loading the WASM backend.' });
          startWorker(true);
        } else if (data.type === 'ready') {
          if (data.backend !== 'WebGPU' && data.backend !== 'WASM') throw new Error('Unknown semantic search backend.');
          loading = false;
          workerReady = true;
          clearDeadline();
          report('ready', { backend: data.backend, message: 'Semantic search is ready on this device.' });
          sendQuery();
        } else if (data.type === 'vector') {
          if (data.id === sentId) { sentId = null; clearDeadline(); }
          if (data.id !== sequence || desiredId !== sequence || !query) return;
          const semantic = semanticSearch(vectorIndex, data.vector);
          const keyword = keywordSearch(keywords, query, { requireAll: semantic.length > 0 });
          // No entry has every word: rank by meaning alone so generic words cannot outrank it,
          // and keep any-word matches only as labels.
          const labels = keyword.length ? [] : keywordSearch(keywords, query);
          onResults(combine(live, keyword, semantic, labels));
          report('ready', { message: 'Semantic search is ready on this device.' });
        } else if (data.type === 'error' && (data.id === undefined || data.id === sequence)) {
          throw new Error(data.message || 'Semantic search failed.');
        }
      } catch (error) { fail(error); }
    };
    current.postMessage({ type: 'load', meta: metadata, forceWasm: wasmOnly });
  }

  async function load() {
    const ticket = epoch;
    loading = true;
    controller = new AbortController();
    const signal = controller.signal;
    report('loading', { loaded: 0, total: 0, backend: '', message: 'Loading the semantic search index.' });
    watch(180000);
    try {
      if (!vectorIndex) {
        const metaResponse = await fetch(new URL('./meta.json', import.meta.url), { signal, cache: 'no-cache' });
        if (!metaResponse.ok) throw new Error(`Search metadata returned HTTP ${metaResponse.status}.`);
        const meta = await metaResponse.json();
        const response = await fetch(new URL('./vectors.bin', import.meta.url), { signal, cache: 'no-cache' });
        if (!response.ok) throw new Error(`Search vectors returned HTTP ${response.status}.`);
        const buffer = await response.arrayBuffer();
        if (ticket !== epoch || disposed || !enabled) return;
        vectorIndex = createVectorIndex(live, meta, buffer);
        metadata = meta;
      }
      if (ticket !== epoch || disposed || !enabled) return;
      controller = null;
      startWorker(false);
    } catch (error) {
      if (ticket === epoch) fail(error);
    }
  }

  function begin(id) {
    if (disposed || !enabled || failure || !query || id !== sequence) return;
    desiredId = id;
    if (workerReady) sendQuery();
    else if (!loading) void load();
  }

  function emitKeywords(reset = false) {
    if (reset) onResults([]);
    if (query) onResults(combine(live, keywordSearch(keywords, query)));
    else if (!reset) onResults([]);
  }

  function search(text) {
    if (disposed) return;
    query = String(text ?? '').trim();
    sequence++;
    desiredId = null;
    clearTimeout(debounce);
    debounce = null;
    emitKeywords(true);
    if (!enabled) report('off', { message: 'Semantic search is off. Keyword search is available.' });
    else if (failure && query) report('error', { message: `${failure} Keyword search is still available. Enable semantic search to retry.` });
    else if (loading) report('loading');
    else report(workerReady ? 'ready' : 'idle', { message: workerReady ? 'Semantic search is ready on this device.' : 'Enter a query to start semantic search.' });
    if (query && enabled && !failure) debounce = setTimeout(() => { debounce = null; begin(sequence); }, 250);
  }

  function setSemantic(value) {
    if (disposed) return;
    enabled = Boolean(value);
    try { globalThis.localStorage.setItem(PREFERENCE, enabled ? 'on' : 'off'); } catch { /* Storage is optional. */ }
    sequence++;
    desiredId = null;
    clearTimeout(debounce);
    debounce = null;
    failure = '';
    emitKeywords(true);
    if (!enabled) {
      stop();
      report('off', { loaded: 0, total: 0, backend: '', message: 'Semantic search is off. Keyword search is available.' });
    } else {
      if (!loading && !workerReady) void load();
      else report(loading ? 'loading' : 'ready', { message: loading ? 'Loading semantic search.' : 'Semantic search is ready on this device.' });
      if (query) begin(sequence);
    }
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    sequence++;
    stop();
  }

  report(enabled ? 'idle' : 'off', { message: enabled ? 'Enter a query to start semantic search.' : 'Semantic search is off. Keyword search is available.' });
  return { search, setSemantic, dispose };
}
