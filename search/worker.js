import { MODEL, normalizeVector } from './core.js';

const runtimeBase = new URL('./runtime/', import.meta.url).href;
let extractor = null;
let latest = null;
let busy = false;
let backend = '';
let failed = false;
const downloads = new Map();

function fail(error) {
  if (failed) return;
  failed = true;
  latest = null;
  extractor = null;
  try {
    self.postMessage({ type: 'error', message: String(error) });
  } finally {
    self.close();
  }
}

// The model loader can reject promises outside the awaited pipeline.
self.addEventListener('unhandledrejection', (event) => {
  fail(event.reason);
  event.preventDefault();
});

function onProgress(info) {
  if (failed || info.status !== 'progress' || /vision_encoder|audio_encoder/.test(info.file)) return;
  if (!Number.isFinite(info.loaded) || !Number.isFinite(info.total)) return;
  downloads.set(info.file, info);
  let loaded = 0;
  let total = 0;
  for (const file of downloads.values()) { loaded += file.loaded; total += file.total; }
  self.postMessage({ type: 'progress', loaded, total });
}

async function load(meta, forceWasm) {
  if (Object.entries(MODEL).some(([key, value]) => meta[key] !== value)) throw new Error('Unexpected search model.');
  const tf = await import(runtimeBase + 'transformers.min.js');
  tf.env.allowLocalModels = false;
  tf.env.backends.onnx.wasm.numThreads = 1;
  tf.env.backends.onnx.wasm.proxy = false;
  // Pin size probes as well as downloads, and reuse the configuration response.
  const hubFetch = tf.env.fetch;
  const configs = new Map();
  tf.env.fetch = (url, init) => {
    if (typeof url !== 'string') return hubFetch(url, init);
    const pinned = url.replace(`/${meta.model}/resolve/main/`, `/${meta.model}/resolve/${meta.revision}/`);
    if (/\/(?:vision|audio)[^/]*\.onnx/.test(pinned)) throw new Error('Only the text model is permitted.');
    if (!pinned.endsWith('/config.json') || init?.headers?.Range) return hubFetch(pinned, init);
    if (!configs.has(pinned)) configs.set(pinned, hubFetch(pinned, init));
    return configs.get(pinned).then((response) => response.clone());
  };
  const runtime = (build) => {
    tf.env.backends.onnx.wasm.wasmPaths = {
      mjs: `${runtimeBase}ort-wasm-simd-threaded.${build}.mjs`,
      wasm: `${runtimeBase}ort-wasm-simd-threaded.${build}.wasm`,
    };
  };
  const config = await tf.AutoConfig.from_pretrained(meta.model, { revision: meta.revision });
  delete config.vision_config;
  delete config.audio_config;
  const options = (device) => ({
    device, dtype: meta.dtype, revision: meta.revision, config, progress_callback: onProgress,
  });
  let adapter = null;
  try {
    if (!forceWasm && self.navigator.gpu) adapter = await self.navigator.gpu.requestAdapter();
  } catch { /* The browser can deny WebGPU access. */ }
  if (adapter) {
    try {
      runtime('asyncify');
      extractor = await tf.pipeline('feature-extraction', meta.model, options('webgpu'));
      backend = 'WebGPU';
    } catch {
      if (failed) return;
      // ONNX Runtime retains WebGPU state. The page must create a fresh worker for WASM.
      self.postMessage({ type: 'fallback' });
      return;
    }
  } else {
    runtime('jsep');
    extractor = await tf.pipeline('feature-extraction', meta.model, options('wasm'));
    backend = 'WASM';
  }
  if (!failed) self.postMessage({ type: 'ready', backend });
}

async function drain() {
  if (busy || !extractor) return;
  busy = true;
  try {
    while (latest) {
      const { id, text } = latest;
      latest = null;
      try {
        const out = await extractor(MODEL.query_prefix + text, { pooling: 'mean', normalize: true });
        if (failed) return;
        if (out.dims.length !== 2 || out.dims[0] !== 1 || !Number.isInteger(out.dims[1]) ||
            out.dims[1] < MODEL.dim || out.data.length !== out.dims[1]) {
          throw new Error('The query model returned an invalid embedding shape.');
        }
        const vector = normalizeVector(out.data);
        self.postMessage({ type: 'vector', id, vector }, [vector.buffer]);
      } catch (error) {
        if (failed) return;
        if (backend === 'WebGPU') {
          self.postMessage({ type: 'fallback' });
          return;
        }
        fail(error);
        return;
      }
    }
  } finally {
    busy = false;
  }
}

self.onmessage = async ({ data }) => {
  if (failed) return;
  if (data.type === 'load') {
    try { await load(data.meta, data.forceWasm); }
    catch (error) { fail(error); }
  } else if (data.type === 'embed' && Number.isInteger(data.id) && typeof data.text === 'string') {
    latest = data;
    void drain();
  }
};
