import fs from 'node:fs';
import path from 'node:path';
import { AutoConfig, env, pipeline } from '@huggingface/transformers';
import { MODEL, createVectorIndex, embeddingTexts, normalizeVector } from './core.js';
import { stageSite } from './stage.mjs';

if (process.argv.length !== 3) throw new Error('usage: node search/build.mjs <output-dir>');
const { entries, searchDir } = stageSite(process.argv[2]);
if (!entries.length) throw new Error('There are no search entries to embed.');
const texts = embeddingTexts(entries);
if (process.env.HF_CACHE_DIR) env.cacheDir = process.env.HF_CACHE_DIR;
const hubFetch = env.fetch;
env.fetch = (url, init) => {
  if (typeof url !== 'string') return hubFetch(url, init);
  const pinned = url.replace(`/${MODEL.model}/resolve/main/`, `/${MODEL.model}/resolve/${MODEL.revision}/`);
  if (/\/(?:vision|audio)[^/]*\.onnx/.test(pinned)) throw new Error('Only the text model is permitted.');
  return hubFetch(pinned, init);
};
const config = await AutoConfig.from_pretrained(MODEL.model, { revision: MODEL.revision });
delete config.vision_config;
delete config.audio_config;
const extractor = await pipeline('feature-extraction', MODEL.model, {
  device: 'cpu', dtype: MODEL.dtype, revision: MODEL.revision, config,
});

const order = texts.map((_, i) => i).sort((a, b) => texts[a].length - texts[b].length);
const codes = new Int8Array(texts.length * MODEL.dim);
const scales = new Float32Array(texts.length);
try {
  for (let start = 0; start < order.length; start += 4) {
    const ids = order.slice(start, start + 4);
    const out = await extractor(ids.map((id) => texts[id]), { pooling: 'mean', normalize: true });
    const [rows, width] = out.dims;
    if (out.dims.length !== 2 || rows !== ids.length || !Number.isInteger(width) || width < MODEL.dim || out.data.length !== rows * width) {
      throw new Error('The document model returned an invalid embedding shape.');
    }
    ids.forEach((id, row) => {
      const vector = normalizeVector(out.data.subarray(row * width, (row + 1) * width));
      const step = Math.max(...vector.map(Math.abs)) / 127;
      scales[id] = step;
      vector.forEach((value, j) => { codes[id * MODEL.dim + j] = Math.round(value / step); });
    });
    console.log(`Embedded ${Math.min(start + 4, order.length)}/${order.length} entries.`);
  }
} finally {
  await extractor.dispose();
}
const meta = { version: 1, ...MODEL, entries };
const buffer = Buffer.concat([Buffer.from(scales.buffer), Buffer.from(codes.buffer)]);
createVectorIndex(entries, meta, buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
fs.writeFileSync(path.join(searchDir, 'vectors.bin'), buffer);
fs.writeFileSync(path.join(searchDir, 'meta.json'), JSON.stringify(meta));
console.log(`Embedded ${entries.length} text entries (${MODEL.dtype}, ${MODEL.dim}-d, int8 storage).`);
