import { validateEntries } from './entries.js';

export const MODEL = Object.freeze({
  model: 'onnx-community/embeddinggemma-2-ONNX',
  revision: 'daa72c51243991dfcaf9f9137d2c573d8f7790c0',
  dtype: 'q8',
  dim: 512,
  query_prefix: 'task: search result | query: ',
});
const RRF_K = 60;
const STOP = new Set(('a an and are as at be by can do does for from how i in is it of on or ' +
  'the this to use what when where which with').split(' '));

// Field labels such as "Resource type: " belong to the embedding text, not to keyword matching:
// "model" in "Model family: " would otherwise match every entry.
const LABEL = /(?:^|\s\|\s)[A-Z][A-Za-z ]{0,30}:\s/g;

export function embeddingTexts(entries) {
  return entries.map((entry) => `title: ${entry.title} | text: ${entry.text}`);
}

export function normalizeVector(full) {
  if (!full || !Number.isInteger(full.length) || full.length < MODEL.dim || !Array.from(full).every(Number.isFinite)) {
    throw new Error('The search model returned an invalid vector.');
  }
  const vector = Float32Array.from(Array.from(full).slice(0, MODEL.dim));
  const norm = Math.hypot(...vector);
  if (!Number.isFinite(norm) || norm <= 0) throw new Error('The search vector has no finite norm.');
  return vector.map((value) => value / norm);
}

function tokens(text) {
  const result = [];
  for (const match of text.toLowerCase().matchAll(/[a-z0-9]+|[぀-ヿ㐀-鿿가-힯]+/g)) {
    const word = match[0];
    if (/^[a-z0-9]/.test(word)) {
      if (!STOP.has(word)) result.push(word.length > 3 && word.endsWith('s') ? word.slice(0, -1) : word);
    } else {
      for (let i = 0; i < word.length; i++) {
        result.push(word[i]);
        if (i + 1 < word.length) result.push(word.slice(i, i + 2));
      }
    }
  }
  return result;
}

export function createKeywordIndex(entries) {
  validateEntries(entries);
  const docs = entries.map((entry) => {
    const counts = new Map();
    const title = tokens(entry.title);
    const body = tokens(`${entry.text.replace(LABEL, ' ')} ${entry.group} ${entry.url}`);
    for (const term of title) counts.set(term, (counts.get(term) || 0) + 3);
    for (const term of body) counts.set(term, (counts.get(term) || 0) + 1);
    return { counts, length: title.length * 3 + body.length };
  });
  const df = new Map();
  for (const doc of docs) for (const term of doc.counts.keys()) df.set(term, (df.get(term) || 0) + 1);
  const average = docs.reduce((sum, doc) => sum + doc.length, 0) / (docs.length || 1);
  return { entries, docs, df, average };
}

/** With requireAll, only entries that contain every query term are returned. */
export function keywordSearch(index, query, { requireAll = false } = {}) {
  const terms = [...new Set(tokens(query))];
  const { docs, df, average } = index;
  const ranked = [];
  docs.forEach((doc, id) => {
    if (requireAll && !terms.every((term) => doc.counts.has(term))) return;
    let score = 0;
    for (const term of terms) {
      const count = doc.counts.get(term);
      if (!count) continue;
      const frequency = df.get(term);
      const idf = Math.log(1 + (docs.length - frequency + 0.5) / (frequency + 0.5));
      score += idf * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * doc.length / average));
    }
    if (score > 0) ranked.push({ id, score });
  });
  return ranked.sort((a, b) => b.score - a.score || a.id - b.id);
}

/** Validate the full entry snapshot before associating any vectors with the live page. */
export function createVectorIndex(entries, meta, buffer) {
  if (!meta || meta.version !== 1 || Object.entries(MODEL).some(([key, value]) => meta[key] !== value)) {
    throw new Error('The semantic search model or index format is out of date.');
  }
  validateEntries(meta.entries);
  const fields = ['id', 'title', 'text', 'group', 'url'];
  if (meta.entries.length !== entries.length || entries.some((entry, i) =>
    fields.some((field) => entry[field] !== meta.entries[i][field]))) {
    throw new Error('The semantic search index does not match the current content.');
  }
  const count = entries.length;
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== count * (4 + MODEL.dim)) {
    throw new Error('The semantic search vector file has an invalid length.');
  }
  const scales = new Float32Array(buffer, 0, count);
  const codes = new Int8Array(buffer, count * 4, count * MODEL.dim);
  for (let i = 0; i < count; i++) {
    let norm = 0;
    for (let j = 0; j < MODEL.dim; j++) norm += (codes[i * MODEL.dim + j] * scales[i]) ** 2;
    if (!Number.isFinite(scales[i]) || scales[i] <= 0 || !Number.isFinite(norm) || norm <= 0) {
      throw new Error('The semantic search index contains an invalid vector.');
    }
  }
  return { scales, codes, count };
}

export function semanticSearch(index, full) {
  if (full?.length !== MODEL.dim) throw new Error('The query vector has an invalid dimension.');
  const query = normalizeVector(full);
  const ranked = [];
  for (let id = 0; id < index.count; id++) {
    let score = 0;
    for (let j = 0; j < MODEL.dim; j++) score += query[j] * index.codes[id * MODEL.dim + j];
    score *= index.scales[id];
    if (score > 0) ranked.push({ id, score });
  }
  return ranked.sort((a, b) => b.score - a.score || a.id - b.id);
}

export function combine(entries, keyword, semantic = []) {
  const scores = new Map();
  const keywordIds = new Set(keyword.map(({ id }) => id));
  const semanticIds = new Set(semantic.map(({ id }) => id));
  for (const ranked of [keyword, semantic]) {
    ranked.forEach(({ id }, rank) => scores.set(id, (scores.get(id) || 0) + 1 / (RRF_K + rank + 1)));
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([id, score]) => ({
      ...entries[id],
      match: keywordIds.has(id) ? (semanticIds.has(id) ? 'both' : 'keyword') : 'semantic',
      score,
    }));
}
