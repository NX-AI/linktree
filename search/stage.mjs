import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { linkEntries, validateEntries } from './entries.js';

const ROOT = path.resolve(import.meta.dirname, '..');

/** Stage browser assets without downloading or running the embedding model. */
export function stageSite(outputDir) {
  const output = path.resolve(ROOT, outputDir);
  const searchDir = path.join(output, 'search');
  fs.mkdirSync(searchDir, { recursive: true });
  for (const name of ['index.html', 'links.json', 'stats.json', 'assets']) {
    fs.cpSync(path.join(ROOT, name), path.join(output, name), { recursive: true });
  }
  const entries = linkEntries(JSON.parse(fs.readFileSync(path.join(ROOT, 'links.json'), 'utf8')));
  validateEntries(entries);
  for (const file of fs.readdirSync(import.meta.dirname, { withFileTypes: true })) {
    if (file.isFile() && (file.name.endsWith('.js') || file.name === 'ui.css')) {
      fs.copyFileSync(path.join(import.meta.dirname, file.name), path.join(searchDir, file.name));
    }
  }
  const modules = path.join(ROOT, 'node_modules');
  const transformers = path.join(modules, '@huggingface', 'transformers');
  if (JSON.parse(fs.readFileSync(path.join(transformers, 'package.json'), 'utf8')).version !== '4.3.1') {
    throw new Error('Search requires @huggingface/transformers 4.3.1. Run npm ci.');
  }
  const runtime = path.join(searchDir, 'runtime');
  fs.mkdirSync(runtime, { recursive: true });
  fs.copyFileSync(path.join(transformers, 'dist', 'transformers.min.js'), path.join(runtime, 'transformers.min.js'));
  for (const build of ['asyncify', 'jsep']) {
    for (const name of [`ort-wasm-simd-threaded.${build}.mjs`, `ort-wasm-simd-threaded.${build}.wasm`]) {
      fs.copyFileSync(path.join(modules, 'onnxruntime-web', 'dist', name), path.join(runtime, name));
    }
  }
  return { entries, searchDir };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  if (process.argv.length !== 3) throw new Error('usage: node search/stage.mjs <output-dir>');
  const { entries, searchDir } = stageSite(process.argv[2]);
  console.log(`Staged ${entries.length} entries and the browser runtime in ${searchDir}; no embeddings were run.`);
}
