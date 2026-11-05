import { createSearch } from './engine.js';

export const MATCH_LABELS = { keyword: 'Keyword', semantic: 'Semantic', both: 'Both' };
export const RESULT_LIMIT = 12;

export function attachSearch(entries, onResults, onState = () => {}) {
  const input = document.getElementById('search');
  const toggle = document.getElementById('semantic-toggle');
  const status = document.getElementById('search-status');
  const progress = document.getElementById('search-progress');
  const action = document.getElementById('search-download');
  const count = document.getElementById('search-count');
  const container = document.getElementById('search-settings');
  const toggleLabel = toggle.closest('label');
  let state = 'idle';

  const engine = createSearch(entries, {
    onResults(results) {
      const total = results.length;
      count.textContent = !input.value.trim() ? '' : total > RESULT_LIMIT
        ? `Showing ${RESULT_LIMIT} of ${total} matches`
        : `${total} match${total === 1 ? '' : 'es'}`;
      onResults(results);
    },
    onStatus(info) {
      toggle.checked = info.enabled;
      progress.hidden = info.state !== 'loading';
      progress.max = info.total || 1;
      progress.value = info.loaded || 0;
      toggleLabel.hidden = info.state === 'off';
      action.hidden = info.state !== 'off' && info.state !== 'error';
      action.textContent = info.state === 'off' ? 'Enable semantic search (about 377 MB)' : 'Retry';
      let message;
      switch (info.state) {
        case 'loading':
          message = info.total
            ? `Loading text model: ${Math.round(info.loaded / 1e6)} of ${Math.round(info.total / 1e6)} MB`
            : 'Loading text model (about 377 MB when not cached)…';
          break;
        case 'searching': message = 'Searching by meaning…'; break;
        case 'ready': message = `Words + meaning${info.backend ? ` · ${info.backend}` : ''}`; break;
        case 'off': message = 'Keyword search is active. Semantic search runs in your browser after you enable it.'; break;
        case 'error': message = 'Semantic search is unavailable. Keyword search still works.'; break;
        default: message = 'Text only · about 377 MB download when not cached · queries stay in your browser.';
      }
      if (status.textContent !== message) status.textContent = message;
      if (state !== info.state) {
        state = info.state;
        onState(info);
      }
      container.dataset.state = info.state;
    },
  });

  input.addEventListener('input', () => engine.search(input.value));
  input.addEventListener('keydown', (event) => {
    if (event.isComposing || event.keyCode === 229) return;
    if (event.key === 'Escape' && input.value) {
      event.preventDefault();
      input.value = '';
      engine.search('');
    }
  });
  toggle.addEventListener('change', () => engine.setSemantic(toggle.checked));
  action.addEventListener('click', () => engine.setSemantic(true));
  window.addEventListener('keydown', (event) => {
    if (event.isComposing || event.altKey || !(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'k') return;
    if (!input.getClientRects().length) return;
    event.preventDefault();
    input.focus();
    input.select();
  });
  window.addEventListener('pagehide', (event) => { if (!event.persisted) engine.dispose(); });
  document.querySelector('.semantic-search').hidden = false;
  if (input.value) engine.search(input.value);
  return engine;
}

const source = document.getElementById('search-entries');
if (source) {
  const input = document.getElementById('search');
  const list = document.getElementById('search-results');
  const empty = document.getElementById('search-empty');
  let latest = [];
  let state = 'idle';
  function updateEmpty() {
    const hasQuery = !!input.value.trim();
    list.hidden = !hasQuery || !latest.length;
    empty.hidden = !hasQuery || !!latest.length;
    empty.textContent = state === 'loading' || state === 'searching'
      ? 'No keyword matches yet. Waiting for semantic results…'
      : 'No matches. Try another word or a shorter question.';
  }
  attachSearch(JSON.parse(source.textContent), (results) => {
    latest = results;
    list.replaceChildren();
    for (const result of results.slice(0, RESULT_LIMIT)) {
      const item = document.createElement('li');
      item.className = 'search-result';
      const link = document.createElement('a');
      link.href = result.url;
      const head = document.createElement('span');
      head.className = 'search-result-head';
      const title = document.createElement('span');
      title.className = 'search-result-title';
      title.textContent = result.title;
      const match = document.createElement('span');
      match.className = 'search-match';
      match.textContent = MATCH_LABELS[result.match];
      head.append(title, match);
      const text = document.createElement('p');
      text.className = 'search-result-text';
      text.textContent = result.snippet;
      link.append(head, text);
      item.append(link);
      list.append(item);
    }
    updateEmpty();
  }, (info) => { state = info.state; updateEmpty(); });
}
