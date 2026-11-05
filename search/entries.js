/** Convert resource records to ordered, text-only search entries without changing the records. */
export function linkEntries(links) {
  const families = {
    xlstm: 'xLSTM architecture and language models',
    tirex: 'TiRex time series forecasting',
    'vision-lstm': 'Vision-LSTM computer vision',
    'bio-xlstm': 'Bio-xLSTM biological and chemical sequences',
    lram: 'LRAM robotics action models',
  };
  return links.map((item) => {
    const id = item.url || item.links?.[0]?.url;
    if (typeof id !== 'string' || !id || typeof item.title !== 'string' || typeof item.model !== 'string') {
      throw new Error('A resource needs a URL, title and model family.');
    }
    const tags = [item.tag, ...(item.links || []).map((link) => link.tag)].filter(Boolean);
    const text = [
      `Model family: ${item.model} ${families[item.model] || ''}`.trim(),
      `Resource type: ${item.type}`,
      `Title: ${item.title}`,
      item.meta && `Details: ${item.meta}`,
      tags.length && `Tags: ${tags.join(', ')}`,
      item.date && `Date: ${item.date}`,
      item.venue && `Venue: ${item.venue}`,
    ].filter(Boolean).join(' | ');
    return { id, title: item.title, text, group: item.model, url: id };
  });
}

export function validateEntries(entries) {
  if (!Array.isArray(entries)) throw new Error('Search entries must be an array.');
  const ids = new Set();
  for (const entry of entries) {
    for (const field of ['id', 'title', 'text', 'group', 'url']) {
      if (typeof entry?.[field] !== 'string' || !entry[field].trim()) {
        throw new Error(`Search entry has an invalid ${field}.`);
      }
    }
    if (ids.has(entry.id)) throw new Error(`Duplicate search entry: ${entry.id}`);
    ids.add(entry.id);
  }
}
