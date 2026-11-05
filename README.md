# NXAI publications & resources page

Content stays in `links.json`; the build generates a text search index for the browser.
- `index.html` contains the page shell, styles and rendering logic.
- `links.json` contains every entry. Edit this file for routine content updates.
- `search/` contains the text index builder, browser worker, ranking and search interface.

## Deploy on GitHub Pages

Set the repository's Pages source to **GitHub Actions**.
The Pages workflow builds the text search index and deploys `_site/` after changes to `main`.
The site is served at `https://nx-ai.github.io/linktree/`.

## Adding or editing an entry

Open `links.json` and add an object to the array:

```json
{ "model": "tirex", "type": "paper", "icon": "paper", "title": "...", "meta": "Author et al. · Venue Year", "tag": "2611.xxxxx", "url": "https://arxiv.org/abs/2611.xxxxx" }
```

Field reference:
- `model`: one of `xlstm`, `tirex`, `vision-lstm`, `bio-xlstm`, `lram`. To add a new model family, also add it to the `MODELS` / `MODEL_KEY` lists near the top of the script in `index.html` — that's the one structural change that does require touching the HTML, since it drives the filter tabs.
- `type`: `model`, `paper` or `code`. Controls which subheading it appears under (Papers first, then Code & resources, then Models). Use `model` with `"icon": "huggingface"` for weights on Hugging Face (tag `Model` for a single repo, `Collection` for a collection).
- `icon`: `paper`, `github`, `huggingface`, `docker`, `jupyter`, `book`, `article`, `globe` (project or intro web page), or `package`. Brand icons render in their own brand colour; the rest render in teal. Papers use the generic `paper` glyph (no arXiv or conference logos).
- `meta`: freeform subtitle line, kept to the format `Author(s) et al. · Venue Year` for consistency.
- `tag`: short label shown as a pill on the right (arXiv ID, "GitHub", "Docs", etc).
- `date` and `venue` (papers only): power the Timeline view. `date` is `"YYYY-MM"` — use the actual conference month (not the arXiv submission month) once a paper is accepted; fall back to the arXiv month only for genuine preprints. `venue` is one of `NeurIPS`, `ICLR`, `ICML`, or `preprint`, and drives the colour-coded badge. Entries without both fields are simply left out of the timeline.

**Same content on two platforms** (like the "TiRex on the edge" post): instead of `"url"`, use a `"links"` array so both destinations get their own pill instead of implying two different articles:

```json
{ "model": "tirex", "type": "code", "icon": "article", "title": "Post title", "meta": "Same article, cross-posted",
  "links": [
    { "tag": "NXAI", "url": "https://www.nx-ai.com/..." },
    { "tag": "Hugging Face", "url": "https://huggingface.co/..." }
  ]
}
```

No `LAST_UPDATED` constant to remember anymore — the footer date is generated automatically from the page load.

## Local testing

Opening `index.html` directly from disk (`file://...`) will fail silently or show a load error, because browsers block `fetch()` of local files for security reasons. Serve the folder over HTTP instead:

```
npm ci
npm run build
python3 -m http.server 8000 --directory _site
```

Then open `http://localhost:8000/`.

Search uses the text encoder of [EmbeddingGemma 2](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX).
The build embeds the entries in `links.json`; browser workers embed queries with the same model.
Queries stay in the browser. Semantic search downloads about 377 MB when the model is not cached.
The browser can cache the model for later searches.
Keyword search works while the model loads and when semantic search is off or unavailable.

## Podcast banner

The Industrial AI Podcast banner under the header (Spotify, YouTube, Apple Podcasts buttons) is hardcoded in `index.html` (`a.podcast`), not in `links.json`, so it stays visible regardless of filter, search or view. Change the platform URLs or copy there.

## Download counts

Entries with an `"hf": "NX-AI/TiRex"` (Hugging Face repo) or `"pypi": "tirex-ts"` (PyPI package) field show an all-time download count (`↓ 3.8M`) next to the tag pill. Entries with the `github` icon also show their GitHub star count (`★ 1.2k`), read from the repo in their `url`, and `hf` entries show Hugging Face likes (`♥ 100`). Stars and likes only appear at 100 or more (`MIN_SOCIAL` in `index.html`); `stats.json` keeps all values. Numbers come from `stats.json`, which `.github/workflows/update-stats.yml` refreshes weekly (Monday mornings) via `scripts/update_stats.py` (Hugging Face API, the public ClickHouse PyPI dataset for PyPI — the same source as https://nx-ai.github.io — and the GitHub API for stars). Run the script by hand with `python3 scripts/update_stats.py`, or trigger the workflow from the Actions tab. Entries without those fields, such as collections, show no number.
