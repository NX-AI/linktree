# NXAI publications & resources page

Two files, no build step, no dependencies beyond a Google Fonts CDN link:
- `index.html` — page shell, styling, rendering logic. You should not need to touch this for routine updates.
- `links.json` — every entry on the page. This is the only file you edit to add, remove, or correct a link.

## Deploy on GitHub Pages

Option A: dedicated repo (recommended for a short URL like `nx-ai.com/links` via CNAME later)
1. Create a repo, e.g. `NX-AI/linktree`.
2. Commit both `index.html` and `links.json` to the `main` branch, in the same folder.
3. Repo Settings > Pages > Source: `main` branch, `/ (root)`.
4. Page is live at `https://nx-ai.github.io/linktree/`.

Option B: fold into an existing repo (e.g. the `tirex` repo) as a `/docs` folder if you want to reuse Pages already configured there. Keep both files together in that folder.

## Adding or editing an entry

Open `links.json` and add an object to the array:

```json
{ "model": "tirex", "type": "paper", "icon": "arxiv", "title": "...", "meta": "Author et al. · Venue Year", "tag": "2611.xxxxx", "url": "https://arxiv.org/abs/2611.xxxxx" }
```

Field reference:
- `model`: one of `xlstm`, `tirex`, `vision-lstm`, `bio-xlstm`, `lram`. To add a new model family, also add it to the `MODELS` / `MODEL_KEY` lists near the top of the script in `index.html` — that's the one structural change that does require touching the HTML, since it drives the filter tabs.
- `type`: `paper` or `code`. Controls which subheading it appears under.
- `icon`: `arxiv`, `github`, `huggingface`, `docker`, `jupyter`, `book`, `article`, or `package`. Brand icons render in their own brand colour; the rest render in teal.
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
python3 -m http.server 8000
```

then open `http://localhost:8000/`. This restriction disappears once deployed to GitHub Pages, since that's real HTTP.
