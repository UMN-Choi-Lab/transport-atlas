# Transport Atlas

A static, browsable atlas of transportation research papers — modeled after [robopaper-atlas](https://gisbi-kim.github.io/robopaper-atlas/). Ingests metadata from OpenAlex + IEEE Xplore and full text from Elsevier ScienceDirect, produces a dedup'd corpus, and renders a four-view static site (Explorer, Papers by Year, Coauthor Network, stretch: Topics).

## Deployed site

Base URL: **https://choi-seongjin.github.io/transport-atlas/**

Per-view URLs (same pattern for every rendered page in `site/`):

- https://choi-seongjin.github.io/transport-atlas/index.html
- https://choi-seongjin.github.io/transport-atlas/explorer.html
- https://choi-seongjin.github.io/transport-atlas/by_year.html
- https://choi-seongjin.github.io/transport-atlas/venues.html
- https://choi-seongjin.github.io/transport-atlas/coauthor_network.html
- https://choi-seongjin.github.io/transport-atlas/topic_space.html
- https://choi-seongjin.github.io/transport-atlas/combined.html

### Repositories

- **Code (this dir)**: `git@github.com:UMN-Choi-Lab/transport-atlas.git` — Python pipeline, templates, scripts, configs. Does **not** contain the built site or any data files (see `.gitignore`).
- **Deployed site**: `/home/chois/gitsrcs/choi-seongjin.github.io/` (repo `git@github.com:choi-seongjin/choi-seongjin.github.io.git`, branch `gh-pages`). GitHub Pages serves `transport-atlas/` subdir under the user site.

### Deploy flow (end-to-end)

```bash
# 1. Rebuild the site (picks up template + data changes)
python3 scripts/04_render.py                         # emits site/*.html with new cacheBust

# 2. Sync into the GH Pages repo
rsync -av --delete site/ /home/chois/gitsrcs/choi-seongjin.github.io/transport-atlas/

# 3. Commit + push (gh-pages branch)
cd /home/chois/gitsrcs/choi-seongjin.github.io
git add transport-atlas/
git commit -m "transport-atlas: <what changed>"
git push origin gh-pages
```

GitHub Pages CDN usually propagates in ~1 min; hard-reload to bypass browser cache since HTML files are served without `?v=` cache-bust.

**Partial deploys**: if a background job is still updating `site/data/`, deploy only the HTML files with `rsync -av site/*.html <dest>/` to avoid disturbing the in-progress data output.

**Cache-busting `?v=` rule**: data JSONs are referenced as `data/foo.json?v={{ cache_bust }}` in templates, so browsers refetch them whenever the render timestamp changes. Bump the cacheBust by re-running `scripts/04_render.py` — don't hand-edit the string.

## Target venues

| Venue | ISSN (print) | ISSN (online) | Primary source |
|-------|-------------|---------------|----------------|
| TR Part A (Policy & Practice) | 0965-8564 | 1879-2375 | OpenAlex + Elsevier |
| TR Part B (Methodological) | 0191-2615 | 1879-2367 | OpenAlex + Elsevier |
| TR Part C (Emerging Technologies) | 0968-090X | 1879-2359 | OpenAlex + Elsevier |
| TR Part D (Transport & Environment) | 1361-9209 | 1879-2340 | OpenAlex + Elsevier |
| TR Part E (Logistics & Transp. Review) | 1366-5545 | 1878-5794 | OpenAlex + Elsevier |
| TR Part F (Traffic Psych. & Behaviour) | 1369-8478 | 1873-5517 | OpenAlex + Elsevier |
| Transportation Research Record (TRB / SAGE) | 0361-1981 | 2169-4052 | OpenAlex |
| IEEE T-ITS | 1524-9050 | 1558-0016 | OpenAlex + IEEE |
| IEEE T-IV | 2379-8858 | 2379-8904 | OpenAlex + IEEE |
| IEEE ITS Magazine | 1939-1390 | 1941-1197 | OpenAlex + IEEE |
| IEEE Intelligent Vehicles Symposium (proc.) | — | — | OpenAlex + IEEE |

## Data source priority

1. **OpenAlex** — primary metadata (title, authors w/ ORCID + OpenAlex author IDs, year, venue, abstract, citation count, concepts). Free, no auth required for reads.
2. **IEEE Xplore** — enriches IEEE venues with abstracts/keywords. Requires `IEEE_API_KEY`.
3. **Elsevier ScienceDirect** — full text for TR A-F only. Requires `ELSEVIER_KEY` + `ELSEVIER_INSTTOKEN`. Used for topic modeling and keyword extraction, **never shipped to `site/`** (license).
4. **Crossref** — fallback for DOIs missing from OpenAlex (polite pool with `CROSSREF_EMAIL`).

## Stack

- Python 3.11
- Storage: Parquet via DuckDB (no CSV for >10k rows)
- Graph: `networkx` build → `python-igraph` + `leidenalg` communities → `fa2_modified` precomputed layout
- Templating: Jinja2 → static HTML
- Frontend (CDN, zero build step): Chart.js 4, Tabulator 6, Sigma.js 2 + Graphology
- Dedup: DOI first; fallback `rapidfuzz.ratio > 95` on normalized title AND same year AND overlapping author surnames

## Directory conventions

```
config/          venue list, pipeline params
data/raw/        per-source, per-venue checkpointed ingest output
data/interim/    dedup'd parquets
data/processed/  final parquets + site-ready JSON (<venue>.json, coauthor_graph.json, by_year.json, papers.json, top_hubs.json)
src/transport_atlas/
  ingest/        openalex.py, elsevier.py, ieee.py, _http.py
  process/       dedupe.py, authors.py, coauthor_graph.py, aggregate.py, topics.py
  site/          render.py, templates/
  utils/         config.py (loads ~/.claude/mcp-servers/refcheck/.env), logger.py
scripts/         01_ingest.py, 02_dedupe.py, 03_graph.py, 04_render.py, refresh.py
tests/           offline unit tests + fixtures/small_corpus.json
site/            rendered static site (gitignored)
```

## API keys

All API keys are read from `~/.claude/mcp-servers/refcheck/.env` via `transport_atlas.utils.config`. Keys: `ELSEVIER_KEY`, `ELSEVIER_INSTTOKEN`, `IEEE_API_KEY`, `S2_API_KEY`, `CROSSREF_EMAIL`. **Never commit keys** to this repo — they live outside the project tree.

## Refresh commands

```bash
# full pipeline (can be resumed)
python scripts/refresh.py

# individual phases
python scripts/01_ingest.py --source openalex        # all venues
python scripts/01_ingest.py --source ieee
python scripts/01_ingest.py --source elsevier --venue TR-C
python scripts/02_dedupe.py
python scripts/03_graph.py
python scripts/04_render.py

# preview
cd site && python -m http.server 8000
```

## Project rules

- **Elsevier full text**: store plaintext + section headings only, never raw XML; keep in `data/raw/elsevier/` (gitignored); never in `site/`.
- **DuckDB over pandas** for anything >10k rows (per global rule); never `ast.literal_eval`.
- **Graph layout** is precomputed in Python (deterministic seed); the browser never re-runs physics for >5k nodes.
- **Cache busting**: every JSON shipped to `site/` is fingerprinted with `?v=<YYYYMMDD-HHMM>`.
- **Tests run offline** — network-dependent paths must be exercised via `tests/fixtures/`.
- **Dedup**: DOI first; fuzzy title + year + author surname overlap second; never title alone.
- **Checkpoints are sacred**: per-venue `_meta.json` and per-DOI file existence mean reruns resume. Don't add "force-refresh" as the default; it must always be an explicit `--force` flag.
- **Strict JSON for browser**: every `json.dumps` that writes to `site/data/` or `data/processed/*.json` must pass `allow_nan=False`. Python defaults to emitting bare `NaN`/`Infinity` tokens which pandas NaN values produce silently — the browser's `JSON.parse` rejects them and the whole page fails to load. When sourcing fields from a DataFrame, check `isinstance(x, str)` (not `if x:`) because `bool(float('nan'))` is `True`.

## Key file references

- Venue ISSNs: `config/venues.yaml`
- Pipeline params (rate limits, seeds, date ranges): `config/pipeline.yaml`
- Secret keys: `~/.claude/mcp-servers/refcheck/.env` (outside this repo)
- Reference HTTP patterns: `~/.claude/mcp-servers/refcheck/src/` (do not import; study only)
