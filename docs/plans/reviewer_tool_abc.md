# Reviewer-finder upgrade: A + B + C

Driven by Geroliminis (TR-C editor) feedback, 2026-04-26.

## Goals

- **A** — let the user paste a paper's bibliography; resolved DOIs become a soft positive
  signal blended into the SPECTER2 query vector.
- **B** — for each ranked reviewer, surface the top-3 of *their* papers most similar to the
  query, so the editor can see the evidence behind each recommendation.
- **C** — make the coauthor-of-author flag prominent (Elsevier shows this clearly; we
  already compute it but render it as a small "d=1" string).

Out of scope (Geroliminis ask D, "diversity"): defer to a follow-up; needs OpenAlex
institution aggregation in the ingest path.

## Constraints

- Static site on GitHub Pages, zero server logic.
- Browser must be able to dequantize and dot-product on demand.
- First-load weight cannot balloon: today's tool already pulls ~45 MB (authors bin + index
  JSON + 30 MB ONNX). Ceiling for "feels OK on broadband" is ~+15 MB on first load.

## Data shape

### New static artifacts

| File | Size | Purpose |
|---|---|---|
| `site/data/paper_emb.bin` | ~83 MB | Per-paper int8 SPECTER2 embeddings + per-row scale. **Range-fetched** (we verified GH-Pages returns HTTP 206 with `accept-ranges: bytes`). |
| `site/data/paper_index.json` | ~12 MB | Per-paper metadata + `doi_to_idx` map. Eager-fetched on first "Find" click. |

### `paper_emb.bin` row layout

Every row is exactly **772 bytes**, so range-fetching row `i` is `bytes=[i*772, i*772+771]`:

```
[scale: float32 LE (4 B)][int8 vec × 768 (768 B)]
```

Browser dequant: `vec_d = scale * int8_d / 127`. Same convention as `reviewer_authors.bin`.

### `paper_index.json` schema

```json
{
  "version": 1,
  "row_bytes": 772,
  "n_papers": <int>,
  "doi_to_idx": { "10.xxxx/yyyy": <int>, ... },
  "papers": [
    { "t": "<title>", "y": <year>, "v": "<venue_slug>", "d": "<doi or null>" },
    ...
  ]
}
```

Filter rule: include a paper iff it has a SPECTER2 embedding **and** appears in at least
one indexed author's contribution list. Papers with no DOI are still included (so ask B
can show evidence) — only the `doi_to_idx` map skips them.

### `reviewer_index.json` extension

Each `authors[i]` gains:

```json
{ ..., "pi": [<paper_idx>, ...] }
```

Sorted by author-centroid cosine descending, capped at **30 papers per author** (enough
for the per-query rerank to find the top 3, while keeping JSON growth modest:
30k authors × 30 ints × ~6 bytes = ~5 MB extra in `reviewer_index.json`).

## Build pipeline (`scripts/10_reviewer_index.py`)

After computing `A` (per-author centroids) and the int8 quantization, add a fourth phase:

1. Iterate the paper-author contribution map already built (`author_rows`).
2. Collect every (author_idx, paper_pidx) pair. Compute author-paper score = cosine of
   author centroid vs paper SPECTER2 embedding.
3. For each author, take top-30 papers by score; record `paper_idxs` (indices into a new
   global `kept_papers` list — only papers that some indexed author cares about).
4. Quantize per-paper embeddings the same way (per-row abs-max, int8) and write
   `paper_emb.bin`.
5. Write `paper_index.json` with metadata and `doi_to_idx`.

Memory: 120k × 768 float32 = 350 MB to hold full `E`; we already do that in the build
script — fine for the build host, never goes to the browser.

## Frontend (`templates/reviewers.html`)

### A — references blend

- New textarea: "References (optional, paste bibliography — we'll DOI-match into the
  atlas corpus and use them as positive signal)".
- Regex DOI extract: `/\b(10\.\d{4,9}\/[-._;()/:A-Za-z0-9]+)\b/g` (greedy, lowercase
  the match).
- Resolve to indices via `doi_to_idx`; report unmatched DOIs in the status line.
- Range-fetch matched rows in parallel (`fetch(url, { headers: { Range: ... } })`).
  HTTP/2 multiplexes; browser cache picks up duplicates across queries.
- Compute mean of dequantized matched embeddings, L2-normalize → `refsVec`.
- Blend: `queryVec = α·titleAbsVec + (1-α)·refsVec`, where α defaults to **0.5** but is
  exposed as a slider ("Refs weight: 0.0 – 1.0").
- L2-renormalize, then score as today.

### B — per-reviewer evidence panel

- After ranking, for the top-N rows, range-fetch each author's `pi[]` paper rows in one
  batched parallel call.
- Score `dot(queryVec, paperVec)` for each paper; sort; take top 3.
- Render under each row in an expandable details element:
  ```
  ▸ Why this match: 3 papers
    • [0.71] Title — Year · Venue
    • [0.68] Title — Year · Venue
    • [0.65] Title — Year · Venue
  ```
  Auto-expanded for top-5; collapsed for the rest.

### C — coauthor badge

Replace the small `d=1` / `d=2` string with:

- `d=1` → red badge "★ Direct coauthor" with tooltip "Coauthored with: <names>"
- `d=2` → amber badge "Co-coauthor (1 hop away)" with tooltip listing the bridge author
- `d=3+` → small grey "—"

Tooltip data: when we run BFS, record the parent for each visited node; on render, walk
the parent chain back to the seed and list the bridge.

## Risks / open questions

- **Range-fetch fan-out**: 50 references × parallel range = 50 sub-requests. Modern
  browsers cap to ~6 concurrent per origin without HTTP/2 ALPN — but GH-Pages negotiates
  HTTP/2 (verified `HTTP/2 200`). Should be fine, but worth profiling.
- **Bibliography DOI extraction**: many bibs are formatted without explicit DOIs (just
  author/year). For v1, only DOI-bearing references count; we'll show "matched X / Y" so
  the user understands.
- **Cache invalidation**: existing `?v={{ cache_bust }}` query string applies. Range
  responses respect ETag, so no extra logic needed.

## Stepwise execution

1. Extend `10_reviewer_index.py` (build new artifacts; add `pi` field to authors).
2. Run inside the embed Docker image to regenerate `data/processed/*` outputs.
3. Update template — add references textarea, A logic, B panel, C badge.
4. `python scripts/04_render.py` — bumps cache-bust.
5. `rsync` to GH-Pages repo, commit, push.
6. Smoke-test live: paste a real paper's title+abstract+refs; verify A blends, B shows
   per-reviewer papers, C highlights coauthors.

## Acceptance

- A real query (e.g., Choi et al. trajectory paper) returns a top-30 list where a few
  known coauthors are flagged with C, the recommended reviewers each show 3 plausible
  papers in B, and adding refs visibly reorders the list (different from no-refs run).
- First-load page weight (without refs) unchanged.
- With refs: extra fetch is ≤ 1 MB for typical 20-ref bibs; ≤ 2 MB for the worst-case
  large bib.
