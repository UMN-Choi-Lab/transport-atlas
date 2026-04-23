# Paper: *Beyond coauthorship — a semantic-structural atlas of transportation research, 1967–2026*

Living workspace for the paper that uses **Transport Atlas** as its empirical
substrate, extending Sun & Rahwan (2017) *Coauthorship network in transportation
research* (TR Part A, 100, 135–151).

## Target venue
Transportation Research Part A: Policy and Practice (primary).
Backup: Scientometrics, Journal of Informetrics.

## Layout

| Path | Purpose |
|---|---|
| `outline.md` | Section-by-section outline with target word counts. The single source of truth for where prose lives. |
| `research_questions.md` | RQ1/2/3 with hypotheses and falsification criteria. Read this *before* interpreting any table or figure. |
| `related_work.md` | Annotated bibliography — key references plus one-line "why we cite" note. |
| `contributions.md` | Explicit list of contributions relative to Sun & Rahwan and other bibliometric transportation papers. |
| `data_plan.md` | Per-figure/table plan — which script generates what, which input file, expected row count / figure size. |
| `analysis/` | Analysis scripts. Each produces tables (as `.tex`) or figures (as `.pdf` + `.png`, 300 DPI, colorblind-friendly). |
| `figures/` | Generated figures, one per script. |
| `tables/` | Generated LaTeX `.tex` fragments — `\\input{tables/foo.tex}` into manuscript. |
| `manuscript/` | LaTeX source — scaffolded when Overleaf link arrives. |

## Status

Phase 1 (scaffolding) — in progress.
Phase 2 (descriptive tables / §4) — pending.
Phase 3 (partition alignment / §7) — pending.
Phase 4 (**phantom-collaborator predictive test / §8 — the novel contribution**) — pending.
Phase 5 (trajectory taxonomy / §9) — pending.
Phase 6 (LaTeX on Overleaf) — waits for user's Overleaf link.

## Corpus snapshot

| | |
|---|---|
| Papers | 111,057 |
| Venues | 29 |
| Year range | 1967–2026 |
| Authors (deduped) | ~149,000 |
| ORCID coverage | ~55.7% |
| SPECTER2 embeddings | 768-d, whitened (top-1 PC removed) + concept TF-IDF (128-d SVD) + venue LDA (28-d) = 925-d hybrid |
| Coauthor communities (Leiden, mainland) | 142 |
| Semantic communities (Leiden on hybrid kNN) | 23 |
| Combined/multiplex communities | 156 |
| Citation source | `max(OpenAlex, Crossref is-referenced-by-count)` |

Snapshot date for paper: pin later. Suggested: the commit SHA of `UMN-Choi-Lab/transport-atlas` on the day of first full-draft submission.

## Reproducibility statement (to migrate into §10)

- Code: `https://github.com/UMN-Choi-Lab/transport-atlas`
- Live atlas: `https://choi-seongjin.github.io/transport-atlas/`
- Upstream data: OpenAlex (CC0), Crossref (CC0), IEEE Xplore (API), Elsevier ScienceDirect (full-text local only — not redistributed per licence).
- Random seeds logged in `config/pipeline.yaml`.
- Figures: matplotlib, 300 DPI, PDF + PNG, colorblind-friendly palette (per lab convention).

## Rules

- **Never fabricate stats.** Every number in the manuscript must trace back to a script in `analysis/` or a file in `data/processed/`. When the atlas regenerates, re-run the analysis scripts before re-compiling the paper.
- **LaTeX lives under `manuscript/`** — never at project root (`~/.claude/CLAUDE.md` rule).
- **No `ast.literal_eval`** — use DuckDB / pandas native reads (`~/.claude/CLAUDE.md` rule).
- **Writing style**: when writing prose, read `~/.claude/writing-style-guide.md` first.
