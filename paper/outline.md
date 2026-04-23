# Paper outline

Target venue: **Transportation Research Part A: Policy and Practice**.
Typical TR-A bibliometric paper length: 10–15k words including refs; we aim
for **~10k body + ~2k appendix**.

---

## §1 Introduction  (~1,200 words)

- Motivation: a decade has passed since Sun & Rahwan (2017) mapped the
  coauthorship structure of transportation research. Team size has grown,
  the field has absorbed ML / CAV / decarbonisation, and paper-embedding
  models (SPECTER2) now let us measure *topical* proximity that
  coauthorship alone cannot see.
- Research questions:
  - **RQ1** How has transportation research's scale and collaboration
    topology evolved since Sun & Rahwan (2017)?
  - **RQ2** Do collaboration communities (Leiden on coauthor edges) and
    topic communities (Leiden on semantic-kNN edges) agree, and where do
    they diverge?
  - **RQ3** Can paper embeddings predict *latent* coauthorships —
    collaborations that hadn't happened by a cutoff year Y but materialise
    in years Y+1…Y+N?
- Contributions (see `contributions.md` for the explicit list).
- Atlas as companion: open, interactive, reproducible.

## §2 Related work  (~800 words)

- Sun & Rahwan 2017 — the seed paper; 20 venues, 1990–2015.
- Jiang, Bhat & Lam 2020 — TR-B 40-year bibliometric (VOSviewer).
- Modak et al. 2019 — Fifty years of TR series, TR Part A 120, 188–223.
- Newman 2001 — foundational coauthorship-network paper.
- SPECTER (Cohan et al. 2020) + SPECTER2 (Singh et al. 2023) for paper
  embeddings.
- Arora et al. 2017 "A Simple but Tough-to-Beat Baseline for Sentence
  Embeddings" — all-but-the-top whitening.
- Traag et al. 2019 (Leiden), Mucha et al. 2010 (multiplex communities),
  Lancichinetti & Fortunato 2009 (community-detection comparison).
- Fortunato & Hric 2016 (community detection review).
- van Eck & Waltman 2014 (VOSviewer) — the default bibliometric tool;
  contrast with our approach (Python-first, open artifacts).

## §3 Data  (~1,000 words)

- Venue selection: 29 journals / magazines / symposium. ISSN table
  (Appendix A1) with first-year / coverage-% columns.
- Ingest: OpenAlex primary; IEEE Xplore for IEEE venues; Elsevier
  ScienceDirect for TR-family (**local only**, not redistributed, per
  licence — match Sun & Rahwan's WoS-only approach but with open-data).
- Dedup: DOI first; fuzzy `rapidfuzz.ratio > 95` on normalised title
  AND matching year AND overlapping author surname set.
- Author disambiguation: OpenAlex author ID → ORCID → normalised name,
  with manual alias resolution for known splits (stored in
  `config/pipeline.yaml`; auto-detected ORCID splits in
  `data/interim/author_aliases_auto.json`).
- Citation counts: `max(OpenAlex, Crossref is-referenced-by-count)` —
  OpenAlex aggregates S2 + MAG (higher for ~64k papers), Crossref
  matches Elsevier display (higher for ~8k papers, including TR-C and
  T-ITS flagship works with historical OpenAlex undercounts).
- SPECTER2 embedding: title + abstract → 768-d. Post-process: subtract
  mean; remove top-1 PC (Arora 2017); z-score per dim. Concatenate with
  concept-TF-IDF (128-d TruncatedSVD on OpenAlex concepts ≥ L2) and
  venue-LDA (28-d). Final hybrid dim = 925.
- Table 1: corpus summary (29 venues × 60 years).

## §4 Descriptive bibliometrics  (~1,800 words)

- **Table 2** (expanded Sun & Rahwan Table 2): per-venue — papers,
  single-authored %, unique authors, avg authors / paper, max authors,
  collaborations, papers / author, avg citations, median career length
  of top-50 contributors.
- **Fig 1** — Papers-per-year, stacked by venue, 1967–2026.
- **Fig 2** — Avg authors-per-paper over time, by venue. Expect
  monotonic rise; Sun & Rahwan showed this up to 2015, we extend to
  2026 and show the ML-driven acceleration post-2018.
- **Fig 3** — Lotka's law productivity distribution (log-log). Compare
  exponent α to Lotka's α=2 prediction.
- **Table 3** — Top-10 contributors per venue (papers). Mirror our
  `venues.html` expandable panels.
- **Table 4** — Top-10 most-cited papers per venue.
- Discussion: big-journal effect (TRR 38k papers > all others combined
  pre-2010), publisher differences, emerging venues (OJ-ITS, TRIP,
  CommTR).

## §5 Coauthor network structure  (~1,500 words)

- **Fig 4** — Coauthor network, colored by Leiden community (142 of
  them; giant component screenshot of the atlas at a fixed zoom).
- **Table 5** — Top-20 communities by size + keyword labels +
  exemplar authors.
- **Table 6** — Top-30 authors by centrality (strength, betweenness,
  eigenvector) with affiliation.
- **Fig 5** — Degree / strength distribution; largest-component
  fraction over time.
- **Fig 6** — Bridge edges: top-100 cross-community ties by
  edge-betweenness. Who's bridging ML↔safety, optimisation↔behaviour?
- Discussion: Sun & Rahwan identified ~20 communities in 2017; we
  now see 142 (with tiny-island merge into `misc`). Interpret growth.

## §6 Semantic structure  (~1,200 words)

- **Fig 7** — SPECTER2 author centroids in UMAP 2D, colored by
  semantic-Leiden community (23).
- **Table 7** — Semantic communities with keyword labels (pavement,
  CAV, demand modelling, transit, safety, logistics, …) and top authors.
- **Ablation** (appendix): abstract-only SPECTER2 vs our hybrid
  (SPECTER + concept TF-IDF + venue LDA). Show modularity / silhouette
  improves with hybrid.
- **Whitening impact**: before/after median pairwise cosine (0.98 →
  0.69 after whitening), with illustrative pair examples.

## §7 Multiplex: where coauthorship and topic diverge  (~1,200 words)

- Combined-Leiden on multiplex edges (coauthor weighted + semantic
  mutual-top-K at sim ≥ 0.6). 156 communities.
- **Fig 8** — Sankey: which semantic communities flow into which
  coauthor communities, and where they mismatch.
- **Table 8** — Partition alignment: pairwise NMI, VI, and ARI for
  {coauthor-Leiden, semantic-Leiden, combined-Leiden}.
- **Case studies** — 3 communities where coauthor and semantic
  diverge (e.g., "pavement" is topically tight but spans several
  geographically isolated coauthor cliques).
- Discussion: collaboration follows geography and advisor lineage;
  topic follows methods. Multiplex reveals where the two decouple.

## §8 Phantom collaborators — a predictive test  (~1,800 words)

The central novel contribution.

- **Definition**: for author *a* at year *Y*, a phantom is an
  author *b* such that (i) *b* is among *a*'s top-K semantic
  nearest neighbours using the Y-train author centroids, (ii) *b*
  is ≥ 3 hops from *a* in the year-Y coauthor graph.
- **Experiment**: train cutoff Y = 2019; holdout 2020–2026.
- **Baselines**:
  - Random same-year-active pair.
  - Configuration-model random (degree-preserving).
  - Same-venue neighbour (top-K authors publishing in overlapping venues
    who haven't coauthored).
- **Metrics**: precision@K (K ∈ {5, 10, 20}), recall@K, lift over each
  baseline, calibration curve (semantic-similarity bucket vs
  realised-coauthorship rate).
- **Fig 9** — Precision@K curve, phantom vs baselines.
- **Fig 10** — Calibration plot.
- **Table 9** — Lift table: phantom / random, phantom / config, phantom / same-venue.
- **Case studies** — phantoms that materialised (e.g., an ML-CAV
  researcher and a traffic-flow theorist who coauthored for the first
  time in 2023).
- Discussion: semantic similarity captures latent opportunity beyond
  what venue overlap or degree preservation does; deploy as a
  recommender?

## §9 Author topic trajectories  (~1,000 words)

- 5-year UMAP bins per author with ≥ 2 papers in at least 2 bins.
- Drift vector per author: centroid-to-centroid displacement in hybrid
  embedding space, normalised.
- **Taxonomy**: stayer / drifter / pivoter, via drift-magnitude +
  direction-consistency clustering.
- **Fig 11** — Stayer / drifter / pivoter examples (3 named authors).
- **Table 10** — Correlates: taxonomy class × career stage × citation
  impact × centrality.
- **Fig 12** — Heatmap of topic-drift directions (PC1 × PC2 of
  displacement vectors).

## §10 The Transport Atlas tool  (~500 words)

- Screenshots: explorer, coauthor network, topic space, combined.
- Tech stack: Python 3.11, DuckDB + Parquet, Leiden, D3, Tabulator,
  Chart.js. No Elsevier full-text shipped.
- Data-source attribution (CC0 OpenAlex + Crossref).
- Reproducibility: `https://github.com/UMN-Choi-Lab/transport-atlas`.

## §11 Discussion  (~1,200 words)

- Limitations:
  - OpenAlex coverage gaps for pre-2000 papers — some venues 40–80%
    coverage before 1990 (Appendix B).
  - SPECTER2 is English-biased; non-English transportation work
    (Chinese-language journals, e.g.) is out of scope.
  - Author disambiguation still has residual homonym errors (e.g.,
    the "Sun, Lijun" 5-way split we caught in dev); manual alias file
    will never be complete.
  - Phantom recommender risks amplifying existing biases (high-status
    authors get more semantic neighbours); equity-of-exposure audit
    belongs in future work.
  - Semantic structure is snapshot — SPECTER2 will drift as the
    upstream model updates. We pin a model version in Appendix C.
- Policy implications:
  - Conference organisers can use phantom lists to target panel
    composition.
  - Funders can use bridge-edge analysis to reward cross-community
    collaboration.
- Future work: multilingual corpus (Chinese, Japanese, Korean TR
  journals), temporal community evolution, direct author
  recommendation deployment.

## §12 Conclusion  (~300 words)

- Recap: 111k papers, 29 venues, 60 years. Semantic-layer adds
  dimensions Sun & Rahwan couldn't see. Phantoms work as a predictor.
- The atlas is live and open; follow-on work invited.

## Appendices

- **A** Venue ISSNs + first-year + coverage %.
- **B** OpenAlex coverage by decade × venue.
- **C** Models / hyperparameters / seeds.
- **D** Author-disambiguation audit: 20 manual examples.
- **E** Phantom-collaborator case studies, full table.
