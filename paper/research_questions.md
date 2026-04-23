# Research questions

Each RQ is paired with (a) a **falsifiable prediction**, (b) the **statistic
or artifact** that decides, (c) the **analysis script** that computes it.

---

## RQ1 — How has transportation research's scale and collaboration topology evolved since Sun & Rahwan (2017)?

**Prediction 1a.** Average authors-per-paper has increased monotonically
across every major venue since 2015, with a visible acceleration post-2018
(driven by ML/CAV waves).
- **Decided by**: per-venue avg-authors-per-paper time series; slope pre- vs
  post-2018.
- **Source**: `paper/analysis/01_descriptive_tables.py` → `figures/team_size_over_time.pdf`.
- **Falsifier**: if any of the top-5 venues shows a flat or decreasing slope
  post-2015, we note and discuss.

**Prediction 1b.** Productivity follows Lotka's law (~2.0 exponent) but with
a heavier tail than Sun & Rahwan found in 2017 (their α ≈ 2.6).
- **Decided by**: OLS log-log fit of author-count vs papers-published.
- **Source**: `paper/analysis/01_descriptive_tables.py` → `figures/lotka_productivity.pdf`.

**Prediction 1c.** The giant component of the coauthor graph now covers a
*larger* fraction of authors than in Sun & Rahwan (≥ 70% vs their 58%).
- **Decided by**: largest-component / total-nodes ratio in the ≥ 2-collab
  graph.
- **Source**: `paper/analysis/02_coauthor_structure.py` → text statistic.

---

## RQ2 — Do collaboration and topic communities agree, and where do they diverge?

**Prediction 2a.** Semantic-Leiden and coauthor-Leiden partitions have
moderate agreement (NMI ∈ [0.3, 0.55]). Perfect agreement (NMI > 0.8) would
mean paper embeddings add nothing beyond what coauthorship already encodes;
no agreement (NMI < 0.15) would mean the semantic signal is noise.
- **Decided by**: NMI / VI / ARI between three partitions.
- **Source**: `paper/analysis/03_partition_alignment.py` → `tables/partition_alignment.tex`.

**Prediction 2b.** Divergence concentrates in *methodologically broad* topics
— e.g., pavement engineering fragments across geographically separate
coauthor cliques (Chinese, US, European), while forming a single semantic
community.
- **Decided by**: per-semantic-community entropy over coauthor communities;
  top-3 most-fragmented semantic communities.
- **Source**: `paper/analysis/03_partition_alignment.py` → `figures/sankey_coauthor_semantic.pdf`.

**Prediction 2c.** Combined/multiplex Leiden communities are strictly
*finer* than coauthor-Leiden on average (the semantic edges break up broad
coauthor clusters that span topics).
- **Decided by**: median community size: combined < coauthor.
- **Source**: `paper/analysis/03_partition_alignment.py` → text statistic.

---

## RQ3 — Can paper embeddings predict *latent* coauthorships?

**Definition (recap).** Phantom(a, b, Y) ≡ b is a top-K semantic neighbour
of a at year Y, and coauthor-dist(a, b, Y) ≥ 3 hops.

**Prediction 3a.** Phantom pairs become real coauthors in the 2020–2026
holdout at a rate at least **3× higher** than random same-active pairs.
- **Decided by**: lift = P(coauthor | phantom) / P(coauthor | random).
- **Source**: `scripts/07_phantom_eval.py` → `data/processed/phantom_eval.json`
  → `paper/analysis/04_phantom_eval.py` → `tables/phantom_lift.tex`.
- **Falsifier**: lift < 1.5 over all three baselines → we report negative
  result, reframe §8 as "semantic similarity is NOT a predictor beyond
  degree preservation" (still publishable but a different paper).

**Prediction 3b.** Precision@K is monotonically decreasing in K but stays
above 5% at K=20 (vs < 1% baseline).
- **Decided by**: precision@K curve.
- **Source**: `paper/analysis/04_phantom_eval.py` → `figures/phantom_precision_at_k.pdf`.

**Prediction 3c.** Semantic-similarity buckets are well-calibrated with
realised-coauthorship rate (positive correlation, monotone).
- **Decided by**: Spearman ρ on 10-bucket calibration curve.
- **Source**: `paper/analysis/04_phantom_eval.py` → `figures/phantom_calibration.pdf`.

---

## Secondary RQ — Trajectory taxonomy (§9, narrative support)

Not a primary RQ; descriptive and exploratory.

**Claim.** The population of transportation researchers splits into roughly
three archetypes by 5-year topic drift: **stayers** (small drift, persistent
focus), **drifters** (gradual angular drift, same broad topic), **pivoters**
(large cross-community jumps). Career stage and citation impact correlate:
pivoters tend to be junior (career ≤ 10y); stayers tend to be senior and
highly cited.

Decided by:
- k-means or GMM clustering on (drift-magnitude, direction-consistency);
  chosen K via silhouette.
- Cramér's V between taxonomy × career-stage bin.
- **Source**: `paper/analysis/05_trajectory_taxonomy.py`.
