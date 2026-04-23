# Data plan — tables and figures

Single source of truth for what gets produced and by which script.
When a row says **"done"**, the artifact already exists in `figures/` or
`tables/` and is referenced in the manuscript.

## Conventions

- **Figures**: matplotlib, 300 DPI, both PDF and PNG, Okabe-Ito colorblind-
  friendly palette. File naming: `figures/{section}_{fig}_{slug}.pdf|png`.
- **Tables**: `booktabs` LaTeX, `\toprule`/`\midrule`/`\bottomrule`,
  numeric columns `S[table-format=...]` (siunitx), tabular-nums for
  any in-text numbers. File naming: `tables/{section}_{slug}.tex`.
- **Pinned snapshot date**: 2026-04-22 (the commit SHA of
  `UMN-Choi-Lab/transport-atlas` at time of draft freeze; finalise
  before submission).

## §3 Data

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| T1 | Corpus summary (29 venues × 60 years) | `analysis/01_descriptive_tables.py` | `data/interim/papers.parquet`, `config/venues.yaml` | `tables/03_corpus_summary.tex` | pending |
| TA1 | Venue ISSN + first-year + coverage (Appendix A) | `analysis/01_descriptive_tables.py` | `config/venues.yaml`, `data/interim/papers.parquet` | `tables/app_a_venues.tex` | pending |

## §4 Descriptive bibliometrics

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| T2 | Per-venue stats (Sun & Rahwan Table 2, expanded) | `analysis/01_descriptive_tables.py` | `data/processed/venue_stats.json` | `tables/04_venue_stats.tex` | pending |
| T3 | Top-10 contributors per venue | `analysis/01_descriptive_tables.py` | `data/processed/venue_stats.json` | `tables/04_top_contributors.tex` | pending |
| T4 | Top-10 most-cited papers per venue | `analysis/01_descriptive_tables.py` | `data/processed/venue_stats.json` | `tables/04_top_papers.tex` | pending |
| F1 | Papers-per-year stacked by venue, 1967–2026 | `analysis/01_descriptive_tables.py` | `data/interim/papers.parquet` | `figures/04_papers_by_year_stacked.pdf` | pending |
| F2 | Avg authors/paper over time, by venue | `analysis/01_descriptive_tables.py` | `data/interim/papers.parquet` | `figures/04_team_size_over_time.pdf` | pending |
| F3 | Lotka's law productivity (log-log) | `analysis/01_descriptive_tables.py` | `data/interim/authors.parquet` | `figures/04_lotka_productivity.pdf` | pending |

## §5 Coauthor network structure

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| F4 | Coauthor network screenshot (Leiden-colored) | atlas export, one-shot | live site screenshot | `figures/05_coauthor_network.pdf` | pending |
| T5 | Top-20 coauthor communities (size + keywords + exemplars) | `analysis/02_coauthor_structure.py` | `data/processed/coauthor_network.json` | `tables/05_top_communities.tex` | pending |
| T6 | Top-30 authors by centrality | `analysis/02_coauthor_structure.py` | `data/processed/coauthor_network.json` | `tables/05_top_centrality.tex` | pending |
| F5 | Degree/strength distribution; giant-component fraction over time | `analysis/02_coauthor_structure.py` | `data/processed/coauthor_network.json`, `data/interim/papers.parquet` | `figures/05_degree_distribution.pdf` | pending |
| F6 | Bridge edges: top-100 cross-community ties | `analysis/02_coauthor_structure.py` | `data/processed/coauthor_network.json` | `figures/05_bridges.pdf` | pending |

## §6 Semantic structure

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| F7 | Author centroids in UMAP 2D, semantic-Leiden colored | `analysis/03_partition_alignment.py` (pass 1: UMAP plot) | `data/processed/topic_coords.json` | `figures/06_topic_space.pdf` | pending |
| T7 | Semantic communities (23) with labels + top authors | from existing `semantic_communities.json` | `data/processed/semantic_communities.json` | `tables/06_semantic_communities.tex` | pending |
| F-app-ablation | SPECTER2-only vs hybrid comparison: modularity + silhouette | **new analysis required** | paper embeddings cache | `figures/app_ablation_hybrid.pdf` | pending (stretch) |
| F-whitening | Pairwise-cosine histogram before/after whitening | one-off diagnostic | paper embeddings cache | `figures/06_whitening_impact.pdf` | pending |

## §7 Multiplex: where coauthorship and topic diverge

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| F8 | Sankey — semantic → coauthor community flow | `analysis/03_partition_alignment.py` | `data/processed/coauthor_network.json`, `topic_coords.json`, `semantic_communities.json` | `figures/07_sankey_coauthor_semantic.pdf` | pending |
| T8 | Partition alignment (NMI/VI/ARI × 3 pairs) | `analysis/03_partition_alignment.py` | same | `tables/07_partition_alignment.tex` | pending |

## §8 Phantom collaborators (**core contribution**)

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| — | Train-cutoff phantom eval dataset (Y=2019) | `scripts/07_phantom_eval.py` | `/data2/.../paper_embeddings.parquet`, `data/interim/papers.parquet` | `data/processed/phantom_eval.json` | **pending — new pipeline work** |
| F9 | Precision@K phantom vs baselines | `analysis/04_phantom_eval.py` | `data/processed/phantom_eval.json` | `figures/08_phantom_precision_at_k.pdf` | pending |
| F10 | Calibration plot (similarity bucket × realized rate) | `analysis/04_phantom_eval.py` | `data/processed/phantom_eval.json` | `figures/08_phantom_calibration.pdf` | pending |
| T9 | Lift table (phantom / random / config / same-venue) | `analysis/04_phantom_eval.py` | `data/processed/phantom_eval.json` | `tables/08_phantom_lift.tex` | pending |
| T10 | Phantom case studies (10 materialised pairs) | `analysis/04_phantom_eval.py` | `data/processed/phantom_eval.json`, `data/interim/papers.parquet` | `tables/08_phantom_cases.tex` | pending |

## §9 Trajectory taxonomy

| ID | Artifact | Source script | Input | Output | Status |
|---|---|---|---|---|---|
| F11 | Stayer / drifter / pivoter example trajectories | `analysis/05_trajectory_taxonomy.py` | `data/processed/author_trajectories.json`, `topic_coords.json` | `figures/09_trajectory_examples.pdf` | pending |
| T10b | Taxonomy × career stage × citation × centrality | `analysis/05_trajectory_taxonomy.py` | above + `coauthor_network.json` | `tables/09_trajectory_taxonomy.tex` | pending |
| F12 | Heatmap of topic-drift PC1×PC2 | `analysis/05_trajectory_taxonomy.py` | `data/processed/author_trajectories.json` | `figures/09_drift_pca.pdf` | pending |

## §10 Atlas

| ID | Artifact | Source | Output | Status |
|---|---|---|---|---|
| F-atlas-1 | Explorer screenshot | atlas, zoomed on query | `figures/10_atlas_explorer.pdf` | pending |
| F-atlas-2 | Coauthor-network screenshot | atlas | `figures/10_atlas_coauthor.pdf` | pending |
| F-atlas-3 | Topic-space screenshot | atlas | `figures/10_atlas_topic.pdf` | pending |
| F-atlas-4 | Combined-view screenshot | atlas | `figures/10_atlas_combined.pdf` | pending |

## Appendices

| ID | Artifact | Source script | Output | Status |
|---|---|---|---|---|
| B | OpenAlex coverage by decade × venue | `analysis/01_descriptive_tables.py` | `tables/app_b_coverage.tex` | pending |
| C | Models / hyperparameters / seeds | hand-written | `manuscript/sections/app_c_hyperparams.tex` | pending |
| D | Author-disambiguation audit (20 examples) | `analysis/06_disambig_audit.py` (pending) | `tables/app_d_disambig.tex` | pending |
| E | Full phantom-collaborator case list | `analysis/04_phantom_eval.py` | `tables/app_e_phantom_full.tex` | pending |

## Running total

| Section | Tables | Figures | Appendices |
|---|---|---|---|
| §3 Data | 1 | 0 | 1 (A) |
| §4 Descriptive | 3 | 3 | 1 (B) |
| §5 Coauthor | 2 | 3 | 0 |
| §6 Semantic | 1 | 2 | 0 |
| §7 Multiplex | 1 | 1 | 0 |
| §8 Phantom | 2 | 2 | 1 (E) |
| §9 Trajectories | 1 | 2 | 0 |
| §10 Atlas | 0 | 4 | 0 |
| Hyperparameters | 0 | 0 | 1 (C) |
| Disambig audit | 0 | 0 | 1 (D) |
| **Total** | **11** | **17** | **5** |
