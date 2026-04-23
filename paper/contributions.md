# Contributions

Explicit contribution list for the paper, contrasted with Sun & Rahwan (2017)
and the other recent transportation bibliometric surveys.

| # | Contribution | First in transportation bibliometrics? | Why it matters |
|---|---|---|---|
| C1 | **10× the corpus** — 111,057 papers (Sun & Rahwan: ~11k; Modak et al. 2019 "50 years": ~24k; Jiang et al. 2020 TR-B: 2,697). | Yes, to our knowledge. | Sun & Rahwan's findings become *trends* once re-measured a decade later at ~10× scale. |
| C2 | **Open-data pipeline** on CC0 sources (OpenAlex + Crossref). | Yes — all previous transportation bibliometric papers used WoS or Scopus (paywalled). | Fully reproducible; no licence friction. Sun & Rahwan used WoS; Jiang et al. used WoS Core Collection. |
| C3 | **Author disambiguation via OpenAlex ID + ORCID + ORCID-split auto-detection**, not just name-normalisation. | Substantively improves on Sun & Rahwan's "name correction algorithm" and Jiang et al.'s manual per-name fixes. | Residual homonym error is visible and quantifiable (Appendix D). |
| C4 | **SPECTER2 paper embeddings + hybrid (whitening + concept TF-IDF + venue LDA)** for topic space. | First use of contemporary paper embeddings on the transportation research corpus. | Enables §6/§7/§8 analyses that coauthorship-only methods (Sun & Rahwan, Jiang et al.) structurally cannot do. |
| C5 | **Multiplex Leiden community detection** combining coauthor edges and semantic k-NN edges, with NMI/VI alignment vs single-layer partitions. | First application in transportation. | Reveals where collaboration and topic structure decouple — neither layer alone tells the full story. |
| C6 | **Phantom-collaborator framework with temporal-holdout predictive validation.** | **This is the paper's core novel contribution.** | Shows paper embeddings carry signal about *future* coauthorships, not just current topical similarity. Publishable as its own methodological lesson. |
| C7 | **Per-author 5-year topic trajectories** in a shared embedding space, with trajectory-based taxonomy (stayer / drifter / pivoter). | First in transportation bibliometrics; prior work (Kotkov 2022, etc.) covered CS. | Gives a concrete lens on how individual researchers move through the field. |
| C8 | **Citation source reconciliation**: `max(OpenAlex, Crossref)` with documented per-paper source. | Novel contribution — shows Crossref ≫ OpenAlex for 8k flagship Elsevier/IEEE papers, OpenAlex ≫ Crossref for 64k papers (S2/MAG aggregation). | Documents a data-quality problem other citation studies silently inherit. |
| C9 | **Live public atlas** (`choi-seongjin.github.io/transport-atlas`) with all five views (Explorer, Papers by Year, Coauthor Network, Topic Space, Combined). | First interactive, public atlas for the field. | Research artifact as deliverable; invites community contribution. |

## Comparison table with prior art

| Property | Sun & Rahwan 2017 | Jiang et al. 2020 | Modak et al. 2019 | **This paper** |
|---|---|---|---|---|
| Venues | ~20 | 1 (TR-B) | 7 (TR series) | **29** |
| Papers | ~11k | 2,697 | ~24k | **111,057** |
| Year range | 1990–2015 | 1979–2019 | 1967–2017 | **1967–2026** |
| Data source | WoS | WoS | WoS + Scopus | **OpenAlex + Crossref (CC0)** |
| Author disambiguation | Name-correction algorithm + manual | Manual | Unclear | **OpenAlex ID + ORCID + auto splits** |
| Paper embeddings | No | No | No | **SPECTER2 hybrid** |
| Community detection | Louvain on coauthor | VOSviewer clustering | VOSviewer | **Leiden × 3 (coauthor / semantic / multiplex)** |
| Temporal holdout prediction | No | No | No | **Yes — phantom collaborators** |
| Trajectories | No | No | No | **Yes — 5-year UMAP bins** |
| Live tool | No | No | No | **Yes — atlas web app** |

## What we deliberately do **not** claim

- Not a causal analysis of what drives collaboration.
- Not a performance benchmark of bibliometric algorithms (we use off-the-
  shelf Leiden + SPECTER2; our novelty is application and the phantom
  experiment).
- Not a Chinese-language or other non-English corpus study (limitations, §11).
- Not a recommender deployment — we validate predictive signal but don't
  ship a system (future work).
