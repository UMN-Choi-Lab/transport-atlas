# Related work — annotated bibliography

Running list. Grouped by theme. **[cite-as]** = the key used in `refs.bib`.
One-line note = why we cite.

## 1. Seed paper and transportation bibliometrics

- **Sun & Rahwan 2017**. *Coauthorship network in transportation research.*
  TR-A 100: 135–151. **[sun2017coauthorship]** —
  Seed paper. 20 venues, 1990–2015. Name-correction algorithm, Louvain
  communities, centrality → citation regression. Their Fig 1 is
  a layout we replicate as Fig 4. Their Table 2 is what our §4 Table 2
  extends.
- **Jiang, Bhat & Lam 2020**. *A bibliometric overview of Transportation
  Research Part B: Methodological in the past forty years (1979–2019).*
  TR-B 138: 268–291. **[jiang2020trb]** — TR-B single-venue deep dive.
  VOSviewer-based. We cite for methodology baseline and comparison.
- **Modak, Merigó, Weber, Manzor & Ortúzar 2019**. *Fifty years of
  Transportation Research journals: A bibliometric overview.* TR-A 120:
  188–223. **[modak2019fifty]** — Covers 7 TR-series journals. Our 29-venue
  scope strictly supersedes.
- **Xu, Kitsos, Hrebenyk, Manchester-Evans 2022**. *Transportation Research
  Part E-logistics and transportation review: 25 years in retrospect.*
  TR-E 161. **[xu2022tre]** — TR-E-only equivalent; method comparison.
- **Mishra & Mohan 2018**. *Evolution of research on Accident Analysis &
  Prevention: bibliometric review.* **[mishra2018aap]** — AAP-focused;
  safety-subfield perspective.

## 2. Coauthorship-network methodology

- **Newman 2001**. *The structure of scientific collaboration networks.*
  PNAS 98(2): 404–409. **[newman2001structure]** — Canonical methodology.
- **Glänzel & Schubert 2005**. *Analysing scientific networks through
  co-authorship.* In *Handbook of Quantitative Science and Technology
  Research*. **[glanzel2005analysing]** — Review of methods.
- **Fortunato & Hric 2016**. *Community detection in networks: A user
  guide.* Physics Reports 659: 1–44. **[fortunato2016community]** —
  Community-detection review; used to justify Leiden over Louvain.

## 3. Leiden and multiplex community detection

- **Traag, Waltman & van Eck 2019**. *From Louvain to Leiden: guaranteeing
  well-connected communities.* Scientific Reports 9: 5233.
  **[traag2019leiden]** — The algorithm we use.
- **Mucha, Richardson, Macon, Porter & Onnela 2010**. *Community structure
  in time-dependent, multiscale, and multiplex networks.* Science 328:
  876–878. **[mucha2010community]** — Foundational for §7's multiplex
  combined-Leiden.
- **Lancichinetti & Fortunato 2009**. *Community detection algorithms: a
  comparative analysis.* PRE 80(5): 056117. **[lancichinetti2009community]**
  — Comparison baseline for our Leiden choices.

## 4. Paper embeddings and topic modelling

- **Cohan, Feldman, Beltagy, Downey & Weld 2020**. *SPECTER: Document-level
  representation learning using citation-informed transformers.* ACL
  2020. **[cohan2020specter]** — v1 of SPECTER; motivation for v2.
- **Singh, D'Arcy, Cohan, Downey & Feldman 2023**. *SciRepEval: A
  multi-format benchmark for scientific document representations.*
  EMNLP 2023. **[singh2023specter2]** — SPECTER2 paper; our embedding
  choice.
- **Arora, Liang & Ma 2017**. *A simple but tough-to-beat baseline for
  sentence embeddings.* ICLR 2017. **[arora2017simple]** — Justifies our
  all-but-the-top whitening.
- **Grootendorst 2022**. *BERTopic: neural topic modeling with a class-based
  TF-IDF procedure.* arXiv:2203.05794. **[grootendorst2022bertopic]** —
  Alternative topic-modelling approach we could contrast with.

## 5. OpenAlex / Crossref / citation data

- **Priem, Piwowar & Orr 2022**. *OpenAlex: A fully-open index of scholarly
  works, authors, venues, institutions, and concepts.* arXiv:2205.01833.
  **[priem2022openalex]** — Primary data source.
- **Hendricks, Tkaczyk, Lin & Feeney 2020**. *Crossref: the sustainable
  source of community-owned scholarly metadata.* Quantitative Science
  Studies 1(1): 414–427. **[hendricks2020crossref]** — Citation-count
  source for §3.
- **Visser, van Eck & Waltman 2021**. *Large-scale comparison of
  bibliographic data sources: Scopus, Web of Science, Dimensions,
  Crossref, and Microsoft Academic.* Quantitative Science Studies 2(1):
  20–41. **[visser2021comparison]** — Justifies our max(OA, CR) merge.

## 6. Lotka's law and productivity

- **Lotka 1926**. *The frequency distribution of scientific productivity.*
  J. Washington Academy of Sciences 16(12): 317–323. **[lotka1926frequency]**
  — Cited for Prediction 1b.

## 7. VOSviewer and bibliometric visualisation

- **van Eck & Waltman 2010**. *Software survey: VOSviewer, a computer
  program for bibliometric mapping.* Scientometrics 84: 523–538.
  **[vaneck2010vosviewer]** — What Jiang et al. used; we contrast
  with our Python-first open pipeline.
- **Waltman, van Eck & Noyons 2010**. *A unified approach to mapping and
  clustering of bibliometric networks.* J. Informetrics 4(4): 629–635.
  **[waltman2010unified]** — Algorithmic basis of VOSviewer.

## 8. Author disambiguation

- **Ferreira, Gonçalves & Laender 2012**. *A brief survey of automatic
  methods for author name disambiguation.* SIGMOD Record 41(2): 15–26.
  **[ferreira2012brief]** — Survey reference for §3.
- **Tang, Fong, Wang & Zhang 2012**. *A unified probabilistic framework
  for name disambiguation in digital library.* TKDE 24(6): 975–987.
  **[tang2012unified]** — Alternative approach; we cite for comparison.

## 9. Link prediction (for §8)

- **Liben-Nowell & Kleinberg 2007**. *The link-prediction problem for
  social networks.* JASIST 58(7): 1019–1031. **[libennowell2007link]** —
  The canonical link-prediction setup. Our §8 is a link-prediction problem
  with semantic features.
- **Martínez, Berzal & Cubero 2016**. *A survey of link prediction in
  complex networks.* ACM Computing Surveys 49(4): 69.
  **[martinez2016survey]** — Survey reference for §8 baselines.
- **Backstrom & Leskovec 2011**. *Supervised random walks: predicting and
  recommending links in social networks.* WSDM 2011.
  **[backstrom2011supervised]** — Strong-baseline alternative we don't
  run but cite.

## 10. Equity / bias in recommender + collaboration systems

- **Sugimoto, Robinson-García, Murray, Yegros-Yegros, Costas & Larivière
  2017**. *Scientists have most impact when they're free to move.*
  Nature 550: 29–31. **[sugimoto2017scientists]** — Cited in §11 policy
  discussion.
- **Chakraborty, Tummala & Patro 2019**. *Equality of voice: towards
  fair representation in crowdsourced top-K recommendations.* FAT*
  2019. **[chakraborty2019equality]** — Equity-of-exposure reference.
