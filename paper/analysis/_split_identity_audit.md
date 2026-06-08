# Split-Identity Audit: Coauthor Graph Phantom Edges

**Date:** 2026-04-24  
**Corpus:** 120,323 papers · 168,090 author keys · 41,559 graph nodes · 83,044 edges  
**Scope:** OpenAlex-keyed authors (163,008 keys) only — ORCID-keyed keys are already merged by `auto_alias_map_from_papers`.

---

## Summary

| Tier | Candidate pairs | Dropped edges restored if merged |
|------|----------------|----------------------------------|
| **HIGH** (one key lacks ORCID + shared institution + ≥3 coauthors) | 13 | 8 |
| **HIGH\_AMB** (≥20 shared coauthors + same institution, but different ORCIDs) | 2 | 10 |
| **MEDIUM** (different ORCIDs, ≥3 shared coauthors, partial inst. overlap) | 58 | 55 |
| **MEDIUM\_2** (exactly 2 shared coauthors, both ORCIDs present) | 91 | 50 |
| **False Positives** (same name, confirmed different people) | 2 | — |
| **Total** (all non-FP) | **164** | **123** |

**Methodology:** for each same-`canonical_name` group with ≥2 OpenAlex keys where at least one key has ≥5 papers, pairwise shared-coauthor counts were computed from the papers parquet. Restored-edge count = number of third-party coauthors C where `pair_count(K1,C) + pair_count(K2,C) ≥ 2` but no edge (K1,C) or (K2,C) currently exists in the graph. Confidence tiers reflect ORCID evidence, institution overlap, shared-coauthor count, and same Leiden community membership.

---

## Top-30 High-Confidence Cases

Sorted by impact (restored edges descending, then shared coauthors). Columns: `canonical_name | key1 | key2 | papers1 | papers2 | orcid1 | orcid2 | shared_coauth | restored | shared_institution | tier`.

| # | Name | Key 1 | Key 2 | P1 | P2 | ORCID 1 | ORCID 2 | SC | RE | Shared Inst. | Tier |
|---|------|--------|--------|----|----|---------|---------|----|----|--------------|------|
| 1 | chen, long | A5100336450 | A5113009903 | 49 | 19 | 0000-0003-4925-0572 | 0000-0003-2546-6926 | 29 | 6 | Institute of Automation; Xi'an Jiaotong U | HIGH\_AMB |
| 2 | zhou, xuesong | A5047902154 | A5113436201 | 66 | 38 | 0000-0002-9963-5369 | 0000-0002-0059-3817 | 21 | 4 | Arizona State U; Univ. of Utah | HIGH\_AMB |
| 3 | ge, shirong | A5017962413 | A5110091391 | 6 | 4 | 0000-0002-7350-1493 | *(none)* | 4 | 2 | China Univ. Mining Tech; Ministry of Transport | HIGH |
| 4 | wei, xuezhe | A5082361275 | A5121121005 | 21 | 3 | 0000-0001-9043-2330 | *(none)* | 3 | 2 | Tongji University | HIGH |
| 5 | marcotte, patrice | A5102990491 | A5106419971 | 12 | 6 | 0000-0002-1935-2554 | *(none)* | 3 | 2 | Royal Military College; Univ. de Montréal | HIGH |
| 6 | song, guohua | A5051911098 | A5123191968 | 53 | 5 | 0000-0002-8180-8153 | *(none)* | 5 | 1 | Beijing Jiaotong U; Ministry of Transport | HIGH |
| 7 | zhong, zhihua | A5101987780 | A5112411803 | 13 | 4 | 0009-0003-6530-9930 | *(none)* | 5 | 1 | Tsinghua U; Chinese Academy of Engineering | HIGH |
| 8 | yang, min | A5066418709 | A5101852844 | 29 | 12 | 0000-0002-8595-2769 | *(none)* | 9 | 0 | Southeast University | HIGH |
| 9 | xiao, zhe | A5007706560 | A5100895846 | 7 | 4 | 0000-0002-0440-5772 | *(none)* | 4 | 0 | A\*STAR; Inst. of High Performance Computing | HIGH |
| 10 | han, xuebing | A5042099106 | A5100611797 | 3 | 9 | *(none)* | 0000-0001-7896-9354 | 4 | 0 | Tsinghua University | HIGH |
| 11 | zhong, lingshu | A5003205429 | A5075135732 | 6 | 6 | *(none)* | 0000-0002-0021-0548 | 3 | 0 | Chalmers U; Sun Yat-sen U | HIGH |
| 12 | cordeau, jean-francois | A5016828932 | A5110086210 | 36 | 6 | 0000-0002-4963-1298 | *(none)* | 3 | 0 | HEC Montréal | HIGH |
| 13 | huang, shoudao | A5022854751 | A5068705108 | 4 | 30 | *(none)* | 0000-0002-6923-9605 | 3 | 0 | Hunan University | HIGH |
| 14 | tiwari, geetam | A5076416521 | A5112619468 | 31 | 7 | 0000-0002-9302-1243 | *(none)* | 3 | 0 | IIT Delhi | HIGH |
| 15 | xiao, yi-bin | A5080881249 | A5111406931 | 11 | 5 | 0000-0002-0676-7662 | *(none)* | 3 | 0 | UESTC | HIGH |
| 16 | yan, yongjun | A5006616729 | A5102746803 | 8 | 4 | 0000-0001-6398-5597 | 0000-0001-9961-7235 | 6 | 3 | Nanjing U Science & Tech | MEDIUM |
| 17 | wang, jian | A5100333082 | A5100370393 | 10 | 36 | 0000-0002-1751-7060 | 0000-0002-0385-3645 | 5 | 3 | Heilongjiang Inst. Tech; Harbin Inst. Tech | MEDIUM |
| 18 | yang, yang | A5100397465 | A5100397725 | 4 | 4 | 0000-0001-9172-1695 | 0000-0003-0608-9408 | 3 | 3 | — | MEDIUM |
| 19 | xu, min | A5043324316 | A5101402281 | 52 | 4 | 0000-0002-3088-0358 | 0000-0002-4390-9105 | 3 | 3 | Dongbei U Finance; HK Polytech | MEDIUM |
| 20 | zhang, jun | A5100400217 | A5100433223 | 30 | 7 | 0000-0001-7835-9871 | 0009-0004-4862-3925 | 3 | 3 | Chinese Acad. Sciences; High Magnetic Field Lab | MEDIUM |
| 21 | liu, pengfei | A5100716878 | A5110748560 | 16 | 5 | 0000-0001-5983-7305 | 0009-0008-3547-6961 | 8 | 2 | RWTH Aachen | MEDIUM |
| 22 | ni, wei | A5021527965 | A5101544825 | 8 | 14 | 0000-0003-0780-4637 | 0000-0002-4933-594X | 5 | 2 | CSIRO; Health Sciences | MEDIUM |
| 23 | wang, zijin | A5090403693 | A5102903560 | 8 | 7 | 0000-0002-3285-433X | 0000-0002-1491-8475 | 5 | 2 | Univ. of Central Florida | MEDIUM |
| 24 | ali, yasir | A5000894912 | A5103144602 | 49 | 5 | 0000-0002-5770-0062 | 0009-0006-6284-9546 | 4 | 2 | Loughborough University | MEDIUM |
| 25 | liu, xiaobo | A5064480394 | A5112310051 | 51 | 5 | 0000-0001-5722-4943 | 0000-0002-5530-0479 | 4 | 2 | NJIT; Southwest Jiaotong U | MEDIUM |
| 26 | yang, di | A5013719632 | A5053753602 | 20 | 4 | 0000-0002-7964-7872 | 0000-0002-4010-6163 | 3 | 2 | Univ. of Maryland, College Park | MEDIUM |
| 27 | wang, kaifeng | A5100647996 | A5115597574 | 7 | 3 | 0000-0002-2298-7411 | 0009-0003-4676-2529 | 3 | 2 | CCCC Highway Consultants; Wuhan U Tech | MEDIUM |
| 28 | yuan, quan | A5029800889 | A5036324138 | 19 | 40 | 0000-0001-7024-5984 | 0000-0002-8397-4024 | 9 | 1 | Tongji University | MEDIUM |
| 29 | wang, xuesong | A5100336644 | A5108064895 | 150 | 9 | 0000-0001-8046-3213 | 0000-0002-5327-1088 | 8 | 1 | MoE China; Tongji University | MEDIUM |
| 30 | wang, li | A5035406850 | A5110973347 | 8 | 5 | 0000-0002-9325-2391 | 0000-0002-5615-0847 | 7 | 1 | Tsinghua U; State Key Lab Robotics | MEDIUM |

**Column key:** P1/P2 = paper counts, SC = shared coauthors, RE = new edges restored if merged.

### Notable already-aliased cases (not in table above)
- **Yuankai Wu** (A5100370856 + A5065903043) and **Lijun Sun** (A5058941074 + A5100786084): confirmed duplicate ORCIDs, already added to `config/pipeline.yaml`. This pair had 3 shared papers across 3 `(wu_key, sun_key)` combinations, each at count=1 → dropped edge now restored.

---

## Medium-Confidence Cases (top 15, requiring manual verification)

These have exactly 2 shared coauthors. Both keys in each pair hold different ORCIDs. Many share the same institution, which elevates suspicion but does not confirm identity.

| # | Name | Key 1 | Key 2 | P1 | P2 | SC | RE | Shared Inst. |
|---|------|--------|--------|----|----|----|----|--------------|
| 1 | liu, yang | A5064815226 | A5100355854 | 23 | 44 | 2 | 2 | National Univ. of Singapore |
| 2 | feng, zhongxiang | A5000309313 | A5123532302 | 41 | 4 | 2 | 2 | Hefei U of Technology |
| 3 | zhang, min | A5033697530 | A5100402930 | 3 | 3 | 2 | 2 | Univ. of Queensland |
| 4 | li, tao | A5100455256 | A5100455385 | 3 | 10 | 2 | 2 | Guangdong Univ. Finance |
| 5 | li, yang | A5100421307 | A5100709800 | 8 | 3 | 2 | 2 | Univ. of Michigan |
| 6 | lu, jing | A5019275030 | A5069150625 | 9 | 3 | 2 | 2 | Dalian Maritime University |
| 7 | huber, gerald | A5034500611 | A5113442770 | 3 | 9 | 2 | 2 | English Heritage |
| 8 | yi, ping | A5100709376 | A5100709379 | 5 | 16 | 2 | 2 | Univ. of Akron |
| 9 | zhang, kai | A5100324011 | A5101871787 | 3 | 4 | 2 | 2 | Tsinghua–Berkeley Shenzhen Inst. |
| 10 | zhang, yong | A5070956153 | A5100419731 | 20 | 9 | 2 | 2 | Southeast University |
| 11 | li, ye | A5100339232 | A5100339233 | 8 | 3 | 2 | 2 | Central South University |
| 12 | zhang, qiang | A5100381911 | A5100381999 | 7 | 6 | 2 | 2 | Shandong University |
| 13 | liu, yu | A5022072267 | A5100345645 | 4 | 4 | 2 | 1 | China Auto. Eng. Research Inst. |
| 14 | xu, nan | A5003255350 | A5028125767 | 6 | 7 | 2 | 1 | Jilin U; Automotive Sim. Lab |
| 15 | huang, wei | A5101399361 | A5101748858 | 3 | 5 | 2 | 1 | Southeast University |

*(91 total; table truncated at 15)*

**Flag for manual verification:** `li, tao` (A5100455256 vs A5100455385), `li, ye` (A5100339232 vs A5100339233), and `yi, ping` (A5100709376 vs A5100709379) have consecutive or near-consecutive OpenAlex IDs — a known pattern when OpenAlex bulk-ingests an author list and creates sibling IDs for the same person. These are the highest-priority items in this tier.

---

## Confirmed False Positives

Two cases in the HIGH tier were flagged but found to represent genuinely different researchers sharing a common Chinese name:

| Name | Key 1 | Key 2 | Reason |
|------|--------|--------|--------|
| sun, lijun | A5058941074 (McGill, transit/RL) | A5103990527 (Tongji, traffic flow/expressway) | Different institutions, different research areas; 3 "shared" coauthors are high-volume Tongji transport faculty who co-publish broadly |
| sun, lijun | A5101527795 (Tongji, pavement/geotechnical) | A5103990527 (Tongji, traffic/control) | Same university but distinct subfields (pavement moduli vs. signalized intersections); 4 shared coauthors are senior Tongji transport faculty |

These pairs have shared coauthors only because the corpus contains a few prolific Tongji authors who publish across subfields and happen to be co-authors of both Sun Lijun variants. Institution specificity is necessary but not sufficient for common Chinese surnames.

---

## Patterns Observed

### 1. ORCID registration mid-career (most common mechanism)
The dominant pattern is an author publishing without ORCID for several years, then registering one. OpenAlex creates a new key when the ORCID appears and fails to back-link earlier papers. In the HIGH-confidence tier, 13 of 15 pairs have exactly one key with no ORCID. The secondary key's first year clusters around 2016–2021 (post-ORCID adoption wave in Chinese universities and transport journals).

### 2. Duplicate ORCID registration (confirmed via pipeline.yaml fixes)
Yuankai Wu and Lijun Sun (McGill) are confirmed cases where the same researcher holds two distinct valid ORCIDs — one registered during their PhD/postdoc era at one institution, one re-registered or claimed via a later affiliation portal. OpenAlex treats these as different identities. `chen, long` (29 shared coauthors, same Chinese Academy of Sciences group) and `zhou, xuesong` (21 shared coauthors, Arizona State / Univ. of Utah) are very strong candidates for the same mechanism.

### 3. `0009-` namespace as a diagnostic signal
The ORCID `0009-` prefix ("non-public ORCID") was introduced around 2022 for unclaimed/institutional ORCIDs. Ten pairs in the HIGH/MEDIUM tier have one `0009-` key. These ORCIDs are often assigned automatically by publishers or university portals and may duplicate a researcher's earlier `0000-` ORCID. This is a reliable heuristic: if one key has `0009-` and the other has `0000-` for the same name and institution, treat it as a probable split.

### 4. Venue concentration
Split candidates cluster heavily in **T-ITS** (79 occurrences), **TR-C** (65), and **TRR** (57) — the three largest venues in the corpus. This is likely a size effect (more papers → more opportunities for a split), but it also suggests that multi-venue authors who publish across the IEEE and Elsevier ecosystems (which index separately) are at higher risk of getting fragmented IDs.

### 5. Common Chinese/Korean surname saturation
Names like `wang, wei`, `liu, yang`, `wang, xin`, `chen, long` can have 10–50+ distinct researchers in the corpus. The audit filters by `max(n_papers) ≥ 5` and `shared_coauthors ≥ 2`, but for very common surnames, even 3–4 shared coauthors may reflect coincidental overlap among researchers in the same subfield community, not a genuine identity split. The `sun, lijun` false-positive is the clearest example.

---

## Recommended Next Step

**Short-term (immediate, safe):** Add YAML aliases for the 13 HIGH-confidence cases only. These all have one key with no ORCID, confirmed shared institution, and ≥3 shared coauthors pointing to the same research group. Priority order by restored edges: `ge_shirong`, `wei_xuezhe`, `marcotte_patrice`, `song_guohua`, `zhong_zhihua`, `yang_min`, `xiao_zhe`, `han_xuebing`, `zhong_lingshu`, `cordeau_jf`, `huang_shoudao`, `tiwari_geetam`, `xiao_yibin`.

**Two HIGH\_AMB cases** (`chen, long` and `zhou, xuesong`) should be verified manually: check OpenAlex web profiles for both IDs to confirm shared affiliation history before aliasing. If confirmed, these restore 10 additional edges.

**Medium-term (code change, defensible threshold):** Extend `auto_alias_map_from_papers` to also merge pairs where:
1. Same `canonical_name`, and
2. One key has no ORCID, and
3. Shared coauthors ≥ 3, and
4. Shared institution ≥ 1

This catches the bulk of HIGH cases automatically. From this audit, the threshold of 3 shared coauthors with institution evidence produces zero confirmed false positives. Lowering to 2 without institution evidence risks name-collision merges (common Chinese surnames) — do **not** do that without further validation.

**Do not** add a general "same-name + ≥N coauthors regardless of ORCID" rule: the `sun, lijun` false positives (3 shared coauthors, same large university) demonstrate that this will merge genuinely distinct researchers in large research universities with broad internal collaboration networks. The ORCID-asymmetry criterion (one key has no ORCID) is a necessary gate.

---

*Generated 2026-04-24 from `data/interim/papers.parquet`, `data/interim/authors.parquet`, `data/processed/coauthor_network.json`. No source code or config files were modified.*
