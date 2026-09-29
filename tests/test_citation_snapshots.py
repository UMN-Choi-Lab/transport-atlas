"""Offline unit tests for the as-of-cutoff citation arithmetic (Task: temporal
leakage fix).  Exercises the pure functions of scripts/fetch_citation_snapshots
— no network, no data/ access.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fetch_citation_snapshots import (  # noqa: E402
    ShardStore,
    asof_citations,
    build_id_map,
    snapshot_record,
    sum_counts_since,
    window_covers_cutoff,
)
from transport_atlas.process.dedupe import _paper_id  # noqa: E402


def _cby(pairs):
    return [{"year": y, "cited_by_count": c} for y, c in pairs]


def test_asof_subtracts_post_cutoff_years_only():
    # W2808845378-style: pub 2018, total 38, 34 citations in 2020+ -> asof 4
    cby = _cby([(2018, 1), (2019, 3), (2020, 10), (2023, 14), (2026, 10)])
    asof, clamped = asof_citations(38, cby, 2019)
    assert asof == 4 and not clamped


def test_asof_empty_counts_is_total_not_missing():
    # zero-citation work AND old-work-outside-window both -> asof == total
    assert asof_citations(0, [], 2019) == (0, False)
    assert asof_citations(10, None, 2019) == (10, False)
    # W609901179-style: sparse counts, 7 pre-window citations only in total
    asof, clamped = asof_citations(10, _cby([(2015, 1), (2024, 1), (2025, 1)]),
                                   2019)
    assert asof == 8 and not clamped


def test_asof_all_citations_post_cutoff():
    # W2990554003-style: pub 2019, all 58 citations 2020+ -> asof 0
    asof, clamped = asof_citations(58, _cby([(2020, 20), (2022, 38)]), 2019)
    assert asof == 0 and not clamped


def test_asof_clamps_negative_reconstruction():
    # async-update artifact: window sum exceeds the total -> clamp at 0 + flag
    asof, clamped = asof_citations(5, _cby([(2021, 9)]), 2019)
    assert asof == 0 and clamped


def test_asof_respects_other_cutoffs():
    cby = _cby([(2019, 2), (2020, 3), (2021, 4)])
    assert asof_citations(9, cby, 2018) == (0, False)   # 9 - (2+3+4)
    assert asof_citations(9, cby, 2020) == (5, False)   # 9 - 4
    assert sum_counts_since(cby, 2020) == 7


def test_sum_counts_ignores_malformed_rows():
    cby = [{"year": None, "cited_by_count": 5}, {"year": 2021},
           {"year": 2022, "cited_by_count": 3}]
    assert sum_counts_since(cby, 2020) == 3


def test_window_covers_cutoff_shelf_life():
    # 10-year guarantee covers 2020 while now <= 2029
    assert window_covers_cutoff(2026, 2019)
    assert window_covers_cutoff(2029, 2019)
    assert not window_covers_cutoff(2030, 2019)


def test_snapshot_record_from_response():
    work = {
        "id": "https://openalex.org/W2808845378",
        "publication_year": 2018,
        "cited_by_count": 38,
        "counts_by_year": _cby([(2019, 4), (2020, 10), (2023, 24)]),
    }
    rec = snapshot_record("W2808845378", work, 2019)
    assert rec["openalex_id"] == "W2808845378"
    assert rec["cited_by_count"] == 38
    assert rec["cited_since_cutoff"] == 34
    assert rec["cited_asof_cutoff"] == 4
    assert rec["clamped"] is False
    assert rec["cutoff_year"] == 2019
    # raw counts_by_year stored verbatim so other cutoffs derive later
    assert rec["counts_by_year"] == work["counts_by_year"]
    # merged work returned under a canonical id keeps the requested key
    rec2 = snapshot_record("W_old", {"id": "https://openalex.org/W_new",
                                     "cited_by_count": 1}, 2019)
    assert rec2["requested_id"] == "W_old" and rec2["openalex_id"] == "W_new"


def test_build_id_map_doi_hash_and_unmatched():
    raw = [
        {"openalex_id": "W1", "doi": "10.1/abc", "title": "Alpha", "year": 2015},
        {"openalex_id": "W2", "doi": None, "title": "No Doi Paper", "year": 2010},
        {"openalex_id": "W3", "doi": "10.1/zzz", "title": "Gamma", "year": 2018},
    ]
    papers = [
        ("doi:10.1/abc", "10.1/abc", "Alpha", 2015),                # doi join
        (_paper_id(None, "No Doi Paper", 2010), None,
         "No Doi Paper", 2010),                                     # h: join
        ("doi:10.9/missing", "10.9/missing", "Orphan", 2012),       # unmatched
    ]
    mapping, unmatched = build_id_map(papers, raw, _paper_id)
    assert mapping["doi:10.1/abc"] == "W1"
    assert mapping[_paper_id(None, "No Doi Paper", 2010)] == "W2"
    assert papers[1][0].startswith("h:")  # really exercised the hash route
    assert len(unmatched) == 1 and unmatched[0][0] == "doi:10.9/missing"


def test_shard_store_checkpoints_and_resumes(tmp_path):
    # Resume-by-default is sacred: records appended in one "run" must be
    # visible (and skippable) in the next, and shards must roll over.
    store = ShardStore(tmp_path / "openalex_citations", shard_size=3)
    store.append([{"requested_id": f"W{i}", "cited_asof_cutoff": i}
                  for i in range(4)])
    # 4 records at shard_size=3 -> two shards
    assert [p.name for p in store.shards()] == ["shard_0000.jsonl",
                                                "shard_0001.jsonl"]
    # fresh instance (a resumed run) sees all fetched ids
    store2 = ShardStore(tmp_path / "openalex_citations", shard_size=3)
    fetched = store2.load_fetched()
    assert set(fetched) == {"W0", "W1", "W2", "W3"}
    # appending after resume fills the partial shard before starting a new one
    store2.append([{"requested_id": "W4"}, {"requested_id": "W5"},
                   {"requested_id": "W6"}])
    fetched = ShardStore(tmp_path / "openalex_citations",
                         shard_size=3).load_fetched()
    assert set(fetched) == {f"W{i}" for i in range(7)}
    sizes = [sum(1 for _ in p.open()) for p in store2.shards()]
    assert sizes == [3, 3, 1]
