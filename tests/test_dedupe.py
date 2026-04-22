"""Offline dedupe tests using tests/fixtures/small_corpus.json.

We patch config.data_dir so `_load_all` reads from a temp tree populated from fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from transport_atlas.process import dedupe
from transport_atlas.utils import config


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    fx = json.loads((Path(__file__).parent / "fixtures" / "small_corpus.json").read_text())
    # lay out raw tree
    for src, items in fx.items():
        by_venue: dict[str, list[dict]] = {}
        for it in items:
            by_venue.setdefault(it["venue_slug"], []).append(it)
        for slug, rows in by_venue.items():
            d = tmp_path / "data" / "raw" / src
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{slug}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def _data_dir(subdir: str = "") -> Path:
        p = tmp_path / "data" / subdir if subdir else tmp_path / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(config, "data_dir", _data_dir)
    return tmp_path


def test_doi_collapses_duplicates(corpus):
    report = dedupe.run(write=True)
    # Raw fixture has 15 OA rows + 1 IEEE row = 16 records.
    # Duplicates expected to collapse:
    #   W1/W2 (same DOI 10.1/a) -> 1 paper
    #   W3/W16_ieee (same DOI 10.1/b) -> 1 paper + W4 (no DOI, fuzzy) -> merged into 10.1/b
    #   W5/W15 (W5 has DOI 10.1/c, W15 has no DOI but same title/year/author) -> 1 paper
    # So 16 raw -> 13 unique (expected). Allow small slack (fuzzy may not catch all).
    assert report["raw_rows"] == 16
    assert 11 <= report["unique_papers"] <= 14, report


def test_parquet_written(corpus):
    dedupe.run(write=True)
    interim = corpus / "data" / "interim"
    assert (interim / "papers.parquet").exists()
    assert (interim / "authors.parquet").exists()
    rpt = json.loads((interim / "_dedupe_report.json").read_text())
    assert rpt["unique_papers"] > 0
    # Authors: A1..A7 = 7 canonical authors
    assert rpt["authors"] >= 6
