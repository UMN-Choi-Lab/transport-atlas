import pandas as pd

from transport_atlas.process.authors import (
    canonical_last_first,
    coauthor_alias_map_from_papers,
    normalize_name,
    normalize_orcid,
    surname,
)
from transport_atlas.process.frontmatter import is_front_matter


def test_front_matter_catches_ieee_masthead():
    # real papers — must pass through
    assert not is_front_matter("Intra-Vehicle Networks: A Review")
    assert not is_front_matter("Deep Learning for Traffic Forecasting")
    assert is_front_matter("")  # empty title treated as front matter (sentinel)

def test_front_matter_rejects_ieee_publication_info():
    # actual titles we saw in the corpus attached to Peter Tuohy
    assert is_front_matter("IEEE Transactions on Intelligent Transportation Systems publication information")
    assert is_front_matter("IEEE Transactions on Intelligent Transportation Systems Publication Information")
    assert is_front_matter("IEEE INTELLIGENT TRANSPORTATION SYSTEMS SOCIETY")
    assert is_front_matter("IEEE Transactions on Intelligent Vehicles Publication Information")
    assert is_front_matter("Table of Contents")
    assert is_front_matter("Editorial: On the Future of Transportation")
    assert is_front_matter("Guest Editorial: Special Issue on CAVs")
    assert is_front_matter("Corrigendum to \"Something something\"")
    assert is_front_matter("List of Reviewers")

def test_front_matter_rejects_bare_journal_names():
    # Bare-journal-name titles are masthead entries
    assert is_front_matter("IEEE Transactions on Intelligent Vehicles")
    assert is_front_matter("IEEE Transactions on Intelligent Transportation Systems")
    assert is_front_matter("IEEE Intelligent Transportation Systems Magazine")
    # But a paper ABOUT a journal topic (has descriptive content) is NOT masthead
    assert not is_front_matter("IEEE Transactions on Vehicles: A Critical Review of the Field")  # has colon + descriptive text
    assert not is_front_matter("A Survey of IEEE Transactions on Intelligent Vehicles Publications")  # starts with "A Survey"


def test_diacritic_strip():
    assert normalize_name("Müller") == "muller"
    assert normalize_name("José García") == "jose garcia"


def test_canonical_last_first():
    assert canonical_last_first("Alice Müller") == "muller, alice"
    assert canonical_last_first("Müller, Alice") == "muller, alice"


def test_surname_extraction():
    assert surname("Alice Müller") == "muller"
    assert surname("Bob Kim") == "kim"


def test_orcid_normalize():
    assert normalize_orcid("https://orcid.org/0000-0001-2345-678X") == "0000-0001-2345-678X"
    assert normalize_orcid(None) is None
    assert normalize_orcid("not-an-orcid") is None


def _paper(pid: str, authors: list[dict]) -> dict:
    return {"paper_id": pid, "authors": authors}


def _a(aid: str, name: str, orcid: str | None = None) -> dict:
    return {"id": aid, "name": name, "orcid": orcid, "institutions": []}


def test_coauthor_alias_merges_same_name_with_shared_collaborators():
    # Two "Kim, Jiwon" entities sharing 2 collaborators in a 3-coauthor world -> merge
    papers = pd.DataFrame([
        _paper("p1", [_a("A1", "Jiwon Kim"), _a("X1", "Hwasoo Yeo"), _a("Y1", "Lijun Sun")]),
        _paper("p2", [_a("A1", "Jiwon Kim"), _a("X1", "Hwasoo Yeo"), _a("Z1", "Raphael Stern")]),
        _paper("p3", [_a("A2", "Jiwon Kim"), _a("X1", "Hwasoo Yeo"), _a("Y1", "Lijun Sun")]),
        _paper("p4", [_a("A2", "Jiwon Kim"), _a("W1", "Mark Hickman")]),
    ])
    mp = coauthor_alias_map_from_papers(papers)
    # One of A1/A2 should map to the other; canonical = more papers (tie -> either works)
    assert ("a1" in mp and mp["a1"] == "a2") or ("a2" in mp and mp["a2"] == "a1")


def test_coauthor_alias_rejects_singleton_single_shared_collaborator():
    # A1 has 10 coauthors, A2 is a 1-paper stub sharing exactly 1 coauthor -> do NOT merge
    big = [_a(f"C{i}", f"Co Author{i}") for i in range(10)]
    papers = pd.DataFrame([
        _paper("p1", [_a("A1", "Jiwon Kim")] + big),
        _paper("p2", [_a("A2", "Jiwon Kim"), _a("C0", "Co Author0"), _a("NEW", "Stranger Person")]),
    ])
    mp = coauthor_alias_map_from_papers(papers)
    assert "a1" not in mp and "a2" not in mp


def test_coauthor_alias_rejects_different_names():
    # Same 2 shared coauthors but different canonical names -> no merge
    papers = pd.DataFrame([
        _paper("p1", [_a("A1", "Jiwon Kim"), _a("X1", "Hwasoo Yeo"), _a("Y1", "Lijun Sun")]),
        _paper("p2", [_a("A2", "Jieun Lee"),  _a("X1", "Hwasoo Yeo"), _a("Y1", "Lijun Sun")]),
    ])
    mp = coauthor_alias_map_from_papers(papers)
    assert mp == {}


def test_coauthor_alias_transitive_via_union_find():
    # A~B via shared {X,Y}; B~C via shared {X,Z}; A and C share only {X}.
    # Transitive closure should still collapse all three to one canonical.
    papers = pd.DataFrame([
        _paper("p1", [_a("A", "Jiwon Kim"), _a("X", "C1"), _a("Y", "C2")]),
        _paper("p2", [_a("A", "Jiwon Kim"), _a("X", "C1"), _a("Y", "C2"), _a("P", "Extra1")]),
        _paper("p3", [_a("B", "Jiwon Kim"), _a("X", "C1"), _a("Y", "C2")]),
        _paper("p4", [_a("B", "Jiwon Kim"), _a("X", "C1"), _a("Z", "C3")]),
        _paper("p5", [_a("C", "Jiwon Kim"), _a("X", "C1"), _a("Z", "C3")]),
        _paper("p6", [_a("C", "Jiwon Kim"), _a("X", "C1"), _a("Z", "C3"), _a("Q", "Extra2")]),
    ])
    mp = coauthor_alias_map_from_papers(papers)
    canonicals = {mp.get(k, k) for k in ("a", "b", "c")}
    assert len(canonicals) == 1, f"expected 1 canonical, got {canonicals}"


def test_coauthor_alias_honors_existing_orcid_merges():
    # If auto_alias_map_from_papers already merged A1->A2, the coauthor pass should
    # treat both as A2 when building coauthor sets and not emit a redundant A1 entry.
    papers = pd.DataFrame([
        _paper("p1", [_a("A1", "Jiwon Kim"), _a("X", "C1"), _a("Y", "C2")]),
        _paper("p2", [_a("A2", "Jiwon Kim"), _a("X", "C1"), _a("Y", "C2")]),
    ])
    mp = coauthor_alias_map_from_papers(papers, existing_aliases={"a1": "a2"})
    assert "a1" not in mp  # already handled by ORCID pass
    assert "a2" not in mp  # nothing else to merge it with
