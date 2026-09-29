"""Front-matter filter + community-label hygiene tests (reviewer response,
2026-07). Reviewer 4 point 7: labels like "staff", "list", "calendar",
"editor column", "scanning issue" leaked from editorial/non-research metadata.

Two defenses are tested here, both offline:
  1. is_front_matter() catches the editorial title patterns found surviving in
     the 2026-07-20 corpus snapshot, WITHOUT catching real papers that share
     tokens with them (staff scheduling, corrections of measurements, ...).
  2. LABEL_STOPWORDS, merged into the TF-IDF label vectorizers, guarantees
     editorial tokens can never surface in community labels — including as
     n-gram members ("editor column", "scanning issue"), because sklearn
     removes stop words BEFORE assembling n-grams.
"""
from __future__ import annotations

from transport_atlas.process.frontmatter import LABEL_STOPWORDS, is_front_matter


# ——— 1a. Editorial titles that must be caught ————————————————————————————————
# Every string below is a real (possibly truncated) title observed in
# data/interim/papers.parquet on 2026-07-20 — i.e. it slipped the old filter.

EDITORIAL_TITLES = [
    # Reviewer-cited label sources
    "Editor's column",                                     # its-mag 2009
    "Editor’s Column",                                # curly apostrophe variant
    "New Features [Editor's Column]",
    "Calendar",                                            # 48x its-mag/vtm/jtg/tpol
    "Calendar of Events",
    "Calendar (2009)",
    "Calendar [Calendar]",
    "Conferences of Interest [Calendar of Events]",
    "[Staff Listing]",                                     # vtm 2012
    "Vehicular Technology Magazine Staff List",            # vtm 2023/2026
    # President's messages (straight + curly apostrophes, past/plural variants)
    "President's Message",                                 # t-its 2004
    "Thank You for Your Support [President’s Message]",
    "Thanks For a Great Experience! [Past President's Message]",
    "ITSS and Open Access Publications [Presidents' Message]",
    # Memorial / obituary
    "In Memoriam: Frank A. Haight 1919–2006",
    "In Memoriam Talib Rothengatter",
    "Obituary - Professor Michael E. Beesley, CBE",
    # Issue boilerplate (Elsevier legacy volumes)
    "Forthcoming papers",                                  # 184x tr-a/b/c, aap
    "List of forthcoming papers",                          # tr-b 1980s
    "List of contents and author index volume 24, 1992",   # aap
    "Book review",                                         # 311x
    "Book Review: Urban Transport in the Developing World: A Handbook",
    "Thanks to reviewers",                                 # aap/tr-a 1990s
    "Reviewers List",                                      # aap 2010
    "Publisher's note",
    "Publisher’s Note",
    "Notice",                                              # tr-a 1992
    "News, queries &amp; answers",                         # aap 1984
    # Special-issue announcements / intros (source of "issue" label tokens)
    "Special issue on human factors in intelligent vehicles",
    "Special Issue on the DARPA Urban Challenge Autonomous Vehicle Competition",
    "Introduction to the Special Issue on Real-Time Traffic State Estimation",
    "Introduction",                                        # 70x bare CACIE intros
    # Errata family
    "Erratum—“Maximum Inventories in Baggage Claim”",
    "Erratum regarding missing Declaration of Competing Interest statements",
    "Corrigendum:Bike share: A synthesis of the literature, Transport Reviews",
    "Correction to: Impact assessment of rural PPP MaaS pilots",
    "Correction",
    "Editorial—A Major Milestone: Transportation Science Turns Fifty",
    # Conference announcements
    "ITSC 2011",
    "ITSC'09",
    "ITSC 2011 Call for Papers",
    # IEEE magazine department tags (editorial denylist)
    "Welcome to the March 2020 issue [From the Editor]",
    "Advanced Air Mobility [From the Guest Editors]",
    "Meeting of the Executive Committee [Society News]",
    "VTC2009-Spring in Barcelona [VTS news]",
    "ITS Society Conferences [Conference Reports]",
    "Intelligent transportation systems in China [Guest Editorial]",
    "2012 IEEE Intelligent Vehicles Symposium 3-7 June [Conference Report]",
    "Get Acquainted with the Most Recent Research [PhD &amp; MPhil Theses' Abstracts]",
    "[PH.D. &amp; M.PHIL. Theses' Abstracts]",
]


def test_editorial_titles_caught():
    missed = [t for t in EDITORIAL_TITLES if not is_front_matter(t)]
    assert not missed, f"editorial titles not caught: {missed}"


# ——— 1b. Real papers that share tokens with editorial patterns ———————————————
# All of these are real corpus titles (or realistic near-copies). They MUST
# survive the filter — precision matters as much as recall here.

REAL_TITLES = [
    # "staff" is a research topic (scheduling, frontline workers, commuters)
    "Staff scheduling for bus drivers with days-off preferences",
    "Mobility behaviors of Italian university students and staff",
    "An integrated framework for electric vehicle rebalancing and staff "
    "relocation in one-way carsharing systems",
    "Tricks and tactics used against troublesome travelers—Frontline "
    "staff's experiences from Swedish buses and trains",
    # "Meeting …" as a research verb, not an announcement
    "Meeting points in ridesharing: A privacy-preserving approach",
    "Meeting an 80% reduction in greenhouse gas emissions from "
    "transportation by 2050: A case study in California",
    # "Correction of X" = real methods papers (vs "Correction to: X" errata)
    "Correction of Field Skid Measurements for Seasonal Variations in Texas",
    "Correction of Plane Strain Analyses for Corrugated Metal Culverts",
    # Research terms sharing tokens with editorial ones
    "Mobile laser scanning for road inventory and asset management",
    "Calendar effects in daily traffic crash counts",
    "Introduction of congestion pricing in Stockholm",
    "On-board charger design for electric vehicles",
    "Low-volume road maintenance strategies",
    "Building information modeling for infrastructure projects",
    "Seismic behavior of bridge columns under cyclic loading",
    "News media coverage of traffic safety campaigns and driver behavior",
    # Content-department magazine tags must survive (only editorial tags die)
    "Emergency message dissemination in vehicular networks [Mobile Radio]",
    "Battery management systems for EVs [Automotive Electronics]",
    "Cooperative perception at intersections [Transportation Systems]",
    # Ordinary papers
    "Deep Learning for Traffic Forecasting",
    "A review of transit assignment models",
]


def test_real_papers_survive():
    killed = [t for t in REAL_TITLES if is_front_matter(t)]
    assert not killed, f"real papers wrongly filtered: {killed}"


def test_empty_and_none_are_front_matter():
    assert is_front_matter(None)
    assert is_front_matter("")
    assert is_front_matter("   .  ")


# ——— 2. LABEL_STOPWORDS content + TF-IDF n-gram behavior —————————————————————


def test_label_stopwords_cover_reviewer_tokens():
    # Every token the reviewer cited must be stopped.
    for tok in ("staff", "list", "calendar", "editor", "column", "issue"):
        assert tok in LABEL_STOPWORDS, tok


def test_label_stopwords_spare_research_tokens():
    # Load-bearing research vocabulary must NOT be stopped: these appear in
    # legitimate community labels (laser scanning, traveler information,
    # traffic volume, pavement condition index, on-board, V2V messages,
    # bridge columns).
    for tok in ("scanning", "information", "volume", "index", "board",
                "message", "columns", "meeting", "laser", "bridge"):
        assert tok not in LABEL_STOPWORDS, tok


def _production_vectorizer(min_df: int):
    """TfidfVectorizer configured exactly like the three label vectorizers
    (coauthor_graph._tfidf_labels and the two blocks in
    scripts/06_author_similarity.py): english + local stops + LABEL_STOPWORDS,
    same token_pattern, same ngram_range."""
    from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                                 TfidfVectorizer)
    return TfidfVectorizer(
        max_df=0.5,
        min_df=min_df,
        stop_words=sorted(set(ENGLISH_STOP_WORDS) | LABEL_STOPWORDS),
        token_pattern=r"[A-Za-z][A-Za-z\-]{2,}",
        ngram_range=(1, 2),
    )


def test_tfidf_stopwords_block_editorial_ngrams():
    """Stop words are removed BEFORE n-gram assembly, so editorial bigrams
    ("editor column", "scanning issue", "president message") cannot form even
    when one member ("scanning", "president") is not itself a stop word."""
    docs = [
        "editor column president message staff list calendar scanning the issue",
        "editor column president message staff list calendar of events scanning the issue",
        "editor column president message staff list calendar",
        "laser scanning point cloud road inventory",
        "laser scanning point cloud mobile mapping",
        "point cloud asset extraction road inventory",
        "congestion pricing equilibrium tolls",
        "congestion pricing network tolls",
    ]
    vec = _production_vectorizer(min_df=2)
    vec.fit(docs)
    vocab = set(vec.get_feature_names_out())

    # No feature may contain a stopworded token — unigram or n-gram member.
    contaminated = {
        f for f in vocab
        if set(f.split()) & LABEL_STOPWORDS
    }
    assert not contaminated, f"stopworded tokens leaked into vocab: {contaminated}"

    # The reviewer-cited bigrams specifically can never exist.
    for bad in ("editor column", "scanning issue", "president message",
                "staff list", "calendar events"):
        assert bad not in vocab, bad

    # Legitimate research terms survive as labels.
    assert "laser scanning" in vocab
    assert "congestion pricing" in vocab
    assert "scanning" in vocab  # unigram survives; only "issue" is stopped


def test_coauthor_tfidf_labels_are_clean():
    """End-to-end through the real production labeler
    (coauthor_graph._tfidf_labels): a community whose members authored
    editorial leftovers must still get labels free of editorial tokens."""
    from transport_atlas.process.coauthor_graph import _tfidf_labels

    # The production vectorizer uses min_df=3 / max_df=0.5 over per-community
    # documents, so we need >= 9 communities: each real topic spans exactly 3
    # of them (df=3 satisfies min_df=3 and stays under max_df=0.5*9=4.5).
    edi = "editor column calendar staff list scanning the issue president message"
    topics = {
        # communities 0-2: contaminated with editorial leftovers + a real topic
        0: [edi, "arterial signal coordination", edi],
        1: [edi, "arterial signal timing optimization"],
        2: ["arterial signal control", edi],
        # communities 3-5: laser scanning (legit use of "scanning")
        3: ["mobile laser scanning inventory", "laser scanning point cloud"],
        4: ["laser scanning asset extraction", "laser scanning point cloud"],
        5: ["laser scanning point cloud inventory"],
        # communities 6-8: congestion pricing
        6: ["congestion pricing equilibrium tolls", "congestion pricing cordon"],
        7: ["congestion pricing equilibrium tolls"],
        8: ["congestion pricing cordon equilibrium"],
    }
    comm_members = {cid: [f"m{cid}"] for cid in topics}
    paper_records = [([f"m{cid}"], 2020, t, 0)
                     for cid, ts in topics.items() for t in ts]

    labels = _tfidf_labels(comm_members, paper_records, label_of={},
                           n_comm=9, misc_cid=None)

    flat = {w for lab in labels for w in lab}
    contaminated = {w for w in flat if set(w.split()) & LABEL_STOPWORDS}
    assert not contaminated, f"editorial tokens in labels: {contaminated}"
    # Real topics still label their communities.
    for cid in (0, 1, 2):
        assert any("signal" in w for w in labels[cid]), (cid, labels[cid])
    for cid in (3, 4, 5):
        assert any("scanning" in w for w in labels[cid]), (cid, labels[cid])
    for cid in (6, 7, 8):
        assert any("pricing" in w for w in labels[cid]), (cid, labels[cid])
