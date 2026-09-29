"""Journal front-matter filter.

Adapted from robopaper-atlas/_clean.py. OpenAlex indexes Tables of Contents,
Editorials, Publication Information, etc. as works with authors attached.
These inflate hubs, pollute top-cited lists, and add noise to community
keyword extraction. Filter them out at the dedup step.
"""
from __future__ import annotations

import re

FRONT_MATTER_EXACT = {
    "table of contents", "front cover", "back cover", "blank page",
    "editorial", "frontispiece", "index", "contents", "toc",
    "publication information", "information for authors",
    "masthead", "colophon", "title page", "cover",
    "call for papers", "acknowledgments to reviewers",
    "corrigendum", "erratum", "retraction notice",
    "in memoriam", "from the editor", "from the editor in chief",
    "editor's note", "editorial note", "letter from the editor",
    "guest editorial", "foreword", "preface",
    "author index", "subject index", "reviewer index",
    "volume contents",
    # New (2026-04-29): patterns surfaced after adding OR/OM venues that also
    # exist in IEEE/Elsevier corpora — IEEE society pages, INFORMS announcements,
    # Wiley/CACIE issue boilerplate, and IEEE VTM editorial-staff listings.
    "staff list", "staff listing", "editorial board", "issue information",
    "in this issue", "announcements", "scanning the issue", "ieee app",
    "techrxiv", "distinguished lecturer program", "ieee policies",
    # New (2026-07, reviewer response): bare-title boilerplate found surviving
    # in data/interim/papers.parquet (counts from the 2026-07-20 snapshot).
    "calendar", "calendar of events",     # 48× its-mag/vtm/jtg/tpol
    "introduction",                       # 70× bare CACIE issue intros
    "forthcoming papers",                 # 184× Elsevier TR-A/B/C, AAP
    "notice", "publisher's note", "publishers note",
    "correction", "corrections",
    "obituary", "bookshelf", "diary",
    "president's message", "editor's column",
    "reviewers list", "thanks to reviewers",
    "news, queries & answers", "news, queries &amp; answers",
}

_FRONT_MATTER_PREFIX = re.compile(
    r"^(?:"
    r"table of contents\b|"
    r"\[?\s*table of contents\s*\]?|"  # bracketed [Table of Contents]
    r"front cover\b|"
    r"back cover\b|"
    r"blank page\b|"
    r"publication information\b|"
    r"information for authors\b|"
    r"instructions?\s+(?:to|for)\s+authors\b|"
    # IEEE society / journal masthead variants: "<Full Journal Name> <trailing>"
    # where the trailing is publication-info / cover / society / volume / index.
    r"ieee[\s/a-z\-]+?"
    r"(?:publication\s+information|society|cover|front\s+cover|back\s+cover|"
    r"volume\s*\d|author\s+index|subject\s+index)\s*$|"
    r"ieee\s+[a-z\s]+?society\s*$|"  # "IEEE X Society" society-page entries
    r"\d{4}\s+index\s*ieee\b|"   # "2018 Index IEEE" or "2018 IndexIEEE…" (no space)
    r"\d{4}\s+(?:vt|its)\s+year\s+end\s+index\b|"
    r"\d{4}\s+ieee\s+[a-z\s]+\s+(?:index|elections)\b|"
    r"volume\s+\d+\s+index\b|"
    r"guest editorial\b|"
    r"editorial(?:[:—–]|\s)|"  # "Editorial: X", "Editorial—X" (em/en dash)
    r"editorial\s+board\b|"
    r"editor'?s? note\b|"
    r"corrigendum\s+to\b|"
    r"erratum\s+to\b|"
    r"retraction\s+notice\b|"
    r"author index\b|"
    r"subject index\b|"
    r"list of reviewers\b|"
    r"acknowledgment[s]?\s+to\s+reviewers\b|"
    # New (2026-04-29): IEEE/INFORMS/Elsevier non-research entries.
    r"staff list(?:ing)?\b|"
    r"in this issue\b|"
    r"issue information\b|"
    r"scanning the issue\b|"
    r"announcements?\b|"
    r"call for (?:papers|reviewers|nominations)\b|"
    r"message from\b|"
    r"highlights? of\b|"
    r"society conferences?\b|"
    r"society information\b|"
    r"conference reports?\b|"
    r"distinguished lecturer program\b|"
    r"techrxiv\b|"
    r"ieee app\b|"
    # New (2026-07, reviewer response): editorial/boilerplate prefixes verified
    # against the 2026-07-20 corpus snapshot (counts in comments). Patterns are
    # kept narrow so real papers survive (e.g. "Correction of Field Skid
    # Measurements…" and "Meeting points in ridesharing…" must NOT match).
    r"editor'?s?\s+column\b|"                      # 2× its-mag
    r"president'?s?\s+message\b|"                  # 1× + tag-suffix variants
    r"in\s+memoriam\b|"                            # 26×
    r"obituar(?:y|ies)\b|"                         # 4×
    r"book\s+reviews?\b|"                          # 311×
    r"forthcoming\s+papers\b|"                     # 184×
    r"list\s+of\s+(?:forthcoming\s+papers|contents)\b|"  # 64×
    r"reviewers'?\s+list\b|"                       # 2×
    r"thanks?\s+(?:to|you\s+to)\s+(?:the\s+)?(?:reviewers|referees)\b|"  # 16×
    r"itsc\s*'?\s*\d|"                             # "ITSC 2011", "ITSC'09" announcements
    r"special\s+issue\b|"                          # 100× SI announcements/guest intros
    r"introduction\s+to\s+the\s+(?:special|featured)\s+(?:issue|section)\b|"
    r"publisher'?s\s+note\b|"
    r"news,\s*queries\b|"                          # AAP "News, queries & answers"
    r"correction(?:s)?\s+to\b|"                    # "Correction to: X" errata
    r"correction:\s|"
    r"errat(?:um|a)\b|"                            # "Erratum—X", "Erratum: X", "Erratum regarding…"
    r"corrigend(?:um|a)\b|"                        # any corrigendum variant
    r"calendar\s*(?:$|\(|\[|of\s+events)|"         # "Calendar (2009)", "Calendar of Events"
    r".{0,80}\bbest (?:transactions )?paper award\b"
    r")",
    re.IGNORECASE,
)

# IEEE-magazine department tags: titles end with "[<department>]". Content
# departments ([Mobile Radio], [Automotive Electronics], [ITS Research Lab],
# [Standards], …) carry real technical articles and MUST survive — only the
# editorial/metadata departments below are front matter. Denylist verified
# against the 1,066 bracket-tagged titles in the 2026-07-20 snapshot.
_EDITORIAL_TAG = re.compile(
    r"\[(?:"
    r"from the (?:guest )?editors?|"
    r"(?:past )?presidents?'?s? message|"
    r"editor'?s column|"
    r"society news|social news|vts news|"
    r"calendar(?: of events)?|"
    r"front cover|back cover|"
    r"(?:guest )?editorial|"
    r"book reviews?|"
    r"in memoriam|"
    r"errat(?:um|a)|corrigend(?:um|a)|"
    r"call for papers|"
    r"conference reports?|"
    r"its people|its fun|its events|its conference activities|"
    r"member activities|technical activities|technical committees|"
    r"awards?|advertisement|"
    r"staff list(?:ing)?|"
    r"scanning the issue|table of contents|in this issue|"
    r"society information|publication information|"
    r"\d{4} index|"
    r"ph\.?d\.?.{0,20}theses'? abstracts?"
    r")\]\s*$",
    re.IGNORECASE,
)

# Suffix-only boilerplate without brackets, e.g.
# "Vehicular Technology Magazine Staff List" (journal-name prefix defeats the
# prefix regex; _JOURNAL_NAME_ONLY requires the name to be the whole title).
_FRONT_MATTER_SUFFIX = re.compile(r"\bstaff\s+list(?:ing)?\s*$", re.IGNORECASE)


# Titles that are *exactly* a journal-masthead-sounding name (no descriptive content).
# Matches e.g. "IEEE Transactions on Intelligent Vehicles" used as a bare title.
_JOURNAL_NAME_ONLY = re.compile(
    r"^\s*(?:"
    r"ieee\s+(?:"
    r"transactions\s+on\s+[a-z\s]+|"
    r"intelligent\s+transportation\s+systems(?:\s+magazine|\s+society)?|"
    r"intelligent\s+vehicles?(?:\s+magazine|\s+symposium)?|"
    r"open\s+journal\s+of\s+[a-z\s]+|"
    r"vehicular\s+technology(?:\s+magazine)?"
    r")|"
    # Non-IEEE: bare journal name as title (e.g. "Journal of Public Transportation").
    r"journal\s+of\s+[a-z\s]+|"
    r"european\s+journal\s+of\s+[a-z\s]+|"
    r"management\s+science|"
    r"operations?\s+research|"
    r"transportation\s+(?:science|research(?:\s+part\s+[a-f])?)|"
    r"informs\s+journal\s+on\s+computing"
    r")\s*$",
    re.IGNORECASE,
)


# Crossref-verified denylist of records where OpenAlex inflated the author list
# with the entire issue's author roster (real title, abstract empty, OA author
# count near or at the 100-cap, Crossref returns 0 authors). Verified
# 2026-04-29 against api.crossref.org/works/<doi>.
INFLATED_AUTHOR_DOIS = frozenset({
    "10.1016/s0965-8564(97)88270-5",  # tr-a 1997, OA=100
    "10.1016/s0965-8564(97)88297-3",  # tr-a 1997, OA=100
    "10.1016/s0965-8564(97)88358-9",  # tr-a 1997, OA=89
    "10.1016/s0965-8564(97)88292-4",  # tr-a 1997, OA=48
    "10.1016/0965-8564(95)90257-0",   # tr-a 1995, OA=57
    "10.1016/0965-8564(95)90285-6",   # tr-a 1995, OA=56
    "10.1016/0377-2217(91)90058-4",   # ejor 1991, OA=25
})


def is_front_matter(title: str | None) -> bool:
    # Normalize curly apostrophes: IEEE metadata mixes "President's" / "President’s".
    t = (title or "").replace("’", "'").strip().rstrip(".").lower()
    if not t:
        return True
    if t in FRONT_MATTER_EXACT:
        return True
    if _FRONT_MATTER_PREFIX.match(t):
        return True
    if _EDITORIAL_TAG.search(t):
        return True
    if _FRONT_MATTER_SUFFIX.search(t):
        return True
    # "Title" that is literally only a journal name — masthead entries.
    if _JOURNAL_NAME_ONLY.match(t) and len(t) < 80:
        return True
    return False


# ——— Community-label hygiene (2026-07, reviewer response) ————————————————————
# Editorial/metadata tokens that must never surface in TF-IDF community labels
# (coauthor_graph._tfidf_labels + the semantic/combined label blocks in
# scripts/06_author_similarity.py). This is defense-in-depth on top of
# is_front_matter(): even if a boilerplate title slips the corpus filter, its
# tokens cannot become label words. sklearn removes stop words BEFORE building
# n-grams, so stopping either member of a bigram ("editor column",
# "scanning issue", "president message") prevents the bigram from forming.
#
# Deliberately NOT included, because they are load-bearing in real research
# terms: "scanning" (laser scanning), "information" (traveler information,
# building information), "volume" (traffic volume), "index" (pavement condition
# index), "board" (on-board), "message" (V2V message dissemination), "meeting"
# (meeting points/appointments), "columns" (bridge columns — only the singular
# editorial "column" is stopped).
LABEL_STOPWORDS = frozenset({
    "editorial", "editorials", "editor", "editors", "column",
    "president", "presidents", "calendar", "staff", "list", "lists",
    "issue", "issues", "memoriam", "obituary", "obituaries",
    "erratum", "errata", "corrigendum", "corrigenda", "correction",
    "corrections", "retraction", "retracted", "foreword", "preface",
    "masthead", "frontispiece", "colophon", "toc", "contents", "cover",
    "announcement", "announcements", "welcome", "reviewers", "referees",
    "award", "awards", "society", "ieee", "bookshelf", "news", "notice",
    "notices", "publisher", "publication", "forthcoming", "papers",
    "paper", "note", "guest", "thanks",
})
