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
    r"editorial:?\s|"
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
    r".{0,80}\bbest (?:transactions )?paper award\b"
    r")",
    re.IGNORECASE,
)


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
    t = (title or "").strip().rstrip(".").lower()
    if not t:
        return True
    if t in FRONT_MATTER_EXACT:
        return True
    if _FRONT_MATTER_PREFIX.match(t):
        return True
    # "Title" that is literally only a journal name — masthead entries.
    if _JOURNAL_NAME_ONLY.match(t) and len(t) < 80:
        return True
    return False
