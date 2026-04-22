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
}

_FRONT_MATTER_PREFIX = re.compile(
    r"^(?:"
    r"table of contents\b|"
    r"front cover\b|"
    r"back cover\b|"
    r"blank page\b|"
    r"publication information\b|"
    r"information for authors\b|"
    # IEEE society / journal masthead variants: "<Full Journal Name> <trailing>"
    # where the trailing is publication-info / cover / society / volume / index.
    r"ieee[\s/a-z\-]+?"
    r"(?:publication\s+information|society|cover|front\s+cover|back\s+cover|"
    r"volume\s*\d|author\s+index|subject\s+index)\s*$|"
    r"ieee\s+[a-z\s]+?society\s*$|"  # "IEEE X Society" society-page entries
    r"\d{4}\s+index\s+ieee\b|"
    r"volume\s+\d+\s+index\b|"
    r"guest editorial\b|"
    r"editorial:?\s|"
    r"editor'?s? note\b|"
    r"corrigendum\s+to\b|"
    r"erratum\s+to\b|"
    r"retraction\s+notice\b|"
    r"author index\b|"
    r"subject index\b|"
    r"list of reviewers\b|"
    r"acknowledgment[s]?\s+to\s+reviewers\b"
    r")",
    re.IGNORECASE,
)


# Titles that are *exactly* a journal-masthead-sounding name (no descriptive content).
# Matches e.g. "IEEE Transactions on Intelligent Vehicles" used as a bare title.
_JOURNAL_NAME_ONLY = re.compile(
    r"^\s*ieee\s+(?:"
    r"transactions\s+on\s+[a-z\s]+|"
    r"intelligent\s+transportation\s+systems(?:\s+magazine|\s+society)?|"
    r"intelligent\s+vehicles?(?:\s+magazine|\s+symposium)?|"
    r"open\s+journal\s+of\s+[a-z\s]+|"
    r"vehicular\s+technology(?:\s+magazine)?"
    r")\s*$",
    re.IGNORECASE,
)


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
