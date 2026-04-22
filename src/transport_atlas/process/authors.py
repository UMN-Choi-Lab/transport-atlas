"""Author normalization + ORCID clustering."""
from __future__ import annotations

import re

from unidecode import unidecode


def normalize_name(name: str | None) -> str:
    """Strip diacritics, lowercase, collapse whitespace."""
    if not name:
        return ""
    s = unidecode(name).lower()
    s = re.sub(r"[^a-z\s,.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_last_first(name: str | None) -> str:
    """Return 'last, first' from various input forms. Returns '' if unparseable."""
    if not name:
        return ""
    s = normalize_name(name)
    if "," in s:
        return s
    parts = s.split(" ")
    if len(parts) < 2:
        return s
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def surname(name: str | None) -> str:
    c = canonical_last_first(name)
    if not c:
        return ""
    return c.split(",")[0].strip()


def normalize_orcid(orcid: str | None) -> str | None:
    if not orcid:
        return None
    m = re.search(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dxX])", orcid)
    return m.group(1).upper() if m else None


_MIN_NAME_KEY_LEN = 5  # reject initials / empty — prevents unrelated-author collapse


def author_key(a: dict) -> str:
    """Canonical lookup key used by both dedupe and coauthor graph.

    Priority: OpenAlex author id > ORCID > normalized-name (≥5 chars).
    Returns "" if no stable identifier is available (caller must filter).
    """
    aid = a.get("id")
    if aid and str(aid).startswith(("A", "a")):
        return str(aid).lower()
    orcid = normalize_orcid(a.get("orcid"))
    if orcid:
        return orcid.lower()
    nm = normalize_name(a.get("name") or "")
    # Require surname+first-letter ("kim, b" -> 6 chars) at minimum; blocks "h", "", etc.
    canonical = canonical_last_first(a.get("name") or "")
    alpha = re.sub(r"[^a-z]", "", canonical)
    if len(alpha) >= _MIN_NAME_KEY_LEN:
        return canonical
    return ""


def auto_alias_map_from_papers(papers) -> dict[str, str]:
    """Auto-detect author-key splits caused by missing OpenAlex IDs on some papers.

    When the same ORCID co-occurs with multiple author_keys across papers, merge them.
    The key with the most papers becomes canonical. Returns {other_key: canonical_key}.

    This fixes the common case where a handful of records lack OpenAlex IDs (yielding
    an ORCID-keyed stub author) while the rest of the same person's papers share a
    single OpenAlex author id. Previously these were split — e.g. ~1k authors, 2k
    split keys, 18k papers affected in a 100k-paper corpus.
    """
    from collections import defaultdict
    orcid_keys: dict[str, set[str]] = defaultdict(set)
    key_papers: dict[str, int] = defaultdict(int)

    for _, r in papers.iterrows():
        al = r.get("authors")
        try:
            if al is None or len(al) == 0:
                continue
        except TypeError:
            continue
        for a in al:
            if not isinstance(a, dict):
                continue
            orc = normalize_orcid(a.get("orcid"))
            k = author_key(a)
            if not k:
                continue
            key_papers[k] += 1
            if orc:
                orcid_keys[orc.lower()].add(k)

    mp: dict[str, str] = {}
    for orc, keys in orcid_keys.items():
        if len(keys) < 2:
            continue
        # Canonical = key with most papers; tie-break by preferring OpenAlex IDs
        def _score(k):
            oa_pref = 1 if k.startswith("a") else 0
            return (key_papers[k], oa_pref)
        canonical = max(keys, key=_score)
        for k in keys:
            if k != canonical:
                mp[k] = canonical
    return mp


def coauthor_alias_map_from_papers(
    papers,
    *,
    existing_aliases: dict[str, str] | None = None,
    overlap_threshold: float = 0.3,
    min_shared: int = 2,
) -> dict[str, str]:
    """Merge same-name author keys whose coauthor sets overlap strongly.

    Complements `auto_alias_map_from_papers` (which handles shared-ORCID splits).
    This catches the harder case: same canonical name, different/missing ORCIDs,
    but strongly overlapping collaborator sets — the typical OpenAlex artifact when
    a single researcher changes affiliation (e.g. Northwestern → UQ) or registers
    ORCID mid-career and old records don't get back-stamped.

    Similarity = overlap coefficient |A∩B| / min(|A|,|B|) over coauthor-key sets,
    with a `min_shared` floor that prevents inflation on tiny sets (e.g. a 1-paper
    stub whose sole coauthor happens to also appear on a veteran's record).

    Merges are transitive via union-find. Canonical = most papers, tie-break prefers
    OpenAlex "a…" ids. Apply `existing_aliases` first so pre-merged keys aren't
    re-examined separately.
    """
    from collections import defaultdict

    existing_aliases = existing_aliases or {}

    def _resolve(k: str) -> str:
        return existing_aliases.get(k, k)

    key_name: dict[str, str] = {}
    key_papers: dict[str, int] = defaultdict(int)
    key_coauthors: dict[str, set[str]] = defaultdict(set)

    for _, r in papers.iterrows():
        al = r.get("authors")
        try:
            if al is None or len(al) == 0:
                continue
        except TypeError:
            continue
        resolved = []
        for a in al:
            if not isinstance(a, dict):
                continue
            k = author_key(a)
            if not k:
                continue
            k = _resolve(k)
            key_papers[k] += 1
            nm = canonical_last_first(a.get("name") or "")
            if nm and k not in key_name:
                key_name[k] = nm
            resolved.append(k)
        unique = set(resolved)
        for k in unique:
            key_coauthors[k] |= (unique - {k})

    by_name: dict[str, list[str]] = defaultdict(list)
    for k, nm in key_name.items():
        if nm:
            by_name[nm].append(k)

    parent: dict[str, str] = {}

    def find(k: str) -> str:
        while parent.get(k, k) != k:
            parent[k] = parent.get(parent[k], parent[k])
            k = parent[k]
        return k

    def score(k: str) -> tuple[int, int]:
        return (key_papers[k], 1 if k.startswith("a") else 0)

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if score(ra) >= score(rb):
            parent[rb] = ra
        else:
            parent[ra] = rb

    for nm, keys in by_name.items():
        if len(keys) < 2:
            continue
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                ca = key_coauthors[a] - {a, b}
                cb = key_coauthors[b] - {a, b}
                if not ca or not cb:
                    continue
                shared = ca & cb
                if len(shared) < min_shared:
                    continue
                if len(shared) / min(len(ca), len(cb)) >= overlap_threshold:
                    union(a, b)

    mp: dict[str, str] = {}
    for k in parent:
        c = find(k)
        if c != k:
            mp[k] = c
    return mp
