from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from .indexing import LocalIndex
from .models import Chunk


# ── 1) Query Classification ──────────────────────────────────────────────────
QueryIntent = Literal["section_list", "normal_qa"]

# Patterns that signal "give me everything under a heading / list all X"
_SECTION_LIST_PATTERNS: list[re.Pattern[str]] = [
    # Turkish
    re.compile(r"nelerdir", re.IGNORECASE),
    re.compile(r"nedir.*(maddeleri|listesi|gereksinimleri|başlıkları)", re.IGNORECASE),
    re.compile(r"(listele|sırala|say\b|maddeleri)", re.IGNORECASE),
    re.compile(r"(altında(ki)?|içindeki)\s+(tüm|her|bütün)", re.IGNORECASE),
    re.compile(r"(tüm|bütün|hepsi|eksiksiz)\s+.*(madde|gereksinim|başlık|teslimat|adım)", re.IGNORECASE),
    re.compile(r"kaç\s+(madde|gereksinim|başlık|teslimat|adım)", re.IGNORECASE),
    # English
    re.compile(r"what\s+are\s+(the|all)", re.IGNORECASE),
    re.compile(r"list\s+(all|the|every)", re.IGNORECASE),
    re.compile(r"(enumerate|summarize\s+all)", re.IGNORECASE),
    re.compile(r"how\s+many\s+(items?|requirements?|sections?)", re.IGNORECASE),
]


def classify_query(query: str) -> QueryIntent:
    """
    Rule-based intent classifier (document-agnostic).

    Returns "section_list" if the query looks like it wants a full list /
    section extraction; "normal_qa" otherwise.
    """
    for pat in _SECTION_LIST_PATTERNS:
        if pat.search(query):
            return "section_list"
    return "normal_qa"


# ── 2) Evidence gathering ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    text: str
    section_id: str
    heading_path: str
    page_start: int
    page_end: int
    kind: str  # parent / child
    score: float


@dataclass
class RetrievalResult:
    intent: QueryIntent
    evidences: List[Evidence]
    section_complete: bool  # True if we did complete-section fetch
    coverage: Optional[CoverageInfo] = None


@dataclass(frozen=True)
class CoverageInfo:
    expected_items: int
    heading_path: str
    section_id: str


def _meta_to_evidence(meta: dict, doc: str, chunk_id: str, score: float) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        text=doc,
        section_id=meta.get("section_id", ""),
        heading_path=meta.get("heading_path", ""),
        page_start=meta.get("page_start", 0),
        page_end=meta.get("page_end", 0),
        kind=meta.get("kind", ""),
        score=score,
    )


# ── Heading-aware section matching ───────────────────────────────────────────

def _tokenize_simple(text: str) -> Set[str]:
    """Simple word tokenizer for TR/EN overlap matching."""
    text = text.lower()
    text = re.sub(r"[^0-9a-zçğıöşüâîû\-]+", " ", text)
    return {t for t in text.split() if len(t) > 1}


def _heading_query_overlap(heading_path: str, query: str) -> float:
    """
    Compute how many query tokens appear in the heading path.
    Returns a score 0..1  (fraction of query tokens found in heading).
    """
    q_tokens = _tokenize_simple(query)
    h_tokens = _tokenize_simple(heading_path)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & h_tokens
    return len(overlap) / len(q_tokens)


def _pick_best_section(
    got_ids: List[str],
    got_metas: List[dict],
    hybrid_scores: Dict[str, float],
    query: str,
) -> Optional[str]:
    """
    From the hybrid search results, pick the best section_id.
    Strategy: combine hybrid score with heading-overlap bonus.
    This prevents "teslimatlar nelerdir" from matching "Teknik Yaklaşım" just
    because it ranks slightly higher in the embedding space.
    """
    # Collect unique section candidates
    candidates: Dict[str, Tuple[float, float]] = {}  # section_id -> (best_hybrid_score, heading_overlap)
    for cid, meta in zip(got_ids, got_metas):
        sid = meta.get("section_id", "")
        if not sid or sid == "root":
            continue
        hp = meta.get("heading_path", "")
        hs = hybrid_scores.get(cid, 0.0)
        ho = _heading_query_overlap(hp, query)
        if sid not in candidates or hs > candidates[sid][0]:
            candidates[sid] = (hs, ho)

    if not candidates:
        return None

    # Score = hybrid_score + 0.5 * heading_overlap
    # The 0.5 bonus is enough to nudge a heading-matching section to the top
    # when hybrid scores are close, but won't override a vastly better match.
    best_sid = max(candidates, key=lambda sid: candidates[sid][0] + 0.5 * candidates[sid][1])
    return best_sid


# ── Section + subtree fetch ──────────────────────────────────────────────────

def _fetch_section_and_subtree(index: LocalIndex, section_id: str) -> List[Evidence]:
    """
    Fetch the section's own chunks (parent + children) AND all chunks whose
    parent_id equals this section_id (the subtree).  This ensures that asking
    for "4. Teslimatlar" also brings in 4.1, 4.2, etc.
    """
    col = index.store._get_collection()

    # 1) Chunks belonging directly to this section
    own = col.get(
        where={"section_id": section_id},
        include=["documents", "metadatas"],
    )

    # 2) Chunks whose parent_id is this section (sub-sections)
    children_of = col.get(
        where={"parent_id": section_id},
        include=["documents", "metadatas"],
    )

    # 3) Collect recursively: sub-sections may have their own children
    # We do one more level (grandchildren) to cover e.g. 4.1.1
    sub_section_ids: Set[str] = set()
    for meta in children_of.get("metadatas", []):
        sid = meta.get("section_id", "")
        if sid and sid != section_id:
            sub_section_ids.add(sid)

    grandchildren_results = []
    for sub_sid in sub_section_ids:
        gc = col.get(
            where={"parent_id": sub_sid},
            include=["documents", "metadatas"],
        )
        grandchildren_results.append(gc)

    # Merge all into a single evidence list (deduplicated)
    seen: Set[str] = set()
    evidences: List[Evidence] = []

    for result_set in [own, children_of] + grandchildren_results:
        ids = result_set.get("ids", [])
        docs = result_set.get("documents", [])
        metas = result_set.get("metadatas", [])
        for cid, doc, meta in zip(ids, docs, metas):
            if cid not in seen:
                seen.add(cid)
                evidences.append(_meta_to_evidence(meta, doc or "", cid, score=1.0))

    return evidences


# ── Coverage counting ────────────────────────────────────────────────────────

def _count_list_items(text: str) -> int:
    """
    Heuristic: count bullet/numbered items, table rows, or repeated
    structural patterns in a section text.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    count = 0

    # 1) Numbered items: "1.", "1)", "- ", "• ", "* ", "a)", "a."
    for ln in lines:
        if re.match(r"^(\d+[\.\)]\s|[a-zA-Z][\.\)]\s|[-•*]\s|[#]\s+\d)", ln):
            count += 1

    if count >= 2:
        return count

    # 2) Table-row heuristic: lines that are short standalone labels
    #    followed by longer description lines (key-value table pattern).
    #    For PDF tables extracted as text, each "row" often appears as
    #    a short label line followed by a description line.
    #    E.g.:
    #       Belge Yükleme
    #       Kullanıcı PDF ve resim formatlarında belge yükleyebilmeli.
    short_lines = []
    for i, ln in enumerate(lines):
        # A "label" line: short, doesn't end with sentence punctuation,
        # and is followed by a longer line (or is at the end).
        is_short = len(ln) < 50
        no_sent_punct = not ln.endswith((".", "!", "?", ":", ";"))
        starts_upper = ln[0:1].isupper() if ln else False
        if is_short and no_sent_punct and starts_upper:
            # Check the next line is longer (description)
            if i + 1 < len(lines) and len(lines[i + 1]) > len(ln):
                short_lines.append(ln)
            elif i + 1 >= len(lines):
                short_lines.append(ln)

    # Filter out heading-like lines (the section title itself)
    # by removing the first line if it looks like a heading
    if short_lines and lines and short_lines[0] == lines[0]:
        short_lines = short_lines[1:]

    # Also filter known non-data labels (table headers like "İşlev", "Beklenen Davranış")
    # We'll keep them if there are at least 3 (likely actual data rows)
    if len(short_lines) >= 2:
        return len(short_lines)

    # 3) Count sub-section headings within the text
    heading_count = 0
    for ln in lines:
        if re.match(r"^[A-Z0-9]+\.\d+", ln) or re.match(r"^\d+\.\d+", ln):
            heading_count += 1
    if heading_count >= 2:
        return heading_count

    return count


# ── 3) Main retrieval pipeline ───────────────────────────────────────────────

def retrieve(
    index: LocalIndex,
    query: str,
    dense_k: int = 10,
    sparse_k: int = 10,
    final_k: int = 8,
) -> RetrievalResult:
    """
    Full retrieval pipeline with query routing.

    1. Classify intent
    2. Hybrid search to find best-matching section
    3. If section_list → heading-aware section selection → complete subtree
       fetch + coverage info
    4. If normal_qa → return top-k evidence
    """
    intent = classify_query(query)

    # Always start with hybrid search to locate the best section
    hybrid = index.hybrid_search(query, dense_k=dense_k, sparse_k=sparse_k, final_k=final_k)

    if not hybrid.ids:
        return RetrievalResult(
            intent=intent,
            evidences=[],
            section_complete=False,
        )

    # Get metadata for top results
    got = index.store.get(hybrid.ids)
    got_ids = got.get("ids", [])
    got_docs = got.get("documents", [])
    got_metas = got.get("metadatas", [])

    if intent == "section_list":
        best_section_id = _pick_best_section(
            got_ids, got_metas, hybrid.scores, query,
        )

        if best_section_id:
            # Complete section + subtree fetch
            section_evidences = _fetch_section_and_subtree(index, best_section_id)

            # Coverage info from the parent chunk text
            coverage = None
            for ev in section_evidences:
                if ev.kind == "parent" and ev.section_id == best_section_id:
                    n = _count_list_items(ev.text)
                    if n > 0:
                        coverage = CoverageInfo(
                            expected_items=n,
                            heading_path=ev.heading_path,
                            section_id=ev.section_id,
                        )
                    break

            return RetrievalResult(
                intent=intent,
                evidences=section_evidences,
                section_complete=True,
                coverage=coverage,
            )

    # Normal QA: return hybrid top-k evidence
    evidences: List[Evidence] = []
    for cid, doc, meta in zip(got_ids, got_docs, got_metas):
        score = hybrid.scores.get(cid, 0.0)
        evidences.append(_meta_to_evidence(meta, doc or "", cid, score))

    return RetrievalResult(
        intent=intent,
        evidences=evidences,
        section_complete=False,
    )
