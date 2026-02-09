from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .models import Chunk


def simple_tokenize(text: str) -> List[str]:
    # TR/EN friendly-ish tokenizer (deterministic, dependency-free)
    text = text.lower()
    text = re.sub(r"[^0-9a-zçğıöşüâîû\-]+", " ", text)
    tokens = [t for t in text.split() if t]

    # Language-agnostic robustness:
    # add character 3-grams to improve matching for suffixes/typos (e.g., "adres" vs "adresini")
    ngrams: list[str] = []
    for tok in tokens:
        # avoid exploding token count on very long strings (IDs/addresses)
        if len(tok) < 5 or len(tok) > 24:
            continue
        # generate up to 12 trigrams per token
        limit = min(len(tok) - 2, 12)
        for i in range(limit):
            ng = tok[i : i + 3]
            ngrams.append(f"~{ng}")

    return tokens + ngrams


@dataclass
class BM25Index:
    # chunk_id -> token list
    tokens_by_id: Dict[str, List[str]]
    bm25: BM25Okapi
    ids: List[str]

    @classmethod
    def build(cls, chunks: List[Chunk]) -> "BM25Index":
        # Index child chunks for retrieval granularity (parents are fetched later)
        child_chunks = [c for c in chunks if c.kind == "child"]
        ids = [c.chunk_id for c in child_chunks]
        toks = [simple_tokenize(c.text) for c in child_chunks]
        bm25 = BM25Okapi(toks)
        return cls(tokens_by_id=dict(zip(ids, toks)), bm25=bm25, ids=ids)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        q = simple_tokenize(query)
        if not q or not self.ids:
            return []
        scores = self.bm25.get_scores(q)
        # top_k indices by score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: List[Tuple[str, float]] = []
        for i in ranked:
            s = float(scores[i])
            if s <= 0:
                continue
            out.append((self.ids[i], s))
        return out

