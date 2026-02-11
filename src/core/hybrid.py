from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def rrf_fuse(
    dense_ids: List[str],
    sparse_ids: List[str],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.

    score(d) = sum_i w_i / (k + rank_i(d))
    where rank is 1-based.
    """
    scores: Dict[str, float] = {}
    for rank, _id in enumerate(dense_ids, start=1):
        scores[_id] = scores.get(_id, 0.0) + dense_weight / (k + rank)
    for rank, _id in enumerate(sparse_ids, start=1):
        scores[_id] = scores.get(_id, 0.0) + sparse_weight / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class HybridResult:
    ids: List[str]
    scores: Dict[str, float]

