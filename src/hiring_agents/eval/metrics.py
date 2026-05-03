from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in positives)
    return hits / len(positives)


def precision_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in positives)
    return hits / k


def ndcg_at_k(ranked_ids: list[str], positives: set[str], k: int) -> float:
    if not positives:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(ranked_ids[:k])
        if cid in positives
    )
    ideal_hits = min(k, len(positives))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
