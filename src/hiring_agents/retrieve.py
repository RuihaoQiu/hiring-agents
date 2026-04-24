from __future__ import annotations

import numpy as np

from hiring_agents.schemas import HardFilters, IngestedCandidate


def apply_hard_filters(
    candidates: list[IngestedCandidate], filters: HardFilters
) -> list[int]:
    allowed: list[int] = []
    for i, c in enumerate(candidates):
        if not _passes_location(c, filters.location_keywords):
            continue
        if not _passes_yoe(c, filters.min_yoe, filters.max_yoe):
            continue
        allowed.append(i)
    return allowed


def cosine_similarity(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query)
    d_norms = np.linalg.norm(docs, axis=1)
    denom = q_norm * d_norms
    safe_denom = np.where(denom == 0, 1.0, denom)
    return (docs @ query) / safe_denom


def retrieve_top_k(
    query: np.ndarray,
    docs: np.ndarray,
    allowed_indices: list[int],
    k: int,
) -> list[tuple[int, float]]:
    if not allowed_indices or k <= 0:
        return []
    subset = docs[allowed_indices]
    sims = cosine_similarity(query, subset)
    n = min(k, len(sims))
    part = np.argpartition(-sims, n - 1)[:n]
    order = part[np.argsort(-sims[part])]
    return [(allowed_indices[int(i)], float(sims[int(i)])) for i in order]


def _passes_location(
    candidate: IngestedCandidate, keywords: list[str] | None
) -> bool:
    if not keywords:
        return True
    location = candidate.structured.location.lower()
    return any(kw.lower() in location for kw in keywords)


def _passes_yoe(
    candidate: IngestedCandidate, min_yoe: int | None, max_yoe: int | None
) -> bool:
    yoe = candidate.structured.total_yoe
    if min_yoe is not None and yoe < min_yoe:
        return False
    return max_yoe is None or yoe <= max_yoe
