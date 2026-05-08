from __future__ import annotations

import asyncio

import numpy as np

from hiring_agents.config import RERANK_TOP_K, RETRIEVAL_TOP_K, SENIORITY_VOCAB, SKIP_RERANK
from hiring_agents.llm.embeddings import embed_query
from hiring_agents.pipeline.normalize import normalize_jd, normalize_query
from hiring_agents.pipeline.rerank import rerank
from hiring_agents.pipeline.retrieve import apply_hard_filters, retrieve_top_k
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    PipelineOutput,
    RetrievedCandidate,
    ScoredCandidate,
)

_JD_THRESHOLD = 200


async def run_search(
    raw: str,
    ingested: list[IngestedCandidate],
    embeddings: np.ndarray,
    *,
    location: str | None = None,
    seniority: list[str] | None = None,
    strict: bool = False,
) -> tuple[PipelineOutput, bool]:
    """Run normalize → filter → retrieve → rerank. Returns (output, filters_relaxed)."""
    normalized = await asyncio.to_thread(
        normalize_jd if len(raw) > _JD_THRESHOLD else normalize_query, raw
    )
    if location or seniority:
        explicit = HardFilters(
            location_keywords=[location] if location else None,
            seniority=[s for s in (seniority or []) if s in SENIORITY_VOCAB] or None,
        )
        normalized = normalized.model_copy(update={"hard_filters": explicit})

    filters_relaxed = False
    allowed = apply_hard_filters(ingested, normalized.hard_filters)
    if not allowed and not strict:
        relaxed = HardFilters(location_keywords=normalized.hard_filters.location_keywords)
        normalized = normalized.model_copy(update={"hard_filters": relaxed})
        allowed = apply_hard_filters(ingested, normalized.hard_filters)
        filters_relaxed = True

    qvec = await asyncio.to_thread(embed_query, normalized.canonical_summary)
    top = retrieve_top_k(qvec, embeddings, allowed, k=RETRIEVAL_TOP_K)
    retrieved = [RetrievedCandidate(candidate=ingested[idx], similarity=sim) for idx, sim in top]

    if SKIP_RERANK:
        ranked: list[ScoredCandidate] = [
            ScoredCandidate(
                candidate_id=rc.candidate.candidate_id,
                score=3,
                must_have_matches=[],
                gaps=[],
                one_line_summary="",
            )
            for rc in retrieved[:RERANK_TOP_K]
        ]
    else:
        ranked = await rerank(normalized, retrieved)

    output = PipelineOutput(normalized=normalized, retrieved=retrieved, ranked=ranked)
    return output, filters_relaxed
