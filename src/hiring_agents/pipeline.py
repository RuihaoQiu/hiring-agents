from __future__ import annotations

import asyncio
import logging

from hiring_agents.config import (
    CANDIDATES_PATH,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
)
from hiring_agents.ingest import ingest_all
from hiring_agents.io_utils import load_models
from hiring_agents.llm.embeddings import embed_query
from hiring_agents.normalize import normalize_query
from hiring_agents.rerank import rerank
from hiring_agents.retrieve import apply_hard_filters, retrieve_top_k
from hiring_agents.schemas import Candidate, PipelineOutput, RetrievedCandidate

logger = logging.getLogger(__name__)


def run_pipeline(raw_query: str) -> PipelineOutput:
    candidates = load_models(CANDIDATES_PATH, Candidate)
    ingested, embeddings = ingest_all(candidates)
    normalized = normalize_query(raw_query)
    allowed = apply_hard_filters(ingested, normalized.hard_filters)
    logger.info("hard filters: %d/%d passed", len(allowed), len(ingested))
    qvec = embed_query(normalized.canonical_summary)
    top = retrieve_top_k(qvec, embeddings, allowed, k=RETRIEVAL_TOP_K)
    retrieved = [
        RetrievedCandidate(candidate=ingested[idx], similarity=sim)
        for idx, sim in top
    ]
    ranked = asyncio.run(rerank(normalized, retrieved, top_k=RERANK_TOP_K))
    return PipelineOutput(normalized=normalized, retrieved=retrieved, ranked=ranked)
