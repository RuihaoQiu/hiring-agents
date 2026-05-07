from __future__ import annotations

import asyncio
import json
import logging

import chainlit as cl
import numpy as np
from langchain_core.tools import tool

from hiring_agents.config import (
    CANDIDATES_PATH,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    SENIORITY_VOCAB,
    SKIP_RERANK,
)
from hiring_agents.ingest import ingest_all
from hiring_agents.io_utils import load_models
from hiring_agents.llm.embeddings import embed_query
from hiring_agents.normalize import normalize_jd, normalize_query
from hiring_agents.rerank import rerank
from hiring_agents.retrieve import apply_hard_filters, retrieve_top_k
from hiring_agents.schemas import (
    Candidate,
    HardFilters,
    IngestedCandidate,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_candidates: list[IngestedCandidate] | None = None
_embeddings: np.ndarray | None = None


async def _ensure_loaded() -> tuple[list[IngestedCandidate], np.ndarray]:
    global _candidates, _embeddings
    if _candidates is None:
        raw = load_models(CANDIDATES_PATH, Candidate)
        _candidates, _embeddings = await asyncio.to_thread(ingest_all, raw)
    return _candidates, _embeddings  # type: ignore[return-value]


async def _normalize(
    query: str, location: str | None, seniority: list[str] | None
) -> NormalizedQuery:
    normalized = await asyncio.to_thread(
        normalize_jd if len(query) > 200 else normalize_query, query
    )
    explicit = HardFilters(
        location_keywords=[location] if location else None,
        seniority=[s for s in (seniority or []) if s in SENIORITY_VOCAB] or None,
    )
    if explicit.location_keywords or explicit.seniority:
        normalized = normalized.model_copy(update={"hard_filters": explicit})
    return normalized


async def _retrieve(
    candidates: list[IngestedCandidate],
    embeddings: np.ndarray,
    normalized: NormalizedQuery,
    strict: bool,
) -> tuple[list[RetrievedCandidate], bool]:
    filters_relaxed = False
    allowed = apply_hard_filters(candidates, normalized.hard_filters)
    if not allowed and not strict:
        relaxed = HardFilters(location_keywords=normalized.hard_filters.location_keywords)
        normalized = normalized.model_copy(update={"hard_filters": relaxed})
        allowed = apply_hard_filters(candidates, normalized.hard_filters)
        filters_relaxed = True
    qvec = await asyncio.to_thread(embed_query, normalized.canonical_summary)
    top = retrieve_top_k(qvec, embeddings, allowed, k=RETRIEVAL_TOP_K)
    retrieved = [RetrievedCandidate(candidate=candidates[idx], similarity=sim) for idx, sim in top]
    return retrieved, filters_relaxed


async def _score(
    normalized: NormalizedQuery, retrieved: list[RetrievedCandidate]
) -> list[ScoredCandidate]:
    if SKIP_RERANK:
        return [
            ScoredCandidate(
                candidate_id=rc.candidate.candidate_id,
                score=3,
                must_have_matches=[],
                gaps=[],
                one_line_summary="",
            )
            for rc in retrieved[:RERANK_TOP_K]
        ]
    return await rerank(normalized, retrieved)


def _to_result(sc: ScoredCandidate, cand: IngestedCandidate, filters_relaxed: bool) -> dict:
    s = cand.structured
    employer = max(s.work_history, key=lambda e: e.start_year).company if s.work_history else "—"
    return {
        "candidate_id": sc.candidate_id,
        "current_title": s.current_title,
        "current_employer": employer,
        "location": s.location,
        "total_yoe": s.total_yoe,
        "score": sc.score,
        "skills": list(s.skills),
        "gaps": list(sc.gaps),
        "suggestion": sc.one_line_summary,
        "filters_relaxed": filters_relaxed,
        "work_history": [
            {"title": e.title, "company": e.company, "start_year": e.start_year, "end_year": e.end_year}
            for e in s.work_history
        ],
    }


@tool
async def search_candidates(
    query: str,
    location: str | None = None,
    seniority: list[str] | None = None,
    strict: bool = False,
) -> str:
    """Search the candidate database for people matching the hiring requirements.

    Args:
        query: Role description, required skills, or full job description text.
        location: City or region to filter by (e.g. 'Berlin', 'London'). None means any location.
        seniority: Seniority levels to include (junior/mid/senior/staff/principal). None means any.
        strict: If True, return empty results rather than relaxing filters when no match is found.

    Returns:
        JSON array of ranked candidates sorted by fit score.
    """
    candidates, embeddings = await _ensure_loaded()
    normalized = await _normalize(query, location, seniority)
    retrieved, filters_relaxed = await _retrieve(candidates, embeddings, normalized, strict)
    scored = await _score(normalized, retrieved)
    by_id = {rc.candidate.candidate_id: rc.candidate for rc in retrieved}
    results = [_to_result(sc, by_id[sc.candidate_id], filters_relaxed) for sc in scored]
    return json.dumps(results)


@tool
async def add_to_shortlist(candidate_id: str, candidate_name: str) -> str:
    """Add a candidate to the recruiter's shortlist.

    Args:
        candidate_id: The candidate's ID (e.g. 'C042').
        candidate_name: Display name or title for the confirmation message.
    """
    shortlist: list[dict] = cl.user_session.get("shortlist") or []
    if any(c["candidate_id"] == candidate_id for c in shortlist):
        return f"{candidate_name} is already in the shortlist."
    shortlist.append({"candidate_id": candidate_id, "name": candidate_name})
    cl.user_session.set("shortlist", shortlist)
    return f"Added {candidate_name} to shortlist. Total: {len(shortlist)}."


@tool
async def get_shortlist() -> str:
    """Return the current shortlist of candidates."""
    shortlist: list[dict] = cl.user_session.get("shortlist") or []
    if not shortlist:
        return "The shortlist is empty."
    lines = [
        f"{i + 1}. {c.get('name', c['candidate_id'])} ({c['candidate_id']})"
        for i, c in enumerate(shortlist)
    ]
    return "Current shortlist:\n" + "\n".join(lines)


@tool
async def clear_shortlist() -> str:
    """Clear all candidates from the shortlist."""
    cl.user_session.set("shortlist", [])
    return "Shortlist cleared."
