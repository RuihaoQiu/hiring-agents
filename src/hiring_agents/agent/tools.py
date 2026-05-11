from __future__ import annotations

import asyncio
import json
import logging

import chainlit as cl
import numpy as np
from langchain_core.tools import tool

from hiring_agents.config import CANDIDATES_PATH, SEARCH_DEFAULT_TOP_K
from hiring_agents.io_utils import load_models
from hiring_agents.llm.tracing import observe_span
from hiring_agents.pipeline.ingest import ingest_all
from hiring_agents.pipeline.search import run_search
from hiring_agents.schemas import Candidate, IngestedCandidate, ScoredCandidate

logger = logging.getLogger(__name__)

_candidates: list[IngestedCandidate] | None = None
_embeddings: np.ndarray | None = None


async def _ensure_loaded() -> tuple[list[IngestedCandidate], np.ndarray]:
    global _candidates, _embeddings
    if _candidates is None:
        raw = load_models(CANDIDATES_PATH, Candidate)
        _candidates, _embeddings = await asyncio.to_thread(ingest_all, raw)
    return _candidates, _embeddings  # type: ignore[return-value]


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


_CACHE_KEY = "search_pool"
_OFFSET_KEY = "search_offset"


@tool
async def search_candidates(
    query: str,
    location: str | None = None,
    seniority: list[str] | None = None,
    strict: bool = False,
    top_k: int = SEARCH_DEFAULT_TOP_K,
) -> str:
    """Search the candidate database for people matching the hiring requirements.
    Always starts a fresh search and resets pagination.

    Args:
        query: Role description, required skills, or full job description text.
        location: City or region to filter by (e.g. 'Berlin', 'London'). None means any location.
        seniority: Seniority levels to include (junior/mid/senior/staff/principal). None means any.
        strict: If True, return empty results rather than relaxing filters when no match is found.
        top_k: Number of candidates to return in this batch (default 5).

    Returns:
        JSON array of ranked candidates sorted by fit score.
    """
    candidates, embeddings = await _ensure_loaded()
    with observe_span(name="search_candidates"):
        output, filters_relaxed = await run_search(
            query, candidates, embeddings,
            location=location, seniority=seniority, strict=strict,
        )
    by_id = {rc.candidate.candidate_id: rc.candidate for rc in output.retrieved}
    pool = [_to_result(sc, by_id[sc.candidate_id], filters_relaxed) for sc in output.ranked]
    cl.user_session.set(_CACHE_KEY, pool)
    cl.user_session.set(_OFFSET_KEY, top_k)
    return json.dumps(pool[:top_k])


@tool
async def show_more_candidates(n: int = SEARCH_DEFAULT_TOP_K) -> str:
    """Return the next n candidates from the most recent search without re-running the pipeline.
    Use this when the user asks to see more results or the next batch.

    Args:
        n: How many more candidates to return (default 5).

    Returns:
        JSON array of the next ranked candidates, or a message if no more are available.
    """
    pool: list[dict] = cl.user_session.get(_CACHE_KEY) or []
    offset: int = cl.user_session.get(_OFFSET_KEY) or 0
    if not pool:
        return "No previous search found. Please run a search first."
    batch = pool[offset: offset + n]
    if not batch:
        return "No more candidates in the current results. Try a new search or adjust filters."
    cl.user_session.set(_OFFSET_KEY, offset + n)
    return json.dumps(batch)


