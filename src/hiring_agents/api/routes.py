from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from hiring_agents.api.models import RichCandidate, SearchRequest, SearchResponse
from hiring_agents.config import RERANK_TOP_K, RETRIEVAL_TOP_K, SKIP_RERANK
from hiring_agents.llm.embeddings import embed_query
from hiring_agents.normalize import normalize_jd, normalize_query
from hiring_agents.rerank import rerank
from hiring_agents.retrieve import apply_hard_filters, retrieve_top_k
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
)

router = APIRouter()


def _to_rich(sc: ScoredCandidate, ingested: IngestedCandidate) -> RichCandidate:
    s = ingested.structured
    current_employer = (
        max(s.work_history, key=lambda e: e.start_year).company
        if s.work_history
        else "—"
    )
    return RichCandidate(
        candidate_id=sc.candidate_id,
        current_title=s.current_title,
        current_employer=current_employer,
        location=s.location,
        total_yoe=s.total_yoe,
        score=sc.score,
        summary=ingested.summary,
        work_history=list(s.work_history),
        skills=list(s.skills),
        gaps=list(sc.gaps),
        suggestion=sc.one_line_summary,
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    if request.app.state.candidates is None:
        raise HTTPException(status_code=503, detail="Cache not ready — run 'hiring-agents ingest' first.")

    graph = request.app.state.graph
    initial_state = {
        "raw_query": body.query,
        "mode": body.mode,
        "preset_filters": body.hard_filters,
        "candidates": request.app.state.candidates,
        "embeddings": request.app.state.embeddings,
    }
    state = await asyncio.to_thread(graph.invoke, initial_state)
    output = state["output"]

    by_id = {rc.candidate.candidate_id: rc.candidate for rc in output.retrieved}
    ranked = [_to_rich(sc, by_id[sc.candidate_id]) for sc in output.ranked]

    return SearchResponse(
        normalized=output.normalized,
        retrieved_count=len(output.retrieved),
        ranked=ranked,
        filters_relaxed=state.get("filters_relaxed", False),
    )


@router.post("/search/stream")
async def search_stream(body: SearchRequest, request: Request) -> StreamingResponse:
    if request.app.state.candidates is None:
        raise HTTPException(status_code=503, detail="Cache not ready — run 'hiring-agents ingest' first.")

    ingested = request.app.state.candidates
    embeddings = request.app.state.embeddings

    async def generate() -> AsyncGenerator[str, None]:
        mode = body.mode
        if mode == "strict":
            normalized = NormalizedQuery(
                raw=body.query,
                hard_filters=body.hard_filters or HardFilters(),
                core_skills=[],
                nice_to_haves=[],
                canonical_summary=body.query,
            )
        elif mode == "jd":
            normalized = await asyncio.to_thread(normalize_jd, body.query)
        else:
            normalized = await asyncio.to_thread(normalize_query, body.query)

        yield json.dumps({"type": "normalized", "data": normalized.model_dump()}) + "\n"

        filters_relaxed = False
        allowed = apply_hard_filters(ingested, normalized.hard_filters)
        if not allowed and mode != "strict":
            relaxed = HardFilters(location_keywords=normalized.hard_filters.location_keywords)
            normalized = normalized.model_copy(update={"hard_filters": relaxed})
            allowed = apply_hard_filters(ingested, normalized.hard_filters)
            filters_relaxed = True

        qvec = await asyncio.to_thread(embed_query, normalized.canonical_summary)
        top = retrieve_top_k(qvec, embeddings, allowed, k=RETRIEVAL_TOP_K)
        retrieved = [RetrievedCandidate(candidate=ingested[idx], similarity=sim) for idx, sim in top]

        if SKIP_RERANK:
            ranked_scored = [
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
            ranked_scored = await rerank(normalized, retrieved)

        yield json.dumps({"type": "retrieved", "count": len(retrieved), "filters_relaxed": filters_relaxed}) + "\n"

        by_id = {rc.candidate.candidate_id: rc.candidate for rc in retrieved}
        for sc in ranked_scored:
            rich = _to_rich(sc, by_id[sc.candidate_id])
            yield json.dumps({"type": "candidate", "data": rich.model_dump()}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
