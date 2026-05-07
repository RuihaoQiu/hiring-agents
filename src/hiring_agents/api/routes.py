from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from hiring_agents.api.models import RichCandidate, SearchRequest, SearchResponse
from hiring_agents.schemas import IngestedCandidate, ScoredCandidate

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
