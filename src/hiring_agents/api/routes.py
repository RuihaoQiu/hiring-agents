from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from hiring_agents.api.models import SearchRequest, SearchResponse

router = APIRouter()


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
    return SearchResponse(
        normalized=output.normalized,
        retrieved_count=len(output.retrieved),
        ranked=output.ranked,
        filters_relaxed=state.get("filters_relaxed", False),
    )
