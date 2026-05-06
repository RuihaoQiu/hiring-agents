from __future__ import annotations

from pydantic import BaseModel, Field

from hiring_agents.schemas import NormalizedQuery, ScoredCandidate


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SearchResponse(BaseModel):
    normalized: NormalizedQuery
    retrieved_count: int
    ranked: list[ScoredCandidate]
    filters_relaxed: bool
