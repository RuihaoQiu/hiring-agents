from __future__ import annotations

from pydantic import BaseModel, Field

from hiring_agents.schemas import NormalizedQuery, WorkEntry


class RichCandidate(BaseModel):
    candidate_id: str
    current_title: str
    current_employer: str
    location: str
    total_yoe: int
    score: int
    summary: str
    work_history: list[WorkEntry]
    skills: list[str]
    gaps: list[str]
    suggestion: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SearchResponse(BaseModel):
    normalized: NormalizedQuery
    retrieved_count: int
    ranked: list[RichCandidate]
    filters_relaxed: bool
