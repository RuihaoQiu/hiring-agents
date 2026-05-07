from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hiring_agents.schemas import HardFilters, NormalizedQuery, WorkEntry


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
    mode: Literal["keyword", "jd", "strict"] = "keyword"
    hard_filters: HardFilters | None = None


class SearchResponse(BaseModel):
    normalized: NormalizedQuery
    retrieved_count: int
    ranked: list[RichCandidate]
    filters_relaxed: bool
