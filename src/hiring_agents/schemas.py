from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hiring_agents.config import SCORE_MAX, SCORE_MIN


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WorkEntry(_Frozen):
    company: str
    title: str
    start_year: int
    end_year: int | None
    description: str


class StructuredResume(_Frozen):
    location: str
    total_yoe: int
    current_title: str
    skills: list[str]
    work_history: list[WorkEntry]


class Candidate(_Frozen):
    candidate_id: str
    resume_text: str


class IngestedCandidate(_Frozen):
    candidate_id: str
    resume_text: str
    structured: StructuredResume
    summary: str
    inferred_seniority: str | None = None


class GroundTruth(_Frozen):
    role_family: str
    seniority: str
    location: str
    total_yoe: int
    tech_stack: list[str]
    domains: list[str]


class HardFilters(_Frozen):
    location_keywords: list[str] | None = None
    seniority: list[str] | None = None


class NormalizedQuery(_Frozen):
    raw: str
    hard_filters: HardFilters
    core_skills: list[str]
    nice_to_haves: list[str]
    canonical_summary: str


class RetrievedCandidate(_Frozen):
    candidate: IngestedCandidate
    similarity: float


class MustHaveMatch(_Frozen):
    requirement: str
    met: bool
    evidence: str


class ScoredCandidate(_Frozen):
    candidate_id: str
    score: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    must_have_matches: list[MustHaveMatch]
    gaps: list[str]
    one_line_summary: str

    @field_validator("score")
    @classmethod
    def _score_in_range(cls, v: int) -> int:
        if not SCORE_MIN <= v <= SCORE_MAX:
            raise ValueError(f"score {v} outside [{SCORE_MIN}, {SCORE_MAX}]")
        return v


class PipelineOutput(_Frozen):
    normalized: NormalizedQuery
    retrieved: list[RetrievedCandidate]
    ranked: list[ScoredCandidate]
