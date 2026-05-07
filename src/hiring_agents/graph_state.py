from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np

from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    NormalizedQuery,
    PipelineOutput,
    RetrievedCandidate,
    ScoredCandidate,
)


class PipelineState(TypedDict, total=False):
    raw_query: str
    mode: Literal["keyword", "jd", "strict"]
    preset_filters: HardFilters | None
    candidates: list[IngestedCandidate]
    embeddings: np.ndarray
    normalized: NormalizedQuery
    retrieved: list[RetrievedCandidate]
    ranked: list[ScoredCandidate]
    filters_relaxed: bool
    output: PipelineOutput
