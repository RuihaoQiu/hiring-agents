from __future__ import annotations

from typing import TypedDict

import numpy as np

from hiring_agents.schemas import (
    IngestedCandidate,
    NormalizedQuery,
    PipelineOutput,
    RetrievedCandidate,
    ScoredCandidate,
)


class PipelineState(TypedDict, total=False):
    raw_query: str
    candidates: list[IngestedCandidate]
    embeddings: np.ndarray
    normalized: NormalizedQuery
    retrieved: list[RetrievedCandidate]
    ranked: list[ScoredCandidate]
    filters_relaxed: bool
    output: PipelineOutput
