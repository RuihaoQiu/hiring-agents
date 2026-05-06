from __future__ import annotations

import pytest
from pydantic import ValidationError

from hiring_agents.schemas import (
    HardFilters,
    MustHaveMatch,
    NormalizedQuery,
    ScoredCandidate,
)


def test_hard_filters_all_optional() -> None:
    hf = HardFilters()
    assert hf.location_keywords is None
    assert hf.seniority is None


def test_normalized_query_round_trips() -> None:
    q = NormalizedQuery(
        raw="senior python dev germany",
        hard_filters=HardFilters(location_keywords=["Germany"], seniority=["senior", "staff"]),
        core_skills=["Python"],
        nice_to_haves=["Kafka"],
        canonical_summary="Senior backend engineer in Germany with Python experience.",
    )
    assert NormalizedQuery.model_validate(q.model_dump()) == q


def test_scored_candidate_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        ScoredCandidate(
            candidate_id="c0001",
            score=6,
            must_have_matches=[MustHaveMatch(requirement="Python", met=True, evidence="...")],
            gaps=[],
            one_line_summary="...",
        )


def test_models_are_frozen() -> None:
    hf = HardFilters(seniority=["senior"])
    with pytest.raises(ValidationError):
        hf.seniority = ["junior"]  # type: ignore[misc]
