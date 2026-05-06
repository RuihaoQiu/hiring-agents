from __future__ import annotations

import numpy as np
import pytest

from hiring_agents.retrieve import (
    apply_hard_filters,
    cosine_similarity,
    retrieve_top_k,
)
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    StructuredResume,
)


def _candidate(
    cid: str, location: str, seniority: str | None = None
) -> IngestedCandidate:
    return IngestedCandidate(
        candidate_id=cid,
        resume_text="...",
        structured=StructuredResume(
            location=location,
            total_yoe=5,
            current_title="Engineer",
            skills=["Python"],
            work_history=[],
        ),
        summary=f"summary {cid}",
        inferred_seniority=seniority,
    )


@pytest.fixture
def pool() -> list[IngestedCandidate]:
    return [
        _candidate("c01", "Berlin, Germany", "senior"),
        _candidate("c02", "Munich, Germany", "junior"),
        _candidate("c03", "Paris, France", "staff"),
        _candidate("c04", "Berlin, Germany", "mid"),
    ]


def test_cosine_identity_and_orthogonal() -> None:
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    sims = cosine_similarity(q, docs)
    assert sims == pytest.approx([1.0, 0.0, 1.0])


def test_cosine_opposite() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([[-1.0, 0.0]], dtype=np.float32)
    assert cosine_similarity(q, docs) == pytest.approx([-1.0])


def test_cosine_handles_zero_doc_without_nan() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array([[0.0, 0.0]], dtype=np.float32)
    sims = cosine_similarity(q, docs)
    assert not np.isnan(sims).any()


def test_no_filters_keeps_all(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(pool, HardFilters()) == [0, 1, 2, 3]


def test_location_substring(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(pool, HardFilters(location_keywords=["Germany"])) == [
        0,
        1,
        3,
    ]


def test_location_case_insensitive(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(pool, HardFilters(location_keywords=["berlin"])) == [
        0,
        3,
    ]


def test_location_multi_keyword_is_or(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(
        pool, HardFilters(location_keywords=["Paris", "Munich"])
    ) == [1, 2]


def test_seniority_single_level(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(pool, HardFilters(seniority=["senior"])) == [0]


def test_seniority_multi_level(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(
        pool, HardFilters(seniority=["senior", "staff"])
    ) == [0, 2]


def test_seniority_unknown_passes(pool: list[IngestedCandidate]) -> None:
    candidates = [_candidate("cx", "Berlin", None)]
    assert apply_hard_filters(candidates, HardFilters(seniority=["senior"])) == [0]


def test_combined_location_and_seniority(pool: list[IngestedCandidate]) -> None:
    filters = HardFilters(location_keywords=["Germany"], seniority=["senior"])
    assert apply_hard_filters(pool, filters) == [0]


def test_empty_result_when_no_location_match(pool: list[IngestedCandidate]) -> None:
    assert apply_hard_filters(pool, HardFilters(location_keywords=["Tokyo"])) == []


def test_top_k_sorted_descending() -> None:
    docs = np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    out = retrieve_top_k(q, docs, allowed_indices=[0, 1, 2], k=3)
    indices = [i for i, _ in out]
    sims = [s for _, s in out]
    assert indices == [0, 2, 1]
    assert sims == sorted(sims, reverse=True)


def test_top_k_respects_allowed_subset() -> None:
    docs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    out = retrieve_top_k(q, docs, allowed_indices=[1, 2], k=2)
    assert [i for i, _ in out] == [1, 2]


def test_top_k_empty_pool() -> None:
    docs = np.array([[1, 0]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    assert retrieve_top_k(q, docs, allowed_indices=[], k=5) == []


def test_top_k_k_larger_than_pool_returns_all() -> None:
    docs = np.array([[1, 0], [0, 1]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    out = retrieve_top_k(q, docs, allowed_indices=[0, 1], k=5)
    assert len(out) == 2


def test_top_k_zero_k_returns_empty() -> None:
    docs = np.array([[1, 0]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    assert retrieve_top_k(q, docs, allowed_indices=[0], k=0) == []
