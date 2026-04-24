from __future__ import annotations

import pytest

from hiring_agents import rerank as rerank_module
from hiring_agents.rerank import rerank
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    MustHaveMatch,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
    StructuredResume,
)


def _retrieved(cid: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        candidate=IngestedCandidate(
            candidate_id=cid,
            resume_text="...",
            structured=StructuredResume(
                location="Berlin",
                total_yoe=5,
                current_title="Engineer",
                skills=["Python"],
                work_history=[],
            ),
            summary=f"summary {cid}",
        ),
        similarity=0.5,
    )


def _scored(cid: str, score: int, met: int, total: int = 2) -> ScoredCandidate:
    matches = [
        MustHaveMatch(requirement=f"r{i}", met=i < met, evidence="")
        for i in range(total)
    ]
    return ScoredCandidate(
        candidate_id=cid,
        score=score,
        must_have_matches=matches,
        gaps=[],
        one_line_summary="",
    )


def _normalized() -> NormalizedQuery:
    return NormalizedQuery(
        raw="raw",
        hard_filters=HardFilters(),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="canonical",
    )


async def test_empty_retrieved_returns_empty() -> None:
    out = await rerank(_normalized(), [], top_k=5)
    assert out == []


async def test_sort_by_score_then_met_and_truncates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = {
        "c01": _scored("c01", score=3, met=2),
        "c02": _scored("c02", score=5, met=0),
        "c03": _scored("c03", score=5, met=2),
        "c04": _scored("c04", score=4, met=1),
    }

    async def fake_score_one(
        normalized: NormalizedQuery, rc: RetrievedCandidate
    ) -> ScoredCandidate:
        return table[rc.candidate.candidate_id]

    monkeypatch.setattr(rerank_module, "_score_one", fake_score_one)

    retrieved = [_retrieved(cid) for cid in ["c01", "c02", "c03", "c04"]]
    out = await rerank(_normalized(), retrieved, top_k=3)

    assert [s.candidate_id for s in out] == ["c03", "c02", "c04"]


async def test_top_k_larger_than_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_score_one(
        normalized: NormalizedQuery, rc: RetrievedCandidate
    ) -> ScoredCandidate:
        return _scored(rc.candidate.candidate_id, score=4, met=1)

    monkeypatch.setattr(rerank_module, "_score_one", fake_score_one)

    retrieved = [_retrieved(cid) for cid in ["c01", "c02"]]
    out = await rerank(_normalized(), retrieved, top_k=10)
    assert len(out) == 2
