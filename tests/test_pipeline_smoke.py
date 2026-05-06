from __future__ import annotations

import numpy as np
import pytest

from hiring_agents.config import EMBEDDING_DIM
from hiring_agents.pipeline import run_pipeline
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    MustHaveMatch,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
    StructuredResume,
)


def _ingested(cid: str) -> IngestedCandidate:
    return IngestedCandidate(
        candidate_id=cid,
        resume_text="...",
        structured=StructuredResume(
            location="Berlin, Germany",
            total_yoe=5,
            current_title="Engineer",
            skills=["Python"],
            work_history=[],
        ),
        summary=f"summary {cid}",
        inferred_seniority="senior",
    )


def test_pipeline_smoke_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [_ingested(f"c0{i}") for i in range(3)]
    embeddings = np.eye(len(candidates), EMBEDDING_DIM, dtype=np.float32)

    normalized = NormalizedQuery(
        raw="python engineer",
        hard_filters=HardFilters(),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="canonical",
    )

    async def fake_rerank(
        norm: NormalizedQuery,
        retrieved: list[RetrievedCandidate],
        top_k: int = 10,
        concurrency: int = 5,
    ) -> list[ScoredCandidate]:
        return [
            ScoredCandidate(
                candidate_id=rc.candidate.candidate_id,
                score=5,
                must_have_matches=[MustHaveMatch(requirement="Python", met=True, evidence="ok")],
                gaps=[],
                one_line_summary=f"{rc.candidate.candidate_id} OK",
            )
            for rc in retrieved[:top_k]
        ]

    query_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query_vec[0] = 1.0

    monkeypatch.setattr("hiring_agents.graph.load_node", lambda s: {"candidates": candidates, "embeddings": embeddings})
    monkeypatch.setattr("hiring_agents.graph.normalize_query", lambda raw: normalized)
    monkeypatch.setattr("hiring_agents.graph.embed_query", lambda text: query_vec)
    monkeypatch.setattr("hiring_agents.graph.rerank", fake_rerank)

    output = run_pipeline("python engineer")

    assert output.normalized.raw == "python engineer"
    assert len(output.retrieved) == 3
    assert output.retrieved[0].candidate.candidate_id == "c00"
    assert output.retrieved[0].similarity == pytest.approx(1.0)
    assert len(output.ranked) == 3
    assert all(s.score == 5 for s in output.ranked)
