from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from hiring_agents import pipeline as pipeline_module
from hiring_agents.config import EMBEDDING_DIM
from hiring_agents.schemas import (
    Candidate,
    HardFilters,
    IngestedCandidate,
    MustHaveMatch,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
    StructuredResume,
)


def _candidate(cid: str) -> Candidate:
    return Candidate(candidate_id=cid, resume_text="...")


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
    )


def test_pipeline_smoke_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    three = [_candidate(f"c0{i}") for i in range(3)]

    def fake_load_models(path: Path, model: type) -> list[Candidate]:
        return three

    def fake_ingest_all(
        candidates: list[Candidate],
    ) -> tuple[list[IngestedCandidate], np.ndarray]:
        ingested = [_ingested(c.candidate_id) for c in candidates]
        embeddings = np.eye(len(ingested), EMBEDDING_DIM, dtype=np.float32)
        return ingested, embeddings

    def fake_normalize_query(raw: str) -> NormalizedQuery:
        return NormalizedQuery(
            raw=raw,
            hard_filters=HardFilters(),
            core_skills=["Python"],
            nice_to_haves=[],
            canonical_summary="canonical",
        )

    def fake_embed_query(text: str) -> np.ndarray:
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[0] = 1.0
        return v

    async def fake_rerank(
        normalized: NormalizedQuery,
        retrieved: Sequence[RetrievedCandidate],
        top_k: int = 10,
        concurrency: int = 5,
    ) -> list[ScoredCandidate]:
        return [
            ScoredCandidate(
                candidate_id=rc.candidate.candidate_id,
                score=5,
                must_have_matches=[
                    MustHaveMatch(requirement="Python", met=True, evidence="ok")
                ],
                gaps=[],
                one_line_summary=f"{rc.candidate.candidate_id} OK",
            )
            for rc in retrieved[:top_k]
        ]

    monkeypatch.setattr(pipeline_module, "load_models", fake_load_models)
    monkeypatch.setattr(pipeline_module, "ingest_all", fake_ingest_all)
    monkeypatch.setattr(pipeline_module, "normalize_query", fake_normalize_query)
    monkeypatch.setattr(pipeline_module, "embed_query", fake_embed_query)
    monkeypatch.setattr(pipeline_module, "rerank", fake_rerank)

    output = pipeline_module.run_pipeline("python engineer")

    assert output.normalized.raw == "python engineer"
    assert len(output.retrieved) == 3
    assert output.retrieved[0].candidate.candidate_id == "c00"
    assert output.retrieved[0].similarity == pytest.approx(1.0)
    assert len(output.ranked) == 3
    assert all(s.score == 5 for s in output.ranked)
