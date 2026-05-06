from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from hiring_agents.api.app import create_app
from hiring_agents.config import EMBEDDING_DIM
from hiring_agents.graph import build_graph
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    MustHaveMatch,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
    StructuredResume,
    WorkEntry,
)


def _make_ingested(cid: str) -> IngestedCandidate:
    work = WorkEntry(company="Acme", title="Senior Engineer", start_year=2020, end_year=2024, description="...")
    structured = StructuredResume(
        location="Berlin, Germany", total_yoe=5, current_title="Senior Engineer",
        skills=["Python"], work_history=[work],
    )
    return IngestedCandidate(
        candidate_id=cid, resume_text="...", structured=structured,
        summary="A senior engineer.", inferred_seniority="senior",
    )


def _make_scored(cid: str) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id=cid, score=4,
        must_have_matches=[MustHaveMatch(requirement="Python", met=True, evidence="...")],
        gaps=[], one_line_summary="Good match.",
    )


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    candidates = [_make_ingested(f"c{i:04d}") for i in range(3)]
    embeddings = np.eye(3, EMBEDDING_DIM, dtype=np.float32)
    normalized = NormalizedQuery(
        raw="senior python dev",
        hard_filters=HardFilters(),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="Senior Python developer.",
    )

    async def _fake_rerank(norm: NormalizedQuery, retrieved: list[RetrievedCandidate], **kw) -> list[ScoredCandidate]:
        return [_make_scored(r.candidate.candidate_id) for r in retrieved]

    query_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query_vec[0] = 1.0

    monkeypatch.setattr("hiring_agents.graph.normalize_query", lambda raw: normalized)
    monkeypatch.setattr("hiring_agents.graph.embed_query", lambda text: query_vec)
    monkeypatch.setattr("hiring_agents.graph.rerank", _fake_rerank)

    # ASGITransport does not trigger the FastAPI lifespan, so set state directly.
    _app = create_app()
    _app.state.candidates = candidates
    _app.state.embeddings = embeddings
    _app.state.graph = build_graph()
    return _app


@pytest.fixture
async def client(app) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_search_happy_path(client: AsyncClient) -> None:
    resp = await client.post("/search", json={"query": "senior python dev"})
    assert resp.status_code == 200
    data = resp.json()
    assert "normalized" in data
    assert "ranked" in data
    assert data["retrieved_count"] == 3
    assert len(data["ranked"]) == 3
    assert data["filters_relaxed"] is False


async def test_search_empty_query_returns_422(client: AsyncClient) -> None:
    resp = await client.post("/search", json={"query": ""})
    assert resp.status_code == 422


async def test_search_no_cache_returns_503() -> None:
    _app = create_app()
    _app.state.candidates = None
    _app.state.embeddings = None
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as c:
        resp = await c.post("/search", json={"query": "python dev"})
    assert resp.status_code == 503
