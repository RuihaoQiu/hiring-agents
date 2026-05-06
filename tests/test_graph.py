from __future__ import annotations

import numpy as np
import pytest

from hiring_agents.graph import build_graph, output_node, relax_filters_node
from hiring_agents.schemas import (
    HardFilters,
    IngestedCandidate,
    MustHaveMatch,
    NormalizedQuery,
    PipelineOutput,
    RetrievedCandidate,
    ScoredCandidate,
    StructuredResume,
    WorkEntry,
)

EMBED_DIM = 8  # small dimension sufficient for retrieve_top_k logic


def _make_ingested(candidate_id: str, seniority: str) -> IngestedCandidate:
    work = WorkEntry(
        company="Acme",
        title=f"{seniority.title()} Engineer",
        start_year=2020,
        end_year=2024,
        description="Built things.",
    )
    structured = StructuredResume(
        location="Berlin, Germany",
        total_yoe=5,
        current_title=f"{seniority.title()} Engineer",
        skills=["Python"],
        work_history=[work],
    )
    return IngestedCandidate(
        candidate_id=candidate_id,
        resume_text="...",
        structured=structured,
        summary=f"A {seniority} engineer.",
        inferred_seniority=seniority,
    )


def _make_normalized(seniority: list[str] | None = None) -> NormalizedQuery:
    return NormalizedQuery(
        raw="senior python dev",
        hard_filters=HardFilters(seniority=seniority),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="Senior Python developer.",
    )


def _make_scored(candidate_id: str) -> ScoredCandidate:
    return ScoredCandidate(
        candidate_id=candidate_id,
        score=4,
        must_have_matches=[MustHaveMatch(requirement="Python", met=True, evidence="Used Python.")],
        gaps=[],
        one_line_summary="Good match.",
    )


# --- node unit tests ---

def test_relax_filters_drops_seniority() -> None:
    state = {"normalized": _make_normalized(seniority=["senior"])}
    result = relax_filters_node(state)
    assert result["filters_relaxed"] is True
    assert result["normalized"].hard_filters.seniority is None


def test_relax_filters_keeps_location() -> None:
    normalized = NormalizedQuery(
        raw="senior python dev germany",
        hard_filters=HardFilters(seniority=["senior"], location_keywords=["Germany"]),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="Senior Python dev in Germany.",
    )
    result = relax_filters_node({"normalized": normalized})
    assert result["normalized"].hard_filters.location_keywords == ["Germany"]
    assert result["normalized"].hard_filters.seniority is None


def test_output_node_assembles_pipeline_output() -> None:
    candidates = [_make_ingested("c0001", "senior")]
    retrieved = [RetrievedCandidate(candidate=candidates[0], similarity=0.9)]
    ranked = [_make_scored("c0001")]
    normalized = _make_normalized()
    result = output_node({"normalized": normalized, "retrieved": retrieved, "ranked": ranked})
    output = result["output"]
    assert isinstance(output, PipelineOutput)
    assert output.normalized == normalized
    assert output.retrieved == retrieved
    assert output.ranked == ranked


# --- E2E smoke ---

def test_graph_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [_make_ingested(f"c{i:04d}", "senior") for i in range(3)]
    rng = np.random.default_rng(42)
    embeddings = rng.random((3, EMBED_DIM)).astype(np.float32)
    query_vec = rng.random(EMBED_DIM).astype(np.float32)
    normalized = _make_normalized()

    async def _fake_rerank(norm, retrieved, top_k):
        return [_make_scored(r.candidate.candidate_id) for r in retrieved]

    monkeypatch.setattr("hiring_agents.graph.load_node", lambda s: {"candidates": candidates, "embeddings": embeddings})
    monkeypatch.setattr("hiring_agents.graph.normalize_query", lambda raw: normalized)
    monkeypatch.setattr("hiring_agents.graph.embed_query", lambda text: query_vec)
    monkeypatch.setattr("hiring_agents.graph.rerank", _fake_rerank)

    state = build_graph().invoke({"raw_query": "senior python dev"})
    assert isinstance(state["output"], PipelineOutput)
    assert len(state["output"].ranked) == 3
    assert not state.get("filters_relaxed")


# --- fallback test ---

def test_graph_fallback_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    # All 3 candidates are junior; first query requires senior → 0 retrieved.
    # relax_filters drops seniority → all 3 pass on second retrieve.
    candidates = [_make_ingested(f"c{i:04d}", "junior") for i in range(3)]
    rng = np.random.default_rng(0)
    embeddings = rng.random((3, EMBED_DIM)).astype(np.float32)
    query_vec = rng.random(EMBED_DIM).astype(np.float32)
    normalized = _make_normalized(seniority=["senior"])

    async def _fake_rerank(norm, retrieved, top_k):
        return [_make_scored(r.candidate.candidate_id) for r in retrieved]

    monkeypatch.setattr("hiring_agents.graph.load_node", lambda s: {"candidates": candidates, "embeddings": embeddings})
    monkeypatch.setattr("hiring_agents.graph.normalize_query", lambda raw: normalized)
    monkeypatch.setattr("hiring_agents.graph.embed_query", lambda text: query_vec)
    monkeypatch.setattr("hiring_agents.graph.rerank", _fake_rerank)

    state = build_graph().invoke({"raw_query": "senior python dev"})
    assert state["filters_relaxed"] is True
    assert len(state["retrieved"]) == 3
    assert isinstance(state["output"], PipelineOutput)
