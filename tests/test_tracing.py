from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from hiring_agents.schemas import HardFilters


# --- unit tests for observe_generation ---

def test_observe_noop_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hiring_agents.llm.tracing._get_client", lambda: None)
    from hiring_agents.llm.tracing import observe_generation

    with observe_generation(name="test", model="gpt-4o-mini", input="q") as gen:
        gen.update(output="r")
    # must not raise


def test_observe_wraps_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_gen = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_gen)
    mock_cm.__exit__ = MagicMock(return_value=False)

    mock_lf = MagicMock()
    mock_lf.start_as_current_observation.return_value = mock_cm

    monkeypatch.setattr("hiring_agents.llm.tracing._get_client", lambda: mock_lf)
    from hiring_agents.llm.tracing import observe_generation

    with observe_generation(name="normalize", model="gpt-4o-mini", input=["msg"]) as gen:
        gen.update(output="result")

    mock_lf.start_as_current_observation.assert_called_once()
    call_kwargs = mock_lf.start_as_current_observation.call_args.kwargs
    assert call_kwargs["name"] == "normalize"
    assert call_kwargs["as_type"] == "generation"
    assert call_kwargs["model"] == "gpt-4o-mini"
    mock_gen.update.assert_called_once_with(output="result")
    mock_lf.flush.assert_called_once()


# --- integration: normalize path uses observe_generation ---

def test_normalize_path_calls_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    from hiring_agents.pipeline.normalize import _NormalizedBody
    from hiring_agents.llm.client import get_sync_client

    body = _NormalizedBody(
        hard_filters=HardFilters(),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="Python dev.",
    )
    mock_resp = MagicMock()
    mock_resp.choices[0].message.parsed = body

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_resp

    get_sync_client.cache_clear()
    monkeypatch.setattr("hiring_agents.pipeline.normalize.get_sync_client", lambda: mock_client)

    calls: list[dict] = []

    @contextmanager
    def fake_observe(**kw):
        calls.append(kw)
        yield MagicMock()

    monkeypatch.setattr("hiring_agents.pipeline.normalize.observe_generation", fake_observe)

    from hiring_agents.pipeline.normalize import normalize_query

    normalize_query("senior python dev")

    assert len(calls) == 1
    assert calls[0]["name"] == "normalize"


# --- integration: rerank path uses observe_generation ---

async def test_rerank_path_calls_observe(monkeypatch: pytest.MonkeyPatch) -> None:
    from hiring_agents.pipeline.rerank import _ScoredBody, _call
    from hiring_agents.schemas import (
        IngestedCandidate,
        MustHaveMatch,
        NormalizedQuery,
        RetrievedCandidate,
        StructuredResume,
        WorkEntry,
    )
    from hiring_agents.llm.client import get_async_client

    scored_body = _ScoredBody(
        score=4,
        must_have_matches=[MustHaveMatch(requirement="Python", met=True, evidence="Used Python")],
        gaps=[],
        one_line_summary="Good match.",
    )
    mock_resp = MagicMock()
    mock_resp.choices[0].message.parsed = scored_body

    mock_client = MagicMock()

    async def fake_parse(**kw):
        return mock_resp

    mock_client.beta.chat.completions.parse = fake_parse

    get_async_client.cache_clear()
    monkeypatch.setattr("hiring_agents.pipeline.rerank.get_async_client", lambda: mock_client)

    calls: list[dict] = []

    @contextmanager
    def fake_observe(**kw):
        calls.append(kw)
        yield MagicMock()

    monkeypatch.setattr("hiring_agents.pipeline.rerank.observe_generation", fake_observe)

    work = WorkEntry(company="Acme", title="Senior Eng", start_year=2020, end_year=2024, description="...")
    structured = StructuredResume(
        location="Berlin", total_yoe=5, current_title="Senior Eng", skills=["Python"], work_history=[work]
    )
    ingested = IngestedCandidate(
        candidate_id="c0001", resume_text="...", structured=structured, summary="...", inferred_seniority="senior"
    )
    normalized = NormalizedQuery(
        raw="python dev",
        hard_filters=HardFilters(),
        core_skills=["Python"],
        nice_to_haves=[],
        canonical_summary="Python developer.",
    )
    rc = RetrievedCandidate(candidate=ingested, similarity=0.9)

    await _call(normalized, rc)

    assert len(calls) == 1
    assert calls[0]["name"] == "rerank"
    assert calls[0]["metadata"]["candidate_id"] == "c0001"
