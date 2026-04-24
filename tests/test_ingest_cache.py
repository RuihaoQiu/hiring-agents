from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hiring_agents import ingest
from hiring_agents.config import EMBEDDING_DIM
from hiring_agents.io_utils import write_json
from hiring_agents.schemas import (
    Candidate,
    IngestedCandidate,
    StructuredResume,
)


def _candidate(i: int) -> Candidate:
    return Candidate(candidate_id=f"c{i:04d}", resume_text=f"resume {i}")


def _ingested(cand: Candidate) -> IngestedCandidate:
    return IngestedCandidate(
        candidate_id=cand.candidate_id,
        resume_text=cand.resume_text,
        structured=StructuredResume(
            location="Berlin",
            total_yoe=5,
            current_title="Engineer",
            skills=["Python"],
            work_history=[],
        ),
        summary=f"summary {cand.candidate_id}",
    )


class _Counter:
    def __init__(self) -> None:
        self.process = 0
        self.embed = 0


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    source = tmp_path / "candidates.json"
    cands = [_candidate(i) for i in range(3)]
    write_json(source, [c.model_dump() for c in cands])
    return {
        "candidates": cands,
        "source": source,
        "ingested": tmp_path / "ingested.json",
        "embeddings": tmp_path / "embeddings.npy",
    }


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> _Counter:
    counter = _Counter()

    def fake_process(cand: Candidate, index: int, total: int) -> IngestedCandidate:
        counter.process += 1
        return _ingested(cand)

    def fake_embed(texts) -> np.ndarray:
        counter.embed += 1
        return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(ingest, "_process", fake_process)
    monkeypatch.setattr(ingest, "embed_documents", fake_embed)
    return counter


def _run(workspace: dict) -> tuple[list[IngestedCandidate], np.ndarray]:
    return ingest.ingest_all(
        workspace["candidates"],
        source_path=workspace["source"],
        ingested_path=workspace["ingested"],
        embeddings_path=workspace["embeddings"],
    )


def test_miss_when_no_cache(workspace: dict, fake_llm: _Counter) -> None:
    out, emb = _run(workspace)
    assert [c.candidate_id for c in out] == ["c0000", "c0001", "c0002"]
    assert emb.shape == (3, EMBEDDING_DIM)
    assert fake_llm.process == 3
    assert fake_llm.embed == 1
    assert workspace["ingested"].exists()
    assert workspace["embeddings"].exists()


def test_hit_when_source_unchanged(workspace: dict, fake_llm: _Counter) -> None:
    _run(workspace)
    fake_llm.process = 0
    fake_llm.embed = 0
    out, emb = _run(workspace)
    assert len(out) == 3
    assert emb.shape == (3, EMBEDDING_DIM)
    assert fake_llm.process == 0
    assert fake_llm.embed == 0


def test_miss_when_source_changed(workspace: dict, fake_llm: _Counter) -> None:
    _run(workspace)
    fake_llm.process = 0
    fake_llm.embed = 0
    workspace["source"].write_text(workspace["source"].read_text() + " ")
    _run(workspace)
    assert fake_llm.process == 3
    assert fake_llm.embed == 1


def test_miss_when_embeddings_missing(workspace: dict, fake_llm: _Counter) -> None:
    _run(workspace)
    fake_llm.process = 0
    fake_llm.embed = 0
    workspace["embeddings"].unlink()
    _run(workspace)
    assert fake_llm.process == 3
    assert fake_llm.embed == 1
