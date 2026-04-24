from __future__ import annotations

from pathlib import Path

import numpy as np

from hiring_agents.io_utils import (
    dump_models,
    file_hash,
    load_embeddings,
    load_models,
    read_json,
    save_embeddings,
    write_json,
)
from hiring_agents.schemas import Candidate


def test_write_and_read_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"
    payload = {"a": 1, "b": [1, 2, 3]}
    write_json(path, payload)
    assert read_json(path) == payload


def test_dump_and_load_models_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    items = [
        Candidate(candidate_id="c0001", resume_text="hello"),
        Candidate(candidate_id="c0002", resume_text="world"),
    ]
    dump_models(path, items)
    loaded = load_models(path, Candidate)
    assert loaded == items


def test_embeddings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "emb.npy"
    arr = np.random.default_rng(42).standard_normal((4, 8)).astype(np.float32)
    save_embeddings(path, arr)
    assert np.array_equal(load_embeddings(path), arr)


def test_file_hash_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"hello world")
    assert file_hash(path) == file_hash(path)
