from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, TypeAdapter

_JSON_INDENT = 2


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=_JSON_INDENT, ensure_ascii=False)


def load_models[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    adapter = TypeAdapter(list[model])
    return adapter.validate_python(read_json(path))


def dump_models(path: Path, items: list[BaseModel]) -> None:
    write_json(path, [m.model_dump() for m in items])


def save_embeddings(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
