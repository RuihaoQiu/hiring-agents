from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from hiring_agents.config import (
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_MAX_ATTEMPTS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_RETRY_WAIT_MIN_SECONDS,
)
from hiring_agents.llm.client import get_sync_client

logger = logging.getLogger(__name__)

_RETRY = retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1, min=LLM_RETRY_WAIT_MIN_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS
    ),
)


@_RETRY
def embed_query(text: str) -> np.ndarray:
    client = get_sync_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(f"unexpected embedding shape {vec.shape}")
    return vec


def embed_documents(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[start : start + EMBED_BATCH_SIZE])
        logger.info("embedding batch %d..%d of %d", start, start + len(batch), len(texts))
        out[start : start + len(batch)] = _embed_batch(batch)
    return out


@_RETRY
def _embed_batch(batch: list[str]) -> np.ndarray:
    client = get_sync_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
