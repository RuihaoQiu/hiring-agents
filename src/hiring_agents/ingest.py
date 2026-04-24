from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from hiring_agents.config import (
    CANDIDATES_PATH,
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    EXTRACTION_MODEL,
    EXTRACTION_TEMPERATURE,
    INGEST_LOG_EVERY,
    INGESTED_PATH,
    LLM_MAX_ATTEMPTS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_RETRY_WAIT_MIN_SECONDS,
    SUMMARY_MODEL,
    SUMMARY_TEMPERATURE,
)
from hiring_agents.io_utils import (
    file_hash,
    load_embeddings,
    read_json,
    save_embeddings,
    write_json,
)
from hiring_agents.llm.client import get_sync_client
from hiring_agents.llm.embeddings import embed_documents
from hiring_agents.llm.prompts import (
    STRUCTURED_EXTRACTION_SYSTEM,
    SUMMARY_SYSTEM,
    SUMMARY_USER,
)
from hiring_agents.schemas import Candidate, IngestedCandidate, StructuredResume

logger = logging.getLogger(__name__)

_SOURCE_HASH_KEY = "source_hash"
_ITEMS_KEY = "items"

_RETRY = retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1, min=LLM_RETRY_WAIT_MIN_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS
    ),
)


def ingest_all(
    candidates: list[Candidate],
    source_path: Path = CANDIDATES_PATH,
    ingested_path: Path = INGESTED_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> tuple[list[IngestedCandidate], np.ndarray]:
    source_hash = file_hash(source_path)
    cached = _load_cache(source_hash, ingested_path, embeddings_path, len(candidates))
    if cached is not None:
        logger.info("ingest cache hit: %d candidates", len(cached[0]))
        return cached

    logger.info("ingest cache miss: processing %d candidates", len(candidates))
    ingested = [_process(c, i, len(candidates)) for i, c in enumerate(candidates, start=1)]
    embeddings = embed_documents([c.summary for c in ingested])
    _save_cache(source_hash, ingested, embeddings, ingested_path, embeddings_path)
    return ingested, embeddings


def _process(cand: Candidate, index: int, total: int) -> IngestedCandidate:
    structured = _extract_structured(cand.resume_text)
    summary = _write_summary(cand.resume_text, structured)
    if index % INGEST_LOG_EVERY == 0:
        logger.info("ingested %d/%d", index, total)
    return IngestedCandidate(
        candidate_id=cand.candidate_id,
        resume_text=cand.resume_text,
        structured=structured,
        summary=summary,
    )


@_RETRY
def _extract_structured(resume_text: str) -> StructuredResume:
    client = get_sync_client()
    resp = client.beta.chat.completions.parse(
        model=EXTRACTION_MODEL,
        temperature=EXTRACTION_TEMPERATURE,
        messages=[
            {"role": "system", "content": STRUCTURED_EXTRACTION_SYSTEM},
            {"role": "user", "content": resume_text},
        ],
        response_format=StructuredResume,
    )
    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("structured extraction returned no parsed content")
    return parsed


@_RETRY
def _write_summary(resume_text: str, structured: StructuredResume) -> str:
    client = get_sync_client()
    user = SUMMARY_USER.format(
        resume_text=resume_text,
        structured_json=structured.model_dump_json(),
    )
    resp = client.chat.completions.create(
        model=SUMMARY_MODEL,
        temperature=SUMMARY_TEMPERATURE,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("empty summary")
    return content.strip()


def _load_cache(
    source_hash: str,
    ingested_path: Path,
    embeddings_path: Path,
    expected_count: int,
) -> tuple[list[IngestedCandidate], np.ndarray] | None:
    if not ingested_path.exists() or not embeddings_path.exists():
        return None
    envelope = read_json(ingested_path)
    if envelope.get(_SOURCE_HASH_KEY) != source_hash:
        return None
    items = [IngestedCandidate(**raw) for raw in envelope[_ITEMS_KEY]]
    embeddings = load_embeddings(embeddings_path)
    if len(items) != expected_count or embeddings.shape != (expected_count, EMBEDDING_DIM):
        return None
    return items, embeddings


def _save_cache(
    source_hash: str,
    items: list[IngestedCandidate],
    embeddings: np.ndarray,
    ingested_path: Path,
    embeddings_path: Path,
) -> None:
    envelope = {
        _SOURCE_HASH_KEY: source_hash,
        _ITEMS_KEY: [item.model_dump() for item in items],
    }
    write_json(ingested_path, envelope)
    save_embeddings(embeddings_path, embeddings)
