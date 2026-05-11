from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hiring_agents.config import (
    RERANK_CONCURRENCY,
    RERANK_MAX_RETRIES,
    RERANK_MODEL,
    RERANK_TEMPERATURE,
    SCORE_MAX,
    SCORE_MIN,
    SEARCH_POOL_SIZE,
)
from hiring_agents.llm.client import get_async_client
from hiring_agents.llm.prompts import RERANK_SYSTEM, RERANK_USER
from hiring_agents.llm.tracing import observe_generation
from hiring_agents.schemas import (
    MustHaveMatch,
    NormalizedQuery,
    RetrievedCandidate,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)


class _ScoredBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: int = Field(ge=SCORE_MIN, le=SCORE_MAX)
    must_have_matches: list[MustHaveMatch]
    gaps: list[str]
    one_line_summary: str


async def rerank(
    normalized: NormalizedQuery,
    retrieved: list[RetrievedCandidate],
    top_k: int = SEARCH_POOL_SIZE,
    concurrency: int = RERANK_CONCURRENCY,
) -> list[ScoredCandidate]:
    if not retrieved:
        return []
    sem = asyncio.Semaphore(concurrency)

    async def bounded(rc: RetrievedCandidate) -> ScoredCandidate:
        async with sem:
            return await _score_one(normalized, rc)

    scored = await asyncio.gather(*(bounded(rc) for rc in retrieved))
    scored.sort(key=_sort_key)
    return scored[:top_k]


def _sort_key(s: ScoredCandidate) -> tuple[int, int]:
    met_count = sum(1 for m in s.must_have_matches if m.met)
    return (-s.score, -met_count)


async def _score_one(
    normalized: NormalizedQuery, rc: RetrievedCandidate
) -> ScoredCandidate:
    attempts = RERANK_MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            body = await _call(normalized, rc)
        except ValidationError:
            logger.warning(
                "rerank validation failed for %s (%d/%d)",
                rc.candidate.candidate_id,
                attempt,
                attempts,
            )
            if attempt >= attempts:
                raise
            continue
        return ScoredCandidate(
            candidate_id=rc.candidate.candidate_id,
            score=body.score,
            must_have_matches=body.must_have_matches,
            gaps=body.gaps,
            one_line_summary=body.one_line_summary,
        )
    raise RuntimeError("unreachable")


async def _call(normalized: NormalizedQuery, rc: RetrievedCandidate) -> _ScoredBody:
    client = get_async_client()
    candidate_payload = {
        "structured": rc.candidate.structured.model_dump(),
        "summary": rc.candidate.summary,
    }
    user = RERANK_USER.format(
        normalized_query_json=normalized.model_dump_json(),
        candidate_payload_json=json.dumps(candidate_payload),
    )
    messages = [
        {"role": "system", "content": RERANK_SYSTEM},
        {"role": "user", "content": user},
    ]
    with observe_generation(
        name="rerank",
        model=RERANK_MODEL,
        input=messages,
        metadata={"candidate_id": rc.candidate.candidate_id},
    ) as gen:
        resp = await client.beta.chat.completions.parse(
            model=RERANK_MODEL,
            temperature=RERANK_TEMPERATURE,
            messages=messages,
            response_format=_ScoredBody,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("rerank returned no parsed content")
        gen.update(output=parsed.model_dump_json())
    return parsed
