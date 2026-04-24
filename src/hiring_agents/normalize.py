from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict
from tenacity import retry, stop_after_attempt, wait_exponential

from hiring_agents.config import (
    LLM_MAX_ATTEMPTS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_RETRY_WAIT_MIN_SECONDS,
    NORMALIZE_MODEL,
    NORMALIZE_TEMPERATURE,
)
from hiring_agents.llm.client import get_sync_client
from hiring_agents.llm.prompts import (
    QUERY_NORMALIZATION_SYSTEM,
    QUERY_NORMALIZATION_USER,
)
from hiring_agents.schemas import HardFilters, NormalizedQuery

logger = logging.getLogger(__name__)


class _NormalizedBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_filters: HardFilters
    core_skills: list[str]
    nice_to_haves: list[str]
    canonical_summary: str


@retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1, min=LLM_RETRY_WAIT_MIN_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS
    ),
)
def normalize_query(raw: str) -> NormalizedQuery:
    client = get_sync_client()
    user = QUERY_NORMALIZATION_USER.format(raw_query=raw)
    resp = client.beta.chat.completions.parse(
        model=NORMALIZE_MODEL,
        temperature=NORMALIZE_TEMPERATURE,
        messages=[
            {"role": "system", "content": QUERY_NORMALIZATION_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=_NormalizedBody,
    )
    body = resp.choices[0].message.parsed
    if body is None:
        raise RuntimeError("query normalization returned no parsed content")
    return NormalizedQuery(
        raw=raw,
        hard_filters=body.hard_filters,
        core_skills=body.core_skills,
        nice_to_haves=body.nice_to_haves,
        canonical_summary=body.canonical_summary,
    )
