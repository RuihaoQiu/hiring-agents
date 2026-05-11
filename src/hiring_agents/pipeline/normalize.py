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
    JD_NORMALIZATION_SYSTEM,
    JD_NORMALIZATION_USER,
    QUERY_NORMALIZATION_SYSTEM,
    QUERY_NORMALIZATION_USER,
)
from hiring_agents.llm.tracing import observe_generation
from hiring_agents.schemas import HardFilters, NormalizedQuery

logger = logging.getLogger(__name__)

_RETRY = retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1, min=LLM_RETRY_WAIT_MIN_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS
    ),
)


class _NormalizedBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hard_filters: HardFilters
    core_skills: list[str]
    nice_to_haves: list[str]
    canonical_summary: str


@_RETRY
def _call_normalize(name: str, messages: list[dict], raw: str) -> NormalizedQuery:
    client = get_sync_client()
    with observe_generation(name=name, model=NORMALIZE_MODEL, input=messages) as gen:
        resp = client.beta.chat.completions.parse(
            model=NORMALIZE_MODEL,
            temperature=NORMALIZE_TEMPERATURE,
            messages=messages,
            response_format=_NormalizedBody,
        )
        body = resp.choices[0].message.parsed
        if body is None:
            raise RuntimeError(f"{name} returned no parsed content")
        gen.update(output=body.model_dump_json())
    return NormalizedQuery(
        raw=raw,
        hard_filters=body.hard_filters,
        core_skills=body.core_skills,
        nice_to_haves=body.nice_to_haves,
        canonical_summary=body.canonical_summary,
    )


def normalize_query(raw: str) -> NormalizedQuery:
    messages = [
        {"role": "system", "content": QUERY_NORMALIZATION_SYSTEM},
        {"role": "user", "content": QUERY_NORMALIZATION_USER.format(raw_query=raw)},
    ]
    return _call_normalize("normalize", messages, raw)


def normalize_jd(text: str) -> NormalizedQuery:
    messages = [
        {"role": "system", "content": JD_NORMALIZATION_SYSTEM},
        {"role": "user", "content": JD_NORMALIZATION_USER.format(jd_text=text)},
    ]
    return _call_normalize("normalize_jd", messages, text[:200])
