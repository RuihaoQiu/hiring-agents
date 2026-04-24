from __future__ import annotations

import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from hiring_agents.config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    LLM_MAX_ATTEMPTS,
    LLM_RETRY_WAIT_MAX_SECONDS,
    LLM_RETRY_WAIT_MIN_SECONDS,
)
from hiring_agents.data_gen.axes import SENIORITY_YOE_RANGE
from hiring_agents.data_gen.sampler import CandidateSpec
from hiring_agents.llm.client import get_sync_client
from hiring_agents.llm.prompts import RESUME_GENERATION_SYSTEM, RESUME_GENERATION_USER
from hiring_agents.schemas import Candidate, GroundTruth

logger = logging.getLogger(__name__)


def generate_pair(spec: CandidateSpec) -> tuple[Candidate, GroundTruth]:
    resume_text = _write_resume(spec)
    candidate = Candidate(candidate_id=spec.candidate_id, resume_text=resume_text)
    ground_truth = _to_ground_truth(spec)
    return candidate, ground_truth


@retry(
    stop=stop_after_attempt(LLM_MAX_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1, min=LLM_RETRY_WAIT_MIN_SECONDS, max=LLM_RETRY_WAIT_MAX_SECONDS
    ),
)
def _write_resume(spec: CandidateSpec) -> str:
    lo, hi = SENIORITY_YOE_RANGE[spec.seniority]
    user = RESUME_GENERATION_USER.format(
        role_family=spec.role_family,
        seniority=spec.seniority,
        yoe_range=f"{lo}-{hi}",
        location=spec.location,
        tech_stack=", ".join(spec.tech_stack),
        domain=spec.domain,
        register=spec.register,
        quality=spec.quality,
        trait=spec.trait,
    )
    client = get_sync_client()
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": RESUME_GENERATION_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError(f"empty generation for {spec.candidate_id}")
    return content.strip()


def _to_ground_truth(spec: CandidateSpec) -> GroundTruth:
    return GroundTruth(
        role_family=spec.role_family,
        seniority=spec.seniority,
        location=spec.location,
        total_yoe=spec.yoe,
        tech_stack=list(spec.tech_stack),
        domains=[spec.domain],
    )
