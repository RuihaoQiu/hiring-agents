from __future__ import annotations

import random
from dataclasses import dataclass

from hiring_agents.config import MESSY_CASE_RATIO
from hiring_agents.data_gen.axes import (
    DOMAINS,
    LOCATIONS,
    MESSY_TRAITS,
    QUALITY_LEVELS,
    REGISTERS,
    ROLE_FAMILIES,
    SENIORITY_YOE_RANGE,
    TECH_STACKS,
    TRAIT_NONE,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    role_family: str
    seniority: str
    yoe: int
    location: str
    tech_stack: tuple[str, ...]
    domain: str
    register: str
    quality: str
    trait: str


def sample_specs(count: int, seed: int) -> list[CandidateSpec]:
    rng = random.Random(seed)
    specs = [_sample_one(rng, idx) for idx in range(count)]
    return specs


def _sample_one(rng: random.Random, idx: int) -> CandidateSpec:
    role_family = rng.choice(ROLE_FAMILIES)
    seniority = rng.choice(list(SENIORITY_YOE_RANGE.keys()))
    yoe_lo, yoe_hi = SENIORITY_YOE_RANGE[seniority]
    yoe = rng.randint(yoe_lo, yoe_hi)
    stack = rng.choice(TECH_STACKS[role_family])
    trait = rng.choice(MESSY_TRAITS) if rng.random() < MESSY_CASE_RATIO else TRAIT_NONE
    return CandidateSpec(
        candidate_id=f"c{idx:04d}",
        role_family=role_family,
        seniority=seniority,
        yoe=yoe,
        location=rng.choice(LOCATIONS),
        tech_stack=stack,
        domain=rng.choice(DOMAINS),
        register=rng.choice(REGISTERS),
        quality=rng.choice(QUALITY_LEVELS),
        trait=trait,
    )
