from __future__ import annotations

from hiring_agents.data_gen.axes import (
    DOMAINS,
    LOCATIONS,
    ROLE_FAMILIES,
    SENIORITY_YOE_RANGE,
    TECH_STACKS,
)
from hiring_agents.data_gen.sampler import sample_specs


def test_sampler_is_deterministic_under_seed() -> None:
    a = sample_specs(count=50, seed=42)
    b = sample_specs(count=50, seed=42)
    assert a == b


def test_sampler_seed_variation_changes_output() -> None:
    a = sample_specs(count=50, seed=42)
    b = sample_specs(count=50, seed=43)
    assert a != b


def test_sampler_produces_unique_ids() -> None:
    specs = sample_specs(count=100, seed=42)
    ids = [s.candidate_id for s in specs]
    assert len(ids) == len(set(ids))


def test_sampler_respects_axis_values() -> None:
    specs = sample_specs(count=200, seed=42)
    for s in specs:
        assert s.role_family in ROLE_FAMILIES
        assert s.seniority in SENIORITY_YOE_RANGE
        assert s.location in LOCATIONS
        assert s.domain in DOMAINS
        assert s.tech_stack in TECH_STACKS[s.role_family]
        lo, hi = SENIORITY_YOE_RANGE[s.seniority]
        assert lo <= s.yoe <= hi


def test_sampler_covers_most_role_families_at_scale() -> None:
    specs = sample_specs(count=500, seed=42)
    observed = {s.role_family for s in specs}
    assert observed == set(ROLE_FAMILIES)
