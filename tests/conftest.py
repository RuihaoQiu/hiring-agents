from __future__ import annotations

import pytest

from hiring_agents.schemas import (
    IngestedCandidate,
    StructuredResume,
    WorkEntry,
)


@pytest.fixture
def sample_work_entry() -> WorkEntry:
    return WorkEntry(
        company="Acme",
        title="Senior Backend Engineer",
        start_year=2020,
        end_year=2024,
        description="Built distributed payment pipelines in Python and Kafka.",
    )


@pytest.fixture
def sample_structured(sample_work_entry: WorkEntry) -> StructuredResume:
    return StructuredResume(
        location="Berlin, Germany",
        total_yoe=7,
        current_title="Senior Backend Engineer",
        skills=["Python", "Kafka", "Postgres"],
        work_history=[sample_work_entry],
    )


@pytest.fixture
def sample_ingested(sample_structured: StructuredResume) -> IngestedCandidate:
    return IngestedCandidate(
        candidate_id="c0001",
        resume_text="...",
        structured=sample_structured,
        summary="Senior backend engineer in Berlin with 7 years of experience in Python.",
    )
