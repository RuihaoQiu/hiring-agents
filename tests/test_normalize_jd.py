from __future__ import annotations

from unittest.mock import MagicMock, patch

from hiring_agents.normalize import normalize_jd
from hiring_agents.schemas import HardFilters, NormalizedQuery

_JD_TEXT = """
Software Engineer — Senior Python Backend

Location: Berlin, Germany
Requirements:
- 5+ years Python experience
- FastAPI or Django
- PostgreSQL
Nice to have:
- Kubernetes experience
- Prior fintech domain
"""

_NORMALIZED = NormalizedQuery(
    raw=_JD_TEXT[:200],
    hard_filters=HardFilters(location_keywords=["Berlin"], seniority=["senior"]),
    core_skills=["Python", "FastAPI", "PostgreSQL"],
    nice_to_haves=["Kubernetes", "fintech"],
    canonical_summary="Senior Python backend engineer in Berlin with 5+ years experience.",
)


def _mock_body() -> MagicMock:
    body = MagicMock()
    body.hard_filters = _NORMALIZED.hard_filters
    body.core_skills = _NORMALIZED.core_skills
    body.nice_to_haves = _NORMALIZED.nice_to_haves
    body.canonical_summary = _NORMALIZED.canonical_summary
    body.model_dump_json.return_value = "{}"
    return body


def test_normalize_jd_returns_normalized_query() -> None:
    mock_resp = MagicMock()
    mock_resp.choices[0].message.parsed = _mock_body()

    with patch("hiring_agents.normalize.get_sync_client") as mock_client, \
         patch("hiring_agents.normalize.observe_generation") as mock_obs:
        mock_obs.return_value.__enter__ = lambda s: MagicMock()
        mock_obs.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.return_value.beta.chat.completions.parse.return_value = mock_resp

        result = normalize_jd(_JD_TEXT)

    assert isinstance(result, NormalizedQuery)
    assert result.raw == _JD_TEXT[:200]
    assert result.core_skills == ["Python", "FastAPI", "PostgreSQL"]
    assert result.hard_filters.location_keywords == ["Berlin"]


