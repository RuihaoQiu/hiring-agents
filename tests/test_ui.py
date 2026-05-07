from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


async def test_call_api_returns_parsed_json() -> None:
    from hiring_agents.ui.app import _call_api

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"ranked": [], "retrieved_count": 0, "filters_relaxed": False}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("hiring_agents.ui.app.httpx.AsyncClient", return_value=mock_client):
        result = await _call_api("senior python dev", "keyword")

    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "/search" in call_kwargs.args[0]
    assert call_kwargs.kwargs["json"] == {"query": "senior python dev", "mode": "keyword"}
    assert result["ranked"] == []


async def test_call_api_sends_hard_filters() -> None:
    from hiring_agents.ui.app import _call_api

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"ranked": [], "retrieved_count": 0, "filters_relaxed": False}
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_resp)

    filters = {"seniority": ["senior"], "location_keywords": ["Berlin"]}
    with patch("hiring_agents.ui.app.httpx.AsyncClient", return_value=mock_client):
        await _call_api("Data Engineer", "strict", hard_filters=filters)

    sent = mock_client.post.call_args.kwargs["json"]
    assert sent["mode"] == "strict"
    assert sent["hard_filters"] == filters


async def test_call_api_raises_on_http_error() -> None:
    from hiring_agents.ui.app import _call_api

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("hiring_agents.ui.app.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.HTTPStatusError):
            await _call_api("python dev", "keyword")
