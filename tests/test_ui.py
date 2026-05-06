from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest


def test_call_api_returns_parsed_json() -> None:
    from hiring_agents.ui.app import _call_api

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"ranked": [], "retrieved_count": 0, "filters_relaxed": False}
    mock_resp.raise_for_status = MagicMock()

    with patch("hiring_agents.ui.app.httpx.post", return_value=mock_resp) as mock_post:
        result = _call_api("senior python dev")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/search" in call_kwargs.args[0]
    assert call_kwargs.kwargs["json"] == {"query": "senior python dev"}
    assert result["ranked"] == []


def test_call_api_raises_on_http_error() -> None:
    from hiring_agents.ui.app import _call_api

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch("hiring_agents.ui.app.httpx.post", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError):
            _call_api("python dev")
