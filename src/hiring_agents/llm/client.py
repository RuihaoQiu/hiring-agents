from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI


@lru_cache(maxsize=1)
def get_sync_client() -> OpenAI:
    load_dotenv()
    _require_key()
    return OpenAI()


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    load_dotenv()
    _require_key()
    return AsyncOpenAI()


def _require_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; copy .env.example to .env and fill it in.")
