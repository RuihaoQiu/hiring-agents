from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from hiring_agents.config import CANDIDATES_PATH
from hiring_agents.graph import build_graph
from hiring_agents.ingest import ingest_all
from hiring_agents.io_utils import load_models
from hiring_agents.schemas import Candidate

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        raw = load_models(CANDIDATES_PATH, Candidate)
        ingested, embeddings = ingest_all(raw)
        app.state.candidates = ingested
        app.state.embeddings = embeddings
        app.state.graph = build_graph()
        logger.info("startup: loaded %d candidates", len(ingested))
    except Exception:
        logger.warning("startup: failed to load candidate data — /search will return 503")
        app.state.candidates = None
        app.state.embeddings = None
        app.state.graph = build_graph()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Hiring Agents", lifespan=lifespan)
    from hiring_agents.api.routes import router

    app.include_router(router)
    return app
