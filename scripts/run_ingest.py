from __future__ import annotations

import logging

from hiring_agents.config import CANDIDATES_PATH
from hiring_agents.ingest import ingest_all
from hiring_agents.io_utils import load_models
from hiring_agents.logging_setup import configure_logging
from hiring_agents.schemas import Candidate

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    candidates = load_models(CANDIDATES_PATH, Candidate)
    logger.info("loaded %d candidates from %s", len(candidates), CANDIDATES_PATH)
    ingested, embeddings = ingest_all(candidates)
    logger.info(
        "ingest done: %d candidates, embeddings shape %s",
        len(ingested),
        embeddings.shape,
    )


if __name__ == "__main__":
    main()
