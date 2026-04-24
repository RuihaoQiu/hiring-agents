from __future__ import annotations

import logging

from hiring_agents.config import (
    CANDIDATE_COUNT,
    CANDIDATES_PATH,
    GENERATION_LOG_EVERY,
    GROUND_TRUTH_PATH,
    RANDOM_SEED,
)
from hiring_agents.data_gen.generate import generate_pair
from hiring_agents.data_gen.sampler import sample_specs
from hiring_agents.io_utils import write_json
from hiring_agents.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    logger.info("sampling %d candidate specs (seed=%d)", CANDIDATE_COUNT, RANDOM_SEED)
    specs = sample_specs(CANDIDATE_COUNT, seed=RANDOM_SEED)

    candidates: list[dict] = []
    ground_truth: dict[str, dict] = {}
    for i, spec in enumerate(specs, start=1):
        cand, gt = generate_pair(spec)
        candidates.append(cand.model_dump())
        ground_truth[spec.candidate_id] = gt.model_dump()
        if i % GENERATION_LOG_EVERY == 0:
            logger.info("generated %d/%d", i, len(specs))

    write_json(CANDIDATES_PATH, candidates)
    write_json(GROUND_TRUTH_PATH, ground_truth)
    logger.info("wrote %s and %s", CANDIDATES_PATH, GROUND_TRUTH_PATH)


if __name__ == "__main__":
    main()
