from __future__ import annotations

import typer

from hiring_agents.logging_setup import configure_logging
from hiring_agents.schemas import PipelineOutput

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command(name="generate")
def generate_command() -> None:
    """Generate synthetic candidates and ground truth."""
    configure_logging()
    from hiring_agents.data_gen.generate import generate_pair
    from hiring_agents.data_gen.sampler import sample_specs
    from hiring_agents.config import (
        CANDIDATE_COUNT,
        CANDIDATES_PATH,
        GENERATION_LOG_EVERY,
        GROUND_TRUTH_PATH,
        RANDOM_SEED,
    )
    from hiring_agents.io_utils import write_json
    import logging

    logger = logging.getLogger(__name__)
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


@app.command(name="ingest")
def ingest_command(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingest even if cache is fresh"),
) -> None:
    """Ingest candidates: extract structured fields, write summaries, infer seniority, embed."""
    configure_logging()
    from hiring_agents.config import CANDIDATES_PATH, EMBEDDINGS_PATH, INGESTED_PATH
    from hiring_agents.ingest import ingest_all
    from hiring_agents.io_utils import load_models
    from hiring_agents.schemas import Candidate
    import logging

    if force:
        INGESTED_PATH.unlink(missing_ok=True)
        EMBEDDINGS_PATH.unlink(missing_ok=True)

    candidates = load_models(CANDIDATES_PATH, Candidate)
    ingested, embeddings = ingest_all(candidates)
    logging.getLogger(__name__).info(
        "ingest complete: %d candidates, embeddings %s", len(ingested), embeddings.shape
    )


@app.command(name="query")
def query_command(
    raw: str = typer.Argument(..., help="Free-text recruiter query"),
) -> None:
    """Run normalize -> retrieve -> rerank on a query and print results."""
    import asyncio

    from hiring_agents.config import CANDIDATES_PATH, RERANK_TOP_K, RETRIEVAL_TOP_K, SKIP_RERANK
    from hiring_agents.ingest import ingest_all
    from hiring_agents.io_utils import load_models
    from hiring_agents.llm.embeddings import embed_query
    from hiring_agents.normalize import normalize_query
    from hiring_agents.rerank import rerank
    from hiring_agents.retrieve import apply_hard_filters, retrieve_top_k
    from hiring_agents.schemas import Candidate, HardFilters, RetrievedCandidate, ScoredCandidate

    configure_logging()
    ingested, embeddings = ingest_all(load_models(CANDIDATES_PATH, Candidate))
    normalized = normalize_query(raw)
    allowed = apply_hard_filters(ingested, normalized.hard_filters)
    if not allowed:
        relaxed = HardFilters(location_keywords=normalized.hard_filters.location_keywords)
        normalized = normalized.model_copy(update={"hard_filters": relaxed})
        allowed = apply_hard_filters(ingested, normalized.hard_filters)
    qvec = embed_query(normalized.canonical_summary)
    top = retrieve_top_k(qvec, embeddings, allowed, k=RETRIEVAL_TOP_K)
    retrieved = [RetrievedCandidate(candidate=ingested[i], similarity=s) for i, s in top]
    if SKIP_RERANK:
        ranked: list[ScoredCandidate] = [
            ScoredCandidate(candidate_id=rc.candidate.candidate_id, score=3, must_have_matches=[], gaps=[], one_line_summary="")
            for rc in retrieved[:RERANK_TOP_K]
        ]
    else:
        ranked = asyncio.run(rerank(normalized, retrieved))
    _print_output(PipelineOutput(normalized=normalized, retrieved=retrieved, ranked=ranked))


@app.command(name="eval")
def eval_command(
    query_id: str | None = typer.Option(None, "--query", "-q", help="Run a single query by ID"),
) -> None:
    """Run the eval harness across all queries, or a single one with --query."""
    configure_logging()
    from hiring_agents.eval.harness import run_eval

    run_eval(query_id=query_id)


def _print_output(output: PipelineOutput) -> None:
    typer.echo("=== NORMALIZED QUERY ===")
    typer.echo(output.normalized.model_dump_json(indent=2))
    typer.echo("")
    typer.echo(f"=== RETRIEVED ({len(output.retrieved)}) ===")
    for rc in output.retrieved:
        typer.echo(
            f"  {rc.candidate.candidate_id}  sim={rc.similarity:.3f}  "
            f"{rc.candidate.structured.current_title} @ "
            f"{rc.candidate.structured.location}"
        )
    typer.echo("")
    typer.echo(f"=== RANKED (top {len(output.ranked)}) ===")
    for r in output.ranked:
        met = sum(1 for m in r.must_have_matches if m.met)
        typer.echo(
            f"  score={r.score} must_have={met}/{len(r.must_have_matches)}  "
            f"{r.candidate_id}: {r.one_line_summary}"
        )
