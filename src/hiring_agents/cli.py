from __future__ import annotations

import typer

from hiring_agents.logging_setup import configure_logging
from hiring_agents.pipeline import run_pipeline
from hiring_agents.schemas import PipelineOutput

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command(name="generate")
def generate_command() -> None:
    """Generate synthetic candidates and ground truth."""
    from scripts.generate_data import main

    main()


@app.command(name="query")
def query_command(
    raw: str = typer.Argument(..., help="Free-text recruiter query"),
) -> None:
    """Run normalize -> retrieve -> rerank on a query and print results."""
    configure_logging()
    output = run_pipeline(raw)
    _print_output(output)


@app.command(name="eval")
def eval_command() -> None:
    """Run the eval harness across all queries."""
    from scripts.run_eval import main

    main()


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
