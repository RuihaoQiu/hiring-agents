from __future__ import annotations

from hiring_agents.graph import build_graph
from hiring_agents.schemas import PipelineOutput


def run_pipeline(raw_query: str) -> PipelineOutput:
    state = build_graph().invoke({"raw_query": raw_query})
    return state["output"]
