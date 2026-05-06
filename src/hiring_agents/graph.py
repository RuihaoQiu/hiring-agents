from __future__ import annotations

import asyncio
import logging

from langgraph.graph import END, StateGraph

from hiring_agents.config import CANDIDATES_PATH, RERANK_TOP_K, RETRIEVAL_TOP_K
from hiring_agents.graph_state import PipelineState
from hiring_agents.ingest import ingest_all
from hiring_agents.io_utils import load_models
from hiring_agents.llm.embeddings import embed_query
from hiring_agents.normalize import normalize_query
from hiring_agents.rerank import rerank
from hiring_agents.retrieve import apply_hard_filters, retrieve_top_k
from hiring_agents.schemas import Candidate, HardFilters, PipelineOutput, RetrievedCandidate

logger = logging.getLogger(__name__)


def load_node(state: PipelineState) -> dict:
    if state.get("candidates") is not None:
        return {}  # pre-loaded by caller (e.g. API startup)
    candidates = load_models(CANDIDATES_PATH, Candidate)
    ingested, embeddings = ingest_all(candidates)
    return {"candidates": ingested, "embeddings": embeddings}


def normalize_node(state: PipelineState) -> dict:
    return {"normalized": normalize_query(state["raw_query"])}


def retrieve_node(state: PipelineState) -> dict:
    normalized = state["normalized"]
    ingested = state["candidates"]
    embeddings = state["embeddings"]
    allowed = apply_hard_filters(ingested, normalized.hard_filters)
    logger.info("hard filters: %d/%d passed", len(allowed), len(ingested))
    qvec = embed_query(normalized.canonical_summary)
    top = retrieve_top_k(qvec, embeddings, allowed, k=RETRIEVAL_TOP_K)
    retrieved = [RetrievedCandidate(candidate=ingested[idx], similarity=sim) for idx, sim in top]
    return {"retrieved": retrieved}


def relax_filters_node(state: PipelineState) -> dict:
    normalized = state["normalized"]
    relaxed = HardFilters(location_keywords=normalized.hard_filters.location_keywords)
    return {
        "normalized": normalized.model_copy(update={"hard_filters": relaxed}),
        "filters_relaxed": True,
    }


def rerank_node(state: PipelineState) -> dict:
    # graph.invoke() (sync, CLI/eval) — asyncio.run bridges the async rerank.
    # graph.ainvoke() (async, FastAPI in Step 3) requires an async version of this node.
    ranked = asyncio.run(rerank(state["normalized"], state["retrieved"], top_k=RERANK_TOP_K))
    return {"ranked": ranked}


def output_node(state: PipelineState) -> dict:
    return {
        "output": PipelineOutput(
            normalized=state["normalized"],
            retrieved=state["retrieved"],
            ranked=state["ranked"],
        )
    }


def _route_after_retrieve(state: PipelineState) -> str:
    if len(state["retrieved"]) == 0 and not state.get("filters_relaxed"):
        return "relax_filters"
    return "rerank"


def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("load", load_node)
    builder.add_node("normalize", normalize_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("relax_filters", relax_filters_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("output", output_node)

    builder.set_entry_point("load")
    builder.add_edge("load", "normalize")
    builder.add_edge("normalize", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"relax_filters": "relax_filters", "rerank": "rerank"},
    )
    builder.add_edge("relax_filters", "retrieve")
    builder.add_edge("rerank", "output")
    builder.add_edge("output", END)

    return builder.compile()
