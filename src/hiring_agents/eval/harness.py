from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from hiring_agents.config import (
    CANDIDATES_PATH,
    QUERIES_PATH,
    RERANK_TOP_K,
    REPORTS_DIR,
    RETRIEVAL_TOP_K,
    SKIP_RERANK,
)
from hiring_agents.eval.labels import load_or_generate_labels
from hiring_agents.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from hiring_agents.ingest import ingest_all
from hiring_agents.io_utils import load_models, read_json
from hiring_agents.llm.embeddings import embed_query
from hiring_agents.normalize import normalize_query
from hiring_agents.rerank import rerank
from hiring_agents.retrieve import apply_hard_filters, retrieve_top_k
from hiring_agents.schemas import (
    Candidate,
    HardFilters,
    IngestedCandidate,
    PipelineOutput,
    RetrievedCandidate,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_RECALL_K = RETRIEVAL_TOP_K
_PRECISION_K = 5
_NDCG_K = 10


def _run_query(raw: str, ingested: list[IngestedCandidate], embeddings) -> PipelineOutput:
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
    return PipelineOutput(normalized=normalized, retrieved=retrieved, ranked=ranked)


def _compute_metrics(
    retrieved_ids: list[str], ranked_ids: list[str], positives: set[str]
) -> dict[str, float]:
    return {
        f"recall_at_{_RECALL_K}": recall_at_k(retrieved_ids, positives, _RECALL_K),
        f"precision_at_{_PRECISION_K}": precision_at_k(ranked_ids, positives, _PRECISION_K),
        f"ndcg_at_{_NDCG_K}": ndcg_at_k(ranked_ids, positives, _NDCG_K),
    }


def run_eval(query_id: str | None = None) -> None:
    queries = read_json(QUERIES_PATH)
    if query_id is not None:
        queries = [q for q in queries if q["query_id"] == query_id]
        if not queries:
            raise ValueError(f"query_id {query_id!r} not found in {QUERIES_PATH}")
    labels = load_or_generate_labels()
    ingested, embeddings = ingest_all(load_models(CANDIDATES_PATH, Candidate))

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_DIR / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for query in queries:
        qid: str = query["query_id"]
        raw: str = query["raw"]
        positives = set(labels.get(qid, []))
        logger.info("running query %s (%d positives)", qid, len(positives))

        output = _run_query(raw, ingested, embeddings)
        retrieved_ids = [rc.candidate.candidate_id for rc in output.retrieved]
        ranked_ids = [sc.candidate_id for sc in output.ranked]
        metrics = _compute_metrics(retrieved_ids, ranked_ids, positives)

        per_query = {
            "query_id": qid,
            "raw": raw,
            "normalized": output.normalized.model_dump(),
            "retrieved": [
                {"candidate_id": rc.candidate.candidate_id, "similarity": rc.similarity}
                for rc in output.retrieved
            ],
            "ranked": [sc.model_dump() for sc in output.ranked],
            "positives": sorted(positives),
            "metrics": metrics,
        }
        _write_json(report_dir / f"{qid}.json", per_query)

        row = {"query_id": qid, "raw": raw[:60], "num_positives": len(positives), **metrics}
        rows.append(row)
        logger.info(
            "%s  recall@%d=%.3f  prec@%d=%.3f  ndcg@%d=%.3f",
            qid,
            _RECALL_K,
            metrics[f"recall_at_{_RECALL_K}"],
            _PRECISION_K,
            metrics[f"precision_at_{_PRECISION_K}"],
            _NDCG_K,
            metrics[f"ndcg_at_{_NDCG_K}"],
        )

    averages = {
        key: sum(r[key] for r in rows) / len(rows)
        for key in rows[0]
        if isinstance(rows[0][key], float)
    }
    summary = {"timestamp": timestamp, "queries": rows, "averages": averages}
    _write_json(report_dir / "summary.json", summary)
    logger.info("reports written to %s", report_dir)
    _print_table(rows, averages)


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_table(rows: list[dict], averages: dict[str, float]) -> None:
    header = f"{'query':6}  {'#pos':>4}  {'rec@20':>6}  {'prec@5':>6}  {'ndcg@10':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in rows:
        logger.info(
            "%-6s  %4d  %6.3f  %6.3f  %7.3f",
            r["query_id"],
            r["num_positives"],
            r[f"recall_at_{_RECALL_K}"],
            r[f"precision_at_{_PRECISION_K}"],
            r[f"ndcg_at_{_NDCG_K}"],
        )
    logger.info("-" * len(header))
    logger.info(
        "%-6s  %4s  %6.3f  %6.3f  %7.3f",
        "avg",
        "",
        averages[f"recall_at_{_RECALL_K}"],
        averages[f"precision_at_{_PRECISION_K}"],
        averages[f"ndcg_at_{_NDCG_K}"],
    )
