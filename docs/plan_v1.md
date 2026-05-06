# Hiring Agents — v1 Plan

A production-shaped service layer on top of the v0 PoC. The goal of v1 is to
expose the pipeline as a REST API, add a minimal UI for human use, migrate
orchestration to LangGraph for controlled extensibility, and instrument every
LLM call with Langfuse — while keeping the same 500 synthetic candidates and
the eval harness fully intact throughout.

## What v1 is and isn't

**v1 is:**
- The v0 pipeline (`normalize → retrieve → rerank`) running inside a LangGraph
  state machine instead of hand-rolled linear functions
- A FastAPI service exposing `POST /search` and `GET /health`, runnable locally
  with `uv run`
- A Streamlit UI (single page: query box, ranked results, per-candidate
  reasoning panel)
- Langfuse tracing on every LLM call via manual generation spans
- `seniority` as a hard-filter field in `NormalizedQuery` and `HardFilters`,
  applied before cosine retrieval alongside location and YOE
- One conditional edge in the graph: if hard filters return 0 candidates,
  relax filters (drop seniority + YOE bounds, keep location) and retry retrieve
- Eval harness unchanged and continuously green

**v1 is not:**
- Real candidate data (still the same 500 synthetic candidates)
- pgvector or any database (still JSON + numpy on disk)
- Auth, multi-tenancy, PII handling
- Production deployment, containers, or CI/CD
- Complex branching, human-in-the-loop, or agent memory
- Hybrid BM25 + vector retrieval (deferred to v2)

Target: one person, ~5 focused days. Incremental LangGraph adoption — build
the linear graph first, add the one conditional edge second, resist the urge
to design all branching upfront.

## Non-negotiable principles (inherited from v0, additions marked)

1. **Ground truth is physically isolated.** Unchanged.
2. **Every tunable is a named constant** in `config.py`. New constants for
   API port, Langfuse config, UI base URL, seniority vocab.
3. **Type hints everywhere.** Pydantic v2 at every module boundary, including
   FastAPI request/response models.
4. **Logging, not print.** Extended to FastAPI via standard `logging`; no
   `print()` anywhere, including Streamlit.
5. **Seed = 42** for every stochastic step.
6. **Tests for pure logic.** LLM calls mocked. Graph nodes tested with fake
   state dicts. FastAPI routes tested with `httpx.AsyncClient`. Langfuse
   calls mocked.
7. **Function <= 30 lines, file <= 300 lines, single purpose.**
8. **No dead code, no speculative abstractions.**
9. **(new) The graph is the source of truth for orchestration.** No business
   logic outside LangGraph nodes. `pipeline.py` is a thin shim that calls
   `graph.invoke()`.
10. **(new) Langfuse is opt-in.** If `LANGFUSE_SECRET_KEY` is absent the
    pipeline runs without tracing — no import errors, no startup failures.

## Stack changes (delta from v0)

| Concern | v0 | v1 |
|---|---|---|
| Orchestration | hand-rolled `pipeline.py` | LangGraph (`langgraph`) |
| Tracing | none | Langfuse (`langfuse`) via manual spans |
| API | typer CLI only | FastAPI + uvicorn |
| UI | none | Streamlit |
| Deps | openai, pydantic, numpy, typer, tenacity | + langgraph, langchain-core, langfuse, fastapi, uvicorn, streamlit, httpx |

**Why Streamlit over React:** no Node toolchain, no build step, pure Python,
can be developed and tested without a running API server. The UI is a
diagnostic tool for v1, not a product shell.

**Why LangGraph:** the zero-results fallback is the exact kind of conditional
control flow LangGraph is designed for. In hand-rolled code it would require
threading a retry flag through `run_pipeline`. In LangGraph it is a named
conditional edge with a named fallback node — auditable and independently
testable.

**Risk — LangGraph version churn:** pin `>=0.2,<0.3` in `pyproject.toml`.
The API moved significantly between 0.1 and 0.2.

**Risk — Langfuse callback vs bare OpenAI SDK:** the LangGraph callback
handler only auto-traces LangChain-wrapped calls. Our `normalize.py` and
`rerank.py` use the bare OpenAI SDK. Mitigation: use manual
`langfuse.generation()` spans at each call site instead of relying on the
callback handler.

## Schema changes

`HardFilters` drops `min_yoe`/`max_yoe` and replaces them with:
```python
seniority: list[str] | None = None  # e.g. ["senior", "staff", "principal"]
```

YOE was a leaky proxy for seniority — a 3-YOE candidate can write a resume
that embeds close to a senior query. Seniority inferred directly from
`current_title + work_history` is a cleaner signal.

`NormalizedQuery` gains:
```python
seniority: str | None = None  # canonical level extracted from query
```

`IngestedCandidate` gains:
```python
inferred_seniority: str | None = None  # inferred at ingest from title + history
```

Seniority vocab (`"junior"`, `"mid"`, `"senior"`, `"staff"`, `"principal"`)
lives as `SENIORITY_VOCAB` in `config.py` — same set used in `data_gen/axes.py`.

`retrieve.py`: remove `_passes_yoe`; add `_passes_seniority` that matches
`candidate.inferred_seniority` against `filters.seniority` list.

All new fields are nullable — backward-compatible with existing `labels.json`,
`queries.json`, and report files.

## Directory layout (delta from v0, new files only)

```
hiring-agents/
├── pyproject.toml              # + langgraph, langchain-core, langfuse,
│                               #   fastapi, uvicorn, streamlit, httpx
├── src/hiring_agents/
│   ├── graph_state.py          # PipelineState TypedDict
│   ├── graph.py                # build_graph() -> CompiledGraph
│   ├── llm/
│   │   └── tracing.py          # trace_generation() — no-ops without creds
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py              # create_app() FastAPI factory
│   │   ├── routes.py           # POST /search, GET /health
│   │   └── models.py           # SearchRequest, SearchResponse
│   └── ui/
│       ├── __init__.py
│       └── app.py              # Streamlit single-page app
├── tests/
│   ├── test_graph.py           # node unit tests + smoke E2E + fallback edge
│   ├── test_api.py             # httpx.AsyncClient route tests
│   └── test_seniority_filter.py
└── Makefile                    # + api, ui targets
```

`pipeline.py` is **not deleted** — it becomes a two-line shim so the eval
harness imports unchanged:

```python
def run_pipeline(raw_query: str) -> PipelineOutput:
    from hiring_agents.graph import build_graph
    return build_graph().invoke({"raw_query": raw_query})
```

## LangGraph design

### State

```python
# graph_state.py
from typing import TypedDict
import numpy as np
from hiring_agents.schemas import (
    NormalizedQuery, IngestedCandidate,
    RetrievedCandidate, ScoredCandidate, PipelineOutput,
)

class PipelineState(TypedDict, total=False):
    raw_query: str
    candidates: list[IngestedCandidate]
    embeddings: np.ndarray
    normalized: NormalizedQuery
    retrieved: list[RetrievedCandidate]
    ranked: list[ScoredCandidate]
    filters_relaxed: bool
    output: PipelineOutput
```

### Nodes

| Node | Responsibility | v0 equivalent |
|------|---------------|---------------|
| `load` | load ingested candidates + embeddings | inline in `run_pipeline` |
| `normalize` | `normalize_query(raw_query)` | `normalize.py` |
| `retrieve` | hard filter + cosine top-K | `retrieve.py` |
| `relax_filters` | drop seniority + YOE bounds, set `filters_relaxed=True` | new |
| `rerank` | async LLM scorer | `rerank.py` |
| `output` | assemble `PipelineOutput` | inline in `run_pipeline` |

### Edges

```
load → normalize → retrieve → conditional → rerank → output
                                   ↓ (zero results, not yet relaxed)
                             relax_filters → retrieve
```

Conditional logic: if `len(state["retrieved"]) == 0` and not
`state.get("filters_relaxed")`, route to `relax_filters`; otherwise forward
to `rerank`. The `filters_relaxed` flag prevents an infinite loop.

### Async note

`graph.invoke()` (sync) is used by the CLI and eval harness — the `rerank`
node calls `asyncio.run(rerank(...))` as today. `graph.ainvoke()` (async) is
used by FastAPI — the `rerank` node must be declared `async def` and call
`await rerank(...)` directly. This asymmetry is an accepted tradeoff for v1;
document it in `graph.py`.

## Build order

### Step 1 — Schema + seniority filter

1. Add `seniority` fields to `HardFilters`, `NormalizedQuery`,
   `IngestedCandidate`. Add `SENIORITY_VOCAB` to `config.py`.
2. Update `ingest.py`: sync `client.chat.completions.create` call to infer
   seniority from `current_title + work_history`. Returns one token from
   `SENIORITY_VOCAB`. Cached via existing content-hash.
3. Update `normalize.py`: extract `seniority` from query; emit only when
   query explicitly signals it (don't filter seniority-agnostic queries).
4. Update `retrieve.py`: remove `_passes_yoe`; add `_passes_seniority`
   predicate in `apply_hard_filters`.
5. Delete `data/ingested.json` + `data/embeddings.npy` — re-ingest once
   (~$0.50, one-time).
6. Tests: `test_seniority_filter.py` (all filter branches), updated
   `test_schemas.py`, updated `conftest.py`.

**Milestone:** `make query Q="senior python backend germany"` shows seniority
in normalized output. `make eval` passes within ±0.05 of v0 baseline.

### Step 2 — LangGraph graph

1. Add `langgraph`, `langchain-core` to `pyproject.toml`; `uv sync`.
2. Write `graph_state.py`.
3. Write `graph.py`: define all nodes, wire linear edges, add conditional
   edge + `relax_filters` node.
4. Update `pipeline.py` to delegate to `build_graph().invoke(...)`.
5. Tests: `test_graph.py` — node unit tests, smoke E2E (3 candidates, stubbed
   LLM), conditional fallback test (assert `filters_relaxed=True` and
   non-empty `retrieved` after fallback).

**Milestone:** `make query` and `make eval` work identically to v0. Fallback
fires correctly in test.

### Step 3 — FastAPI service

1. Add `fastapi`, `uvicorn[standard]` to `pyproject.toml`; `uv sync`.
2. Write `api/models.py`: `SearchRequest`, `SearchResponse` (includes
   `filters_relaxed: bool`).
3. Write `api/app.py`: `create_app()`. Load candidates + embeddings at
   startup into module-level singletons. Return 503 if cache absent.
4. Write `api/routes.py`: `POST /search` (calls `await graph.ainvoke(...)`),
   `GET /health`.
5. Add `api` Makefile target.
6. Tests: `test_api.py` — happy path, empty query (422), health (200). All
   graph nodes monkeypatched.

**Milestone:** `curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"senior python backend germany"}'` returns ranked JSON.

### Step 4 — Langfuse tracing

1. Add `langfuse` to `pyproject.toml`; `uv sync`.
2. Write `llm/tracing.py`: `trace_generation(name, model, input, output,
   metadata) -> None`. No-ops when `LANGFUSE_SECRET_KEY` is absent.
3. Wrap LLM calls in `normalize.py`, `rerank.py`, and seniority inference in
   `ingest.py`.
4. Update `.env.example` with `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY`.
5. Tests: assert `trace_generation` is called on normalize + rerank paths;
   assert no error when key absent.

**Milestone:** query with Langfuse creds produces a visible trace in the
dashboard. Without creds, no error.

### Step 5 — Streamlit UI

1. Add `streamlit` to `pyproject.toml`; `uv sync`.
2. Write `ui/app.py`:
   - Query input + Search button.
   - Calls `POST /search` via `httpx` (talks to running FastAPI, not importing
     pipeline directly).
   - One `st.expander` per candidate: score, `one_line_summary`,
     `must_have_matches`, `gaps`.
   - Sidebar: normalized query fields (`hard_filters`, `core_skills`,
     `seniority`, `canonical_summary`).
   - Banner when `filters_relaxed=True`: "No results with original filters —
     filters relaxed automatically."
3. Add `ui` Makefile target.
4. Tests: test `_call_api(query, top_k) -> SearchResponse` helper in
   isolation, mocked. Don't test Streamlit rendering.

**Milestone:** UI at `localhost:8501` shows results, sidebar, and relaxation
banner when fallback fires.

### Step 6 — Eval verification + polish

1. Run `make eval` — confirm metrics within ±0.05 of v0 baseline. Document
   any deliberate change in `vision.md`.
2. Update `README.md`: add "Running the API" and "Running the UI" sections.
3. Buffer for whatever overran in earlier steps.

## `config.py` additions

```python
SENIORITY_VOCAB: list[str] = ["junior", "mid", "senior", "staff", "principal"]
SENIORITY_INFER_MODEL: str = EXTRACTION_MODEL

API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
UI_API_BASE_URL: str = "http://localhost:8000"
```

Langfuse credentials are read via `os.getenv` at runtime, not from this
module.

## Risks summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| LangGraph `invoke` vs `ainvoke` async mismatch | High | Async rerank node for FastAPI; sync for CLI/eval |
| Langfuse doesn't auto-trace bare OpenAI SDK calls | Certain | Manual `trace_generation()` spans |
| Seniority filter hurts recall on seniority-agnostic queries | Medium | Normalize only emits seniority filter when query explicitly signals it |
| Re-ingest cost for seniority field | Low | < $0.50, one-time, cached |
| LangGraph API churn | Medium | Pin `>=0.2,<0.3` |
| Streamlit state complexity | Low | Keep UI stateless; all state in API response |

## Definition of done for v1

1. `make query Q="any query"` returns a coherent ranked top-10 (via LangGraph).
2. `POST /search` returns a valid `SearchResponse` JSON.
3. Streamlit UI shows results, normalized query sidebar, and relaxation banner
   when the zero-result fallback fires.
4. A Langfuse trace is visible for every query run with credentials set.
   Running without credentials produces no error.
5. `make eval` produces metrics within ±0.05 of v0 baseline. Any deliberate
   change is documented in `vision.md`.
6. `uv run pytest` green.
7. Someone new can clone, `uv sync`, `cp .env.example .env`, add their key,
   and reproduce eval numbers and run the API locally.
