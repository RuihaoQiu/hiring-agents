# Hiring Agents — v0 Plan

A proper-quality v0 of a recruiting retrieval + rerank pipeline over synthetic
candidates. The goal of v0 is to prove the *shape* of the pipeline works
end-to-end with a working eval loop, before investing in real data, a database,
or a UI.

## What v0 is and isn't

**v0 is** a typed, tested Python package that:
- generates 500 synthetic candidates along explicit variation axes
- accepts a free-text query, runs `normalize → retrieve → rerank`
- returns ranked results with per-candidate reasoning
- ships an eval harness with 5 queries, programmatic positive labels derived
  from a separated ground-truth file, and retrieval/rerank metrics

**v0 is not** a database, a web service, a real ingest pipeline, a UI, a
privacy layer, or a multi-tenant anything. Those are Phase 1.

Target: one person, 4–5 days of focused work. API cost for full build + many
eval iterations: < $15.

## Non-negotiable principles

1. **Ground truth is physically isolated.** `Candidate` objects carry no
   ground-truth fields. Ground truth lives in a separate file keyed by
   `candidate_id`, loaded only by the data generator and the label generator.
   The pipeline cannot see it.
2. **Every tunable is a named constant** in `config.py`. No magic numbers or
   strings in business logic.
3. **Type hints everywhere.** Pydantic v2 at every module boundary.
4. **Logging, not print.** One `configure_logging()` call from CLI entry points.
5. **Seed = 42** for every stochastic step (axis sampling, messy-case
   injection, any shuffling).
6. **Tests for pure logic.** Metrics math, hard-filter branches, label
   criteria, cosine, schema validation, cache hit/miss. LLM calls mocked.
7. **Function ≤ 30 lines, file ≤ 300 lines, single purpose.** Split if
   you need "and" in the name.
8. **No dead code, no speculative abstractions.** Three similar lines beats a
   premature helper.

## Stack

- Python **3.12**
- Package manager: **uv** (`uv sync`, `uv run`)
- Linter/formatter: **ruff**, line length 100
- Tests: **pytest** (+ `pytest-asyncio`)
- CLI: **typer**
- Models:
  - Extraction, summary, normalization, rerank, data generation: `gpt-4o-mini`
  - Embedding: `text-embedding-3-small` (1536 dim)
- SDK: **openai** only (single dependency, simpler ops)
- In-memory vectors via numpy; no pgvector, no sklearn

## Directory layout

```
hiring-agents/
├── pyproject.toml              # deps, ruff, pytest config
├── uv.lock
├── Makefile                    # format, lint, test, generate, query, eval
├── README.md
├── .env.example                # OPENAI_API_KEY=
├── .gitignore                  # .env, data/cache, reports/
├── src/hiring_agents/
│   ├── __init__.py
│   ├── config.py               # all constants, paths, model IDs
│   ├── schemas.py              # Pydantic v2 models
│   ├── logging_setup.py
│   ├── io_utils.py             # JSON + numpy load/save, content-hash cache
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # single OpenAI client factory
│   │   ├── embeddings.py       # embed_query / embed_documents
│   │   └── prompts.py          # prompt templates as module constants
│   ├── data_gen/
│   │   ├── __init__.py
│   │   ├── axes.py             # ROLE_FAMILIES, SENIORITY, LOCATIONS, TECH_STACKS, DOMAINS
│   │   ├── sampler.py          # seeded combinatorial + messy-case sampler
│   │   └── generate.py         # resume writer (temp high for variety)
│   ├── ingest.py               # resume → structured + summary + embedding
│   ├── normalize.py            # raw query → NormalizedQuery
│   ├── retrieve.py             # hard filter + cosine, pure numpy
│   ├── rerank.py               # async scorer, concurrency-capped
│   ├── pipeline.py             # compose the four stages
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── labels.py           # programmatic positives from ground truth
│   │   ├── metrics.py          # recall@k, precision@k, ndcg@k
│   │   └── harness.py          # run all queries, emit reports
│   └── cli.py                  # typer app: generate | query | eval
├── data/
│   ├── candidates.json         # fixture: resume_text + id only
│   ├── ground_truth.json       # {candidate_id: GroundTruth} — pipeline never loads this
│   ├── ingested.json           # cache: structured + summary per candidate
│   ├── embeddings.npy          # cache: (500, 1536) float32
│   ├── queries.json
│   └── labels.json             # {query_id: [candidate_id, ...]}
├── reports/                    # per-run eval dumps (gitignored)
├── tests/
│   ├── conftest.py             # tiny fixture set, stubbed LLM
│   ├── test_schemas.py
│   ├── test_retrieve.py        # cosine, hard filter branches
│   ├── test_labels.py
│   ├── test_metrics.py
│   ├── test_ingest_cache.py
│   └── test_pipeline_smoke.py  # end-to-end with 3 candidates, stubbed LLM
└── scripts/
    ├── generate_data.py        # one-shot candidates + ground truth
    └── run_eval.py             # one-shot full eval
```

## `config.py` surface (sketch)

```python
from pathlib import Path

RANDOM_SEED = 42

# Data
CANDIDATE_COUNT = 500
MESSY_CASE_RATIO = 0.15           # ~75 messy out of 500

# Models
GENERATION_MODEL = "gpt-4o-mini"
EXTRACTION_MODEL = "gpt-4o-mini"
SUMMARY_MODEL = "gpt-4o-mini"
NORMALIZE_MODEL = "gpt-4o-mini"
RERANK_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Temperatures
GENERATION_TEMPERATURE = 0.9      # variety for synthetic resumes
EXTRACTION_TEMPERATURE = 0.0
SUMMARY_TEMPERATURE = 0.0
NORMALIZE_TEMPERATURE = 0.0
RERANK_TEMPERATURE = 0.0

# Summary shape
SUMMARY_WORD_MIN = 200
SUMMARY_WORD_MAX = 300

# Pipeline
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 10
RERANK_CONCURRENCY = 5
RERANK_MAX_RETRIES = 1

# Scoring scale
SCORE_MIN = 1
SCORE_MAX = 5

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
INGESTED_PATH = DATA_DIR / "ingested.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
QUERIES_PATH = DATA_DIR / "queries.json"
LABELS_PATH = DATA_DIR / "labels.json"
```

## Schemas (Pydantic v2)

- `Candidate` — `candidate_id: str`, `resume_text: str`
- `IngestedCandidate(Candidate)` — adds `structured: StructuredResume`,
  `summary: str` (embeddings are held separately as a numpy array aligned by
  index)
- `StructuredResume` — `location`, `total_yoe`, `current_title`, `skills`,
  `work_history: list[WorkEntry]`
- `GroundTruth` — `role_family`, `seniority`, `location`, `total_yoe`,
  `tech_stack`, `domains` (lives in its own file, never flows through pipeline)
- `HardFilters` — `location_keywords: list[str] | None`, `min_yoe: int | None`,
  `max_yoe: int | None`
- `NormalizedQuery` — `hard_filters: HardFilters`, `core_skills: list[str]`,
  `nice_to_haves: list[str]`, `canonical_summary: str`
- `RetrievedCandidate` — `candidate: IngestedCandidate`, `similarity: float`
- `MustHaveMatch` — `requirement: str`, `met: bool`, `evidence: str`
- `ScoredCandidate` — `candidate_id: str`, `score: int`,
  `must_have_matches: list[MustHaveMatch]`, `gaps: list[str]`,
  `one_line_summary: str`
- `PipelineOutput` — `normalized: NormalizedQuery`,
  `retrieved: list[RetrievedCandidate]`, `ranked: list[ScoredCandidate]`

## Build order (with tests interleaved)

### Day 1 — Foundation + data generation

1. Scaffold: `pyproject.toml`, `uv sync`, ruff config, pytest config,
   `Makefile`, `.env.example`, `logging_setup.py`, `config.py`, `schemas.py`,
   `io_utils.py`.
2. `llm/client.py`, `llm/embeddings.py`.
3. `data_gen/axes.py` — enumerate role families, seniorities, locations, stacks
   per family, domains.
4. `data_gen/sampler.py` — seeded sampler that draws `CANDIDATE_COUNT` combos
   with stratified coverage, then injects messy cases (role switchers, gaps,
   IC→manager, niche stacks, ambiguous location, inflated titles, clear
   obvious-fits for planned eval queries).
5. `data_gen/generate.py` — for each combo, LLM writes a resume with varied
   register and quality. Returns `(Candidate, GroundTruth)` pairs.
6. `scripts/generate_data.py` — runs once, writes `candidates.json` + (separate)
   `ground_truth.json`.
7. Tests: `test_schemas.py`, sampler determinism test (seed=42 → same combos).

**Milestone:** 500 candidates on disk, manually inspected for variety. Ground
truth in a separate file.

### Day 2 — Ingest + retrieve

1. `ingest.py`: `ingest_all(candidates) -> list[IngestedCandidate]` + aligned
   `np.ndarray` of embeddings. Content-hash cache — recompute only when
   `candidates.json` changes.
2. `retrieve.py`: `apply_hard_filters`, `cosine_similarity`, `retrieve_top_k`.
   Pure numpy, no sklearn.
3. `normalize.py`: `normalize_query(raw: str) -> NormalizedQuery`.
4. Tests: `test_retrieve.py` (cosine math, filter branches, empty-pool),
   `test_ingest_cache.py` (hit/miss), manual eyeball of normalize on 5 inputs.

**Milestone:** can retrieve top-20 for any query; cached ingest loads in
seconds.

### Day 3 — Rerank + pipeline + CLI

1. `rerank.py`: async scorer using OpenAI structured outputs (JSON schema mode),
   `asyncio.Semaphore(RERANK_CONCURRENCY)`, one retry on schema-validation
   failure, sort by `(-score, -sum(met))`.
2. `pipeline.py`: `run_pipeline(raw_query) -> PipelineOutput`.
3. `cli.py`: typer app with three subcommands:
   - `hiring-agents generate` — runs data gen
   - `hiring-agents query "..."` — prints normalized query + top 10 + reasoning
   - `hiring-agents eval` — runs the eval harness
4. Tests: `test_pipeline_smoke.py` with 3 tiny candidates and a stubbed LLM
   protocol.

**Milestone:** `uv run hiring-agents query "python dev 5 years germany"` prints
a coherent ranked list.

### Day 4 — Eval

1. `data/queries.json` — 5 queries: 2 short phrases, 2 fake JDs, 1 edge case
   (contradictory / niche combo).
2. `eval/labels.py` — positive criteria per query as Python callables over
   `GroundTruth`; produces `labels.json`. Hand-refine after generation.
3. `eval/metrics.py` — `recall_at_k`, `precision_at_k`, `ndcg_at_k` (binary
   relevance).
4. `eval/harness.py` — runs pipeline for each query, computes metrics, dumps
   per-query JSON to `reports/<timestamp>/<query_id>.json` plus a
   `summary.json` with the metrics table.
5. `scripts/run_eval.py`.
6. Tests: `test_labels.py`, `test_metrics.py` (edge cases: empty positives,
   all-positive, ties).

**Milestone:** baseline metrics table + per-query detailed reports.

### Day 5 — Iterate + polish

- Read detailed reports, pick the biggest failure mode, make one targeted
  change (prompt tweak, hard-filter refinement, summary prompt adjustment),
  rerun eval, record before/after.
- README with setup, commands, and the before/after story.
- Buffer for whatever bit the previous days over-ran on.

## Eval details

- **Retrieval metric:** recall@20 (meaningful now — 20/500, not 20/50).
- **Rerank metrics:** precision@5, NDCG@10 (binary relevance).
- **Positive generation:** programmatic criteria against separated
  `GroundTruth`, then eyeball-refined. The ground-truth file is the only input
  to label generation — confirms separation.
- **Per-query report JSON:** normalized query, retrieved list with sim,
  ranked list with score + reasoning + must-have matches, positives, computed
  metrics.
- **Stdout summary:** via logger, a small metrics table.

## Explicit non-goals for v0

- Database / persistence beyond JSON + npy on disk
- Chunk-level embeddings (single summary embedding only)
- Hybrid BM25 + vector (pure vector is fine at 500)
- PII stripping / bias mitigation (synthetic data, no real people)
- Web UI
- Multi-tenant / auth / anything user-facing beyond CLI
- Re-generation determinism of LLM outputs beyond what `temperature=0` gives

## Open hooks for Phase 1 (intentionally not built)

- Swap `text-embedding-3-small` → `-large` without code change (one constant).
- Swap any stage's model via `config.py` — all calls go through `llm/client.py`.
- Swap JSON fixtures → DB-backed loader behind `io_utils.py` interface.
- Replace in-memory cosine with pgvector by implementing the same
  `retrieve_top_k` signature.

## Definition of done for v0

At the end of Day 5 / early Day 6:

1. `uv run hiring-agents query "any query"` returns a coherent ranked top-10
   with reasoning.
2. `uv run hiring-agents eval` produces a metrics table and detailed per-query
   JSON reports under `reports/<timestamp>/`.
3. Tests pass: `uv run pytest` green.
4. README documents: before-metrics → one change → after-metrics → why.
5. Someone new can clone, `uv sync`, `cp .env.example .env`, add their key,
   and reproduce the eval numbers.

If all five are true, v0 has succeeded and we have a mandate to move to Phase 1
on real data.
