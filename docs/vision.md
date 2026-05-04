# Hiring Agents — Vision Log

A running record of what we learned, what changed, and why.

---

## v0 — Prove the shape (completed 2026-05-04)

**Goal:** typed, tested Python PoC of `normalize → retrieve → rerank` over 500 synthetic
candidates, with a working eval loop. No database, no UI, no real data.

**Stack:** Python 3.12, uv, OpenAI SDK (`gpt-4o-mini` + `text-embedding-3-small`), numpy
for in-memory vectors, typer CLI, pytest.

### What we built

- 500 synthetic candidates generated along explicit axes (role family, seniority, location,
  tech stack, domain, register, quality, messy traits)
- `ingest`: resume → structured fields + summary + embedding, content-hash cached
- `normalize`: free-text query → `HardFilters` + `core_skills` + `canonical_summary`
- `retrieve`: hard filter (location, YOE) + cosine similarity top-K
- `rerank`: async LLM scorer with structured output, concurrency-capped
- eval harness: recall@20, precision@5, NDCG@10 over 5 labelled queries

### Baseline metrics

| query | description                        | #pos | rec@20 | prec@5 | ndcg@10 |
|-------|------------------------------------|------|--------|--------|---------|
| q1    | senior python backend engineer     |  11  |  0.545 |  1.000 |   0.727 |
| q2    | ml engineer pytorch                |  16  |  0.562 |  1.000 |   0.934 |
| q3    | senior frontend react/ts           |  33  |  0.545 |  1.000 |   1.000 |
| q4    | data scientist python sql          |  26  |  0.577 |  1.000 |   1.000 |
| q5    | devops kubernetes                  |  25  |  0.560 |  1.000 |   0.936 |
| avg   |                                    |      |  0.558 |  1.000 |   0.919 |

### Iteration: seniority alignment

**Biggest failure mode:** q1 recall@20=0.545, ndcg@10=0.727. Investigation revealed two
issues:

1. **Label bug** — q1 criterion (`role_family == "backend" and "Python" in tech_stack`) had
   no seniority check, so juniors and mids were counted as positives for "senior python
   backend engineer". Fixed by adding `seniority in ("senior", "staff")` to the criterion.

2. **Rerank prompt too weak on seniority** — mid-level candidates with full skill coverage
   were scoring 4–5 even when the query specified "senior". Fixed by adding an explicit cap:
   if the query specifies a senior level and the candidate is clearly more junior, score ≤ 2.

**After iteration:**

| query | #pos | rec@20 | prec@5 | ndcg@10 |
|-------|------|--------|--------|---------|
| q1    |   3  |  0.667 |  0.400 |   0.651 |
| q2    |  16  |  0.500 |  1.000 |   0.791 |
| q3    |  33  |  0.485 |  1.000 |   1.000 |
| q4    |  26  |  0.577 |  1.000 |   1.000 |
| q5    |  25  |  0.600 |  1.000 |   0.857 |
| avg   |      |  0.566 |  0.880 |   0.860 |

q1 ndcg improved on tighter labels. q2/q5 regressed slightly — the seniority cap in the
rerank prompt is too blunt for queries that don't specify seniority.

### What v0 taught us

- **prec@5 is strong** — the top 5 results are almost always correct when labels are clean.
- **recall@20 is the bottleneck** — relevant candidates exist in the pool but don't surface
  in top-20 retrieval. Root cause: a single embedding can't distinguish seniority signals,
  so hard filters + cosine alone aren't enough.
- **Label quality matters as much as pipeline quality** — the first eval run looked worse
  than it was because the label criterion was too broad.
- **Seniority is a first-class signal** — YOE as a proxy for seniority is leaky. A 3-YOE
  candidate can write a resume that embeds close to a senior query.

---

## v1 — Real data, real infra (planned)

**Goal:** move from synthetic PoC to a deployable service on real candidate data.

### Open questions to resolve before building

1. **Data source** — where does real candidate data come from? What schema?
2. **Seniority signal** — add `seniority` as a structured field to `NormalizedQuery` and
   use it as a hard filter or dedicated rerank dimension, rather than relying on YOE proxy.
3. **Tracing** — Langfuse for LLM call observability; integrates cleanly with LangGraph.
4. **Orchestration** — migrate pipeline to LangGraph for conditional branching,
   human-in-the-loop, and easier extensibility.
5. **Hybrid retrieval** — BM25 + vector to improve recall on keyword-heavy queries.
6. **Persistence** — pgvector (or similar) to replace JSON + numpy on disk.
7. **API / UI** — expose beyond CLI; FastAPI + minimal frontend or Streamlit.
