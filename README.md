# hiring-agents

A recruiting pipeline PoC: `normalize → retrieve → rerank`, with a Chainlit chat UI and a FastAPI backend.

- [docs/plan_v2.md](docs/plan_v2.md) — v2 architecture, Chainlit UI, streaming search
- [docs/vision.md](docs/vision.md) — decisions, eval results, lessons learned

## Setup

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY (and optionally LANGFUSE_* for tracing)
```

## Run

```bash
make ingest            # embed and index candidates (run once after generate)
make api               # start FastAPI backend on :8000
make ui                # start Chainlit UI on :8501
```

Open [http://localhost:8501](http://localhost:8501) and pick a search mode:

- **Keyword search** — free-text query, LLM normalizes skills/location/seniority
- **Job Description search** — paste a JD, extracts structured requirements
- **Strict search** — select title + location + seniority, no filter relaxation

Results stream in as each candidate is ranked. Add candidates to a shortlist and export as CSV.

## Other commands

```bash
make generate          # create 500 synthetic candidates (one-off)
make query Q="python dev 5y germany"   # run the pipeline from the CLI
make eval              # run the eval harness
make all-checks        # ruff + pytest
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | Used for embeddings, normalization, and reranking |
| `LANGFUSE_PUBLIC_KEY` | no | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | no | Langfuse tracing |
| `SKIP_RERANK` | no | Set to `1` to skip LLM rerank (fast UI testing) |

## Layout

```
src/hiring_agents/
  api/          FastAPI app and routes
  llm/          OpenAI client, prompts, embeddings, tracing
  ui/           Chainlit app
  normalize.py  Query / JD normalization
  retrieve.py   Embedding-based retrieval
  rerank.py     LLM-based reranking
  graph.py      LangGraph pipeline
scripts/
  generate_data.py   Synthetic candidate generation
  run_eval.py        Eval harness
data/                Candidates, embeddings, labels (git-ignored)
public/              Chainlit UI assets
```
