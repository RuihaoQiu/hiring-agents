# hiring-agents

A recruiting pipeline PoC: `normalize → retrieve → rerank`, with a Chainlit conversational UI backed by a LangGraph ReAct agent.

- [docs/vision.md](docs/vision.md) — decisions, eval results, lessons learned

## Setup

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY (and optionally LANGFUSE_* for tracing)
```

## Run

```bash
make ingest            # embed and index candidates (run once after generate)
make ui                # start Chainlit UI on :8501
```

Open [http://localhost:8501](http://localhost:8501). The agent handles free-text queries, JD paste, and shortlist management conversationally.

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
  agent/        LangGraph ReAct agent (tools, prompts)
  llm/          OpenAI client, prompts, embeddings, tracing
  ui/           Chainlit app
  normalize.py  Query / JD normalization
  retrieve.py   Embedding-based retrieval
  rerank.py     LLM-based reranking
data/           Candidates, embeddings, labels (git-ignored)
public/         Chainlit UI assets
```
