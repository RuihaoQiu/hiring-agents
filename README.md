# hiring-agents

Synthetic-data PoC for a recruiting pipeline: `normalize → retrieve → rerank`,
with an eval harness over 5 queries.

- [docs/plan.md](docs/plan.md) — architecture, schema, config surface, directory layout
- [docs/vision.md](docs/vision.md) — decisions, eval results, lessons learned, v1 roadmap

## Setup

```bash
uv sync
cp .env.example .env          # then fill in OPENAI_API_KEY
```

## Commands

```bash
make generate                        # one-shot: create 500 synthetic candidates
make query Q="python dev 5y germany" # run the pipeline for one query
make eval                            # run the eval harness
make all-checks                      # ruff + pytest
```

## Layout

See [docs/plan.md](docs/plan.md#directory-layout).
