.PHONY: sync format lint test generate ingest query eval ui all-checks

sync:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest

all-checks: lint test

generate:
	uv run python -m scripts.generate_data

ingest:
	uv run hiring-agents ingest

eval:
	uv run python -m scripts.run_eval

query:
	@test -n "$(Q)" || (echo "usage: make query Q='your query here'" && exit 1)
	uv run hiring-agents query "$(Q)"

ui:
	uv run chainlit run src/hiring_agents/ui/app.py --port 8501
