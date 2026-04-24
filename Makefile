.PHONY: sync format lint test generate query eval all-checks

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

eval:
	uv run python -m scripts.run_eval

query:
	@test -n "$(Q)" || (echo "usage: make query Q='your query here'" && exit 1)
	uv run hiring-agents query "$(Q)"
