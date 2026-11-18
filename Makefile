.PHONY: check test build

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run ty check src
	uv run pyright

test:
	uv run pytest tests -q

build:
	uv build
