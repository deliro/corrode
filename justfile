default: lint test

install:
    uv sync

lint: lint-ruff typecheck

lint-ruff:
    uv run ruff check .

typecheck: lint-mypy lint-basedpyright lint-ty lint-pyrefly

lint-mypy:
    uv run mypy
    uv run mypy tests/type_checking/typesafety.py

lint-basedpyright:
    uv run --with basedpyright basedpyright

lint-ty:
    uv run --with ty ty check src/corrode/ tests/type_checking/typesafety.py

lint-pyrefly:
    uv run --with pyrefly pyrefly check src/corrode/ tests/type_checking/typesafety.py

test:
    uv run pytest

test-cov:
    uv run pytest --cov=corrode --cov-report=term-missing

build:
    uv build
