.PHONY: help setup env models run test verify verify-ollama clean frontend-setup

help:
	@echo "setup          Create .venv and install backend dependencies"
	@echo "env            Copy .env.example to .env (if missing)"
	@echo "models         Pull the required Ollama models"
	@echo "run            Start the FastAPI dev server"
	@echo "test           Run the test suite"
	@echo "verify         Run the Phase 1 setup checker"
	@echo "verify-ollama  Check the local Ollama connection"
	@echo "clean          Remove caches and the virtualenv"

setup:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

env:
	@test -f .env || cp .env.example .env

models:
	ollama pull qwen3:4b
	ollama pull embeddinggemma

run:
	./.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

test:
	./.venv/bin/pytest

verify:
	./.venv/bin/python scripts/check_setup.py

verify-ollama:
	./.venv/bin/python scripts/check_ollama.py

clean:
	rm -rf .venv .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
