# ============================================================
# Makefile — Developer Command Shortcuts
# ============================================================
# WHY: Makefiles standardize commands across the team.
# Instead of remembering "python -m pytest tests/ -v --tb=short",
# you just type "make test". Used at Google, Netflix, Stripe.
#
# USAGE: Run `make help` to see all available commands.
# ============================================================

.PHONY: help install install-dev test test-integration test-all test-cov lint \
	format format-check typecheck quality clean run-api run-dashboard setup \
	docker-build docker-up docker-down smoke

# Every path the linters check, declared once. This used to be spelled out per
# target as `src/ config/ tests/` while CI checked `src/ config/ tests/ scripts/
# dashboard/` — so `make quality` passed on a laptop and the same commit went
# red in CI on a file the local run never opened. Same failure mode as the mypy
# split below: two copies of "what gets checked" that drifted.
PY_PATHS = src/ config/ tests/ scripts/ dashboard/

# Default target when you just type `make`
help: ## Show this help message
	@echo "=========================================="
	@echo "Predictive Maintenance + GenAI"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---- Setup ----

setup: ## Full project setup (venv + deps + dirs)
	/opt/homebrew/bin/python3.12 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements-dev.txt
	@echo ""
	@echo "✅ Setup complete! Activate venv with: source venv/bin/activate"

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt

# ---- Code Quality ----

test: ## Run unit tests (integration excluded; see test-all)
	python -m pytest tests/ -v --tb=short

test-integration: ## Run integration tests (slow; needs the generated dataset)
	python -m pytest tests/integration -v --tb=short -m integration

test-all: ## Run unit + integration tests
	python -m pytest tests/ -v --tb=short -m ""

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --tb=short --cov=src --cov=config --cov-report=term-missing

lint: ## Run flake8 linter
	flake8 $(PY_PATHS)

format: ## Format code with Black + isort
	black $(PY_PATHS)
	isort $(PY_PATHS)

format-check: ## Check formatting without changing files
	black --check $(PY_PATHS)
	isort --check-only $(PY_PATHS)

typecheck: ## Run mypy type checker (both local and CI conditions)
# Two runs, because they check different programs. The first sees the installed
# venv, so pandas and langchain have real types. The second is what CI does:
# only the linters are installed there, so `ignore_missing_imports` turns those
# libraries into `Any` and `warn_return_any` starts firing at every third-party
# seam. Four such errors once reached main because the local run was clean.
	mypy src/ config/
	mypy --no-site-packages src/ config/

quality: lint format-check typecheck ## Run all code quality checks

# ---- Application ----

run-api: ## Start FastAPI server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard: ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

# ---- Utilities ----

docker-build: ## Build both container images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Run API + dashboard in containers (needs a trained model)
	docker compose -f docker/docker-compose.yml up

docker-down: ## Stop the containers
	docker compose -f docker/docker-compose.yml down

clean: ## Remove auto-generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "✅ Cleaned all auto-generated files"

smoke: ## Run smoke test to verify setup
	python -c "import tensorflow; print(f'✅ TensorFlow {tensorflow.__version__}')"
	python -c "import langchain; print(f'✅ LangChain {langchain.__version__}')"
	python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')"
	python -c "import streamlit; print(f'✅ Streamlit {streamlit.__version__}')"
	python -c "from config.settings import get_settings; s = get_settings(); print(f'✅ Config loaded: {s.APP_NAME}')"
	@echo ""
	@echo "🎉 All smoke tests passed!"
