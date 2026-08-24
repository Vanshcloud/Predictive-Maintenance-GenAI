# ============================================================
# Makefile — Developer Command Shortcuts
# ============================================================
# WHY: Makefiles standardize commands across the team.
# Instead of remembering "python -m pytest tests/ -v --tb=short",
# you just type "make test". Used at Google, Netflix, Stripe.
#
# USAGE: Run `make help` to see all available commands.
# ============================================================

.PHONY: help install install-dev test lint format clean run-api run-dashboard setup

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

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-integration: ## Run integration tests (slow; needs the generated dataset)
	python -m pytest tests/integration -v --tb=short -m integration

test-all: ## Run unit + integration tests
	python -m pytest tests/ -v --tb=short -m ""

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --tb=short --cov=src --cov=config --cov-report=term-missing

lint: ## Run flake8 linter
	flake8 src/ config/ tests/

format: ## Format code with Black + isort
	black src/ config/ tests/
	isort src/ config/ tests/

format-check: ## Check formatting without changing files
	black --check src/ config/ tests/
	isort --check-only src/ config/ tests/

typecheck: ## Run mypy type checker
	mypy src/ config/

quality: lint format-check typecheck ## Run all code quality checks

# ---- Application ----

run-api: ## Start FastAPI server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard: ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

# ---- Utilities ----

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
