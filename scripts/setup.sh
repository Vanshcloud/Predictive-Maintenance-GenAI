#!/bin/bash
# ============================================================
# scripts/setup.sh — One-Command Project Setup
# ============================================================
# WHY: New team members should be able to set up the entire
# project with a SINGLE command. This script automates:
#   1. Virtual environment creation
#   2. Dependency installation
#   3. .env file creation
#   4. Directory structure verification
#
# USAGE: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ============================================================

set -e  # Exit immediately if any command fails

echo "=========================================="
echo "🔧 Predictive Maintenance + GenAI Setup"
echo "=========================================="
echo ""

# --- Step 1: Check Python version ---
echo "📌 Step 1: Checking Python version..."
# TensorFlow requires Python ≤3.12. We prefer python3.12 if available.
PYTHON_CMD=""
if command -v /opt/homebrew/bin/python3.12 &> /dev/null; then
    PYTHON_CMD="/opt/homebrew/bin/python3.12"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
else
    echo "❌ Python 3.10–3.12 required for TensorFlow compatibility."
    echo "   Install: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   ✅ Using $PYTHON_CMD (Python $PYTHON_VERSION)"

# --- Step 2: Create virtual environment ---
echo ""
echo "📌 Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  venv/ already exists, skipping creation"
else
    $PYTHON_CMD -m venv venv
    echo "   ✅ Virtual environment created"
fi

# --- Step 3: Activate and install dependencies ---
echo ""
echo "📌 Step 3: Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements-dev.txt --quiet
echo "   ✅ All dependencies installed"

# --- Step 4: Create .env if not exists ---
echo ""
echo "📌 Step 4: Setting up environment variables..."
if [ -f ".env" ]; then
    echo "   ⚠️  .env already exists, skipping"
else
    cp .env.example .env
    echo "   ✅ Created .env from .env.example"
    echo "   📝 Remember to update .env with your actual API keys!"
fi

# --- Step 5: Verify directory structure ---
echo ""
echo "📌 Step 5: Verifying directory structure..."
DIRS=("data/raw" "data/processed" "data/sample" "models" "logs" "notebooks" "docker" "dashboard")
for dir in "${DIRS[@]}"; do
    mkdir -p "$dir"
done
echo "   ✅ All directories present"

# --- Step 6: Run smoke tests ---
echo ""
echo "📌 Step 6: Running smoke tests..."
python -m pytest tests/unit/test_smoke.py -v --tb=short 2>&1 | tail -20

echo ""
echo "=========================================="
echo "🎉 Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv:  source venv/bin/activate"
echo "  2. Run tests:      make test"
echo "  3. See commands:   make help"
echo ""
