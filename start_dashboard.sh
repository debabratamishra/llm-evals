#!/bin/bash

# LLM Evaluation Dashboard Quick Start Script

echo "🧠 Starting LLM Evaluation Dashboard..."
echo "======================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH"
    exit 1
fi

# Check if llm_ui environment exists
if ! conda env list | grep -q "llm_ui"; then
    echo "❌ Error: Conda environment 'llm_ui' not found"
    echo "Please create the environment first with: conda create -n llm_ui python=3.12"
    exit 1
fi

# Navigate to dashboard directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📂 Working directory: $SCRIPT_DIR"

# Activate environment and check dependencies
echo "🔧 Activating llm_ui environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_ui

# Install/update dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Check if evaluation data exists
EVAL_DIR="./data"
if [ ! -d "$EVAL_DIR" ]; then
    echo "⚠️  Warning: Evaluation directory not found at $EVAL_DIR"
    echo "You can specify a different data directory in the dashboard sidebar."
fi

# Start the dashboard
echo "🚀 Starting Streamlit dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo "🔄 Press Ctrl+C to stop the dashboard"
echo ""

streamlit run app.py --server.port 8501

echo ""
echo "👋 Dashboard stopped. Thanks for using the LLM Evaluation Dashboard!"