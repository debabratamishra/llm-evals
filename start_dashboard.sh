#!/bin/bash

# LLM Evaluation Dashboard Starter Framework Quick Start Script
echo "🧠 Starting LLM Evaluation Starter Framework..."
echo "================================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' is not installed or not in PATH"
    echo "This project requires uv for python environment management."
    echo "Please install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Node/NPM is available
if ! command -v npm &> /dev/null; then
    echo "❌ Error: Node.js/NPM is not installed"
    echo "Please install Node.js (v18+) to run the React frontend."
    exit 1
fi

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📂 Project root: $SCRIPT_DIR"

# Clean up any leftover processes on script exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    echo "👋 Dashboard stopped. Thank you!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start FastAPI backend
echo "⚡ Starting FastAPI backend..."
uv run python backend/main.py &
BACKEND_PID=$!

# Start React frontend
echo "⚛️  Starting Vite React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Give servers a moment to bind ports
sleep 2

echo ""
echo "🚀 Servers are up and running!"
echo "   ---------------------------------------"
echo "   ⚛️  Frontend Dashboard: http://localhost:3000"
echo "   ⚡ Backend API Docs:   http://localhost:8000/docs"
echo "   ---------------------------------------"
echo "🔄 Press Ctrl+C to stop both servers"
echo ""

# Keep script running and wait for background processes
wait $BACKEND_PID $FRONTEND_PID