#!/bin/bash

# AI Image Generation Tool - Run Script

set -e

echo "🎨 AI Image Generation Tool - Starting..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -x "venv/bin/python" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

PYTHON_BIN="venv/bin/python"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
fi

# Parse command line arguments
case "${1:-start}" in
    start)
        echo "Starting server..."
        "$PYTHON_BIN" main.py
        ;;
    dev)
        echo "Starting server in development mode..."
        "$PYTHON_BIN" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ;;
    docker)
        echo "Starting with Docker..."
        docker compose up -d
        ;;
    stop)
        echo "Stopping Docker containers..."
        docker compose down
        ;;
    test)
        echo "Running tests..."
        "$PYTHON_BIN" -m unittest discover -s tests -v
        ;;
    *)
        echo "Usage: $0 {start|dev|docker|stop|test}"
        exit 1
        ;;
esac
