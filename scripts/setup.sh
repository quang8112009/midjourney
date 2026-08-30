#!/bin/bash

# AI Image Generation Tool - Setup Script

set -e

echo "🎨 AI Image Generation Tool - Setup"
echo "===================================="

# Check Python version
echo "Checking Python version..."
python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' \
    || { echo "❌ Python 3.10+ is required."; exit 1; }

# Create virtual environment when needed. Calling its interpreter directly
# avoids stale absolute paths in a relocated activation script.
if [ ! -x "venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
PYTHON_BIN="venv/bin/python"

# Upgrade pip
echo "Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your settings"
fi

# Create directories
echo "Creating necessary directories..."
mkdir -p models/cache outputs

# Download model (optional)
echo ""
echo "Do you want to download the Stable Diffusion model now?"
echo "This may take several minutes depending on your internet connection."
read -p "(y/n): " download_model

if [ "$download_model" = "y" ] || [ "$download_model" = "Y" ]; then
    echo "Downloading model..."
    "$PYTHON_BIN" -c "from app.core.config import settings; from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained(settings.MODEL_ID, cache_dir=settings.MODEL_CACHE_DIR, use_safetensors=True)"
    echo "✅ Model downloaded successfully!"
else
    echo "⏭️  Skipping model download. Model will be downloaded on first run."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  ./scripts/run.sh start"
echo ""
echo "Then visit:"
echo "  http://localhost:8000/docs  (API documentation)"
echo "  http://localhost:8000       (Root endpoint)"
echo ""
echo "Or run with Docker:"
echo "  docker compose up -d"
