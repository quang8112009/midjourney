#!/usr/bin/env bash
set -euo pipefail

# Build script for arXiv paper submission
MAIN="main"

echo "Building ${MAIN}.pdf..."

if command -v tectonic &> /dev/null; then
    echo "Using tectonic..."
    tectonic "${MAIN}.tex"
elif command -v latexmk &> /dev/null; then
    echo "Using latexmk..."
    latexmk -pdf "${MAIN}.tex"
elif command -v pdflatex &> /dev/null; then
    echo "Using pdflatex + bibtex..."
    pdflatex -interaction=nonstopmode "${MAIN}.tex"
    bibtex "${MAIN}"
    pdflatex -interaction=nonstopmode "${MAIN}.tex"
    pdflatex -interaction=nonstopmode "${MAIN}.tex"
else
    echo "Error: No supported LaTeX engine found (tectonic, latexmk, or pdflatex)."
    exit 1
fi

echo "Build successful: ${MAIN}.pdf"
