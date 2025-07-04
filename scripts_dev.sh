#!/bin/bash
# Development Environment Setup Script
# Central Bank Speech Analysis Platform

set -e

echo "🏛️  Setting up Central Bank Speech Analysis Platform..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLP models
echo "Downloading NLP models..."
python -m spacy download en_core_web_sm

# Setup environment file
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Initialize database
echo "Initializing database..."
if [ ! -d "alembic/versions" ]; then
    alembic init alembic
fi

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pip install pre-commit
pre-commit install

# Run tests
echo "Running tests..."
pytest testing/ -v

echo "✅ Setup complete!"
echo ""
echo "To start the development server:"
echo "  source venv/bin/activate"
echo "  python tools/cli.py health"
echo "  uvicorn api.main:app --reload"
echo ""
echo "To collect speeches:"
echo "  python tools/cli.py collect --institution FED --start 2024-01-01 --end 2024-01-31"