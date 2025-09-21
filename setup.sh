#!/bin/bash

# Setup script for Stock Prediction System
echo "🚀 Setting up Stock Prediction System in virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies first (lighter packages)
echo "📦 Installing core dependencies..."
pip install pandas numpy yfinance fastapi uvicorn streamlit plotly \
    redis psycopg2-binary sqlalchemy python-dotenv pyyaml click \
    aiohttp websockets prometheus-client tqdm

# Install ML libraries (these are larger)
echo "🤖 Installing ML libraries..."
pip install scikit-learn xgboost lightgbm

# Note: TensorFlow and PyTorch are large, we'll make them optional
echo "⚠️  Note: TensorFlow and PyTorch are large packages. Install them separately if needed:"
echo "  pip install tensorflow==2.14.0"
echo "  pip install torch==2.1.0"

# Create project directories
echo "📁 Creating project structure..."
mkdir -p src/{core,data,models,features,api,gui,utils}
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p tests

echo "✅ Setup complete! Activate the environment with: source venv/bin/activate"