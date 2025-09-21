#!/bin/bash

# Start the Stock Prediction App with proper environment

echo "🚀 Starting Stock Prediction Application..."

# Navigate to the app directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check for required packages
echo "🔍 Checking dependencies..."
python -c "import streamlit, plotly, pandas, yfinance, xgboost" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing missing dependencies..."
    pip install -r requirements.txt
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating from example..."
    cp .env.example .env 2>/dev/null || echo "Please configure .env file with your API keys"
fi

# Start the application
echo "✨ Starting Streamlit application..."
echo "📊 Access the app at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"

streamlit run app.py