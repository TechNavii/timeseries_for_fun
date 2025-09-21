"""Configuration settings for the Stock Prediction Application."""
from typing import Dict, List, Any

# Application Settings
APP_CONFIG = {
    "title": "🤖 AI Stock Price Prediction System",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model Configuration
MODEL_CONFIG = {
    "default_model": "xgboost",
    "available_models": ["xgboost", "lightgbm", "random_forest", "ensemble"],
    "model_descriptions": {
        "xgboost": "Gradient Boosting - Fast and accurate",
        "lightgbm": "Light Gradient Boosting - Efficient for large datasets",
        "random_forest": "Random Forest - Robust to overfitting",
        "ensemble": "Ensemble - Combines multiple models for better accuracy"
    }
}

# XGBoost Parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# LightGBM Parameters
LIGHTGBM_PARAMS = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'objective': 'regression',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

# Random Forest Parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Ensemble Model Parameters
ENSEMBLE_PARAMS = {
    'xgb': {
        'n_estimators': 50,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'rf': {
        'n_estimators': 50,
        'max_depth': 5,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Prediction Settings
PREDICTION_CONFIG = {
    "default_horizon": 5,
    "min_horizon": 1,
    "max_horizon": 30,
    "confidence_thresholds": {
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3
    },
    "prediction_threshold": 0.01,  # Threshold for UP/DOWN/NEUTRAL classification
    "uncertainty_iterations": 10,   # Number of iterations for uncertainty estimation
    "uncertainty_noise": 0.001,     # Noise level for uncertainty estimation
    "min_samples_for_training": 50
}

# Data Settings
DATA_CONFIG = {
    "default_lookback_days": 365,
    "max_news_articles": 10,
    "news_lookback_days": 7,
    "cache_ttl_minutes": 15
}

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    "moving_averages": [20, 50],
    "rsi_period": 14,
    "bollinger_bands_period": 20,
    "bollinger_bands_std": 2,
    "atr_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9
}

# Feature Explanations for UI
FEATURE_EXPLANATIONS = {
    'RSI': 'Relative Strength Index (Momentum)',
    'MACD': 'Moving Average Convergence Divergence',
    'SMA_20': '20-Day Simple Moving Average',
    'SMA_50': '50-Day Simple Moving Average',
    'EMA_20': '20-Day Exponential Moving Average',
    'EMA_50': '50-Day Exponential Moving Average',
    'Volume_Ratio': 'Volume vs Average Volume',
    'Volatility': 'Price Volatility (Risk)',
    'BB_Upper': 'Bollinger Band Upper (Resistance)',
    'BB_Lower': 'Bollinger Band Lower (Support)',
    'ATR': 'Average True Range (Volatility)',
    'Return_5d': '5-Day Return Performance',
    'Return_20d': '20-Day Return Performance',
    'OBV': 'On-Balance Volume (Accumulation)',
    'Price_to_SMA20': 'Price vs 20-Day Average',
    'Price_to_SMA50': 'Price vs 50-Day Average',
    'Volume_MA': 'Volume Moving Average',
    'High_Low_Ratio': 'High/Low Price Ratio',
    'Close_Open_Ratio': 'Close/Open Price Ratio'
}

# Confidence Level Messages
CONFIDENCE_MESSAGES = {
    "high": "✅ **High Confidence**: Strong signals align across multiple indicators",
    "medium": "⚠️ **Moderate Confidence**: Mixed signals, exercise caution",
    "low": "❌ **Low Confidence**: Weak or conflicting signals, high uncertainty"
}

# Chart Configuration
CHART_CONFIG = {
    "template": "plotly_dark",
    "candlestick_height": 400,
    "volume_height": 200,
    "gauge_height": 300,
    "feature_chart_height": 250,
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10}
}

# Market Configuration
MARKET_CONFIG = {
    "us_stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "JNJ"],
    "japan_stocks": ["7203.T", "9984.T", "6758.T", "6861.T", "8306.T", "7974.T", "6902.T", "9432.T", "6501.T", "4063.T"],
    "market_suffixes": {
        "japan": ".T",
        "us": ""
    }
}

# Color Schemes
COLORS = {
    "positive": "#00ff00",
    "negative": "#ff4444",
    "neutral": "#999999",
    "warning": "#ffaa00",
    "success": "#44ff44",
    "danger": "#ff0000",
    "info": "#0088ff"
}

# Validation Rules
VALIDATION = {
    "min_ticker_length": 1,
    "max_ticker_length": 10,
    "japan_ticker_digits": 4,
    "min_data_points": 20,
    "max_api_retries": 3,
    "api_timeout_seconds": 30
}

# Error Messages
ERROR_MESSAGES = {
    "insufficient_data": "Insufficient data for prediction. Need at least {min} data points.",
    "api_error": "API error occurred: {error}",
    "invalid_ticker": "Invalid ticker symbol: {ticker}",
    "model_not_trained": "Model not trained yet. Please train the model first.",
    "news_fetch_failed": "Failed to fetch news: {error}",
    "prediction_failed": "Prediction failed: {error}",
    "data_fetch_failed": "Failed to fetch stock data: {error}"
}

# Success Messages
SUCCESS_MESSAGES = {
    "data_fetched": "✅ Fetched {count} days of data",
    "news_fetched": "✅ News analysis complete: {sentiment} sentiment",
    "model_trained": "✅ Model trained successfully",
    "prediction_complete": "✅ Prediction completed"
}