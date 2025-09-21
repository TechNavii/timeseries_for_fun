"""Streamlit GUI for Stock Prediction System.

This application is provided for fun and educational purposes only. It does not
constitute financial advice, and the authors cannot be held responsible for any
actions taken based on its output.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Import our modules
from src.data.fetcher import MarketDataFetcher
from src.features.engineer import FeatureEngineer
from src.models.predictor import PredictionService
from src.core.database import db_manager

# Page configuration
st.set_page_config(
    page_title="AI Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Fix white background issues */
    .stMetric {
        background-color: rgba(28, 31, 38, 0.8) !important;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 31, 38, 0.8) !important;
        border: 1px solid rgba(250, 250, 250, 0.2);
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="metric-container"] label {
        color: rgba(250, 250, 250, 0.9) !important;
    }
    .prediction-up {
        color: #00ff00;
        font-weight: bold;
    }
    .prediction-down {
        color: #ff4444;
        font-weight: bold;
    }
    .prediction-neutral {
        color: #999999;
        font-weight: bold;
    }
    /* Fix expander background */
    div[data-testid="stExpander"] {
        background-color: rgba(28, 31, 38, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_service' not in st.session_state:
    st.session_state.prediction_service = PredictionService()

if 'fetcher' not in st.session_state:
    st.session_state.fetcher = MarketDataFetcher()

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Data persistence across model switches
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

if 'news_cache' not in st.session_state:
    st.session_state.news_cache = {}

if 'predictions_cache' not in st.session_state:
    st.session_state.predictions_cache = {}

if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None

def main():
    """Main application function."""
    st.title("🤖 AI Stock Price Prediction System")
    st.markdown("**Predict US and Japanese stock prices using advanced ML models**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Market selection
        market = st.selectbox(
            "Select Market",
            ["US", "Japan", "Both"],
            help="Choose which market to analyze"
        )

        # Get popular stocks
        stocks = st.session_state.fetcher.get_popular_stocks()

        if market == "US":
            available_symbols = stocks['US']
        elif market == "Japan":
            available_symbols = stocks['JP']
        else:
            available_symbols = stocks['US'] + stocks['JP']

        # Stock selection
        st.subheader("📊 Stock Selection")

        # Search box with help text
        search_term = st.text_input(
            "🔍 Search ticker",
            placeholder="Enter symbol (e.g., AAPL, 7203, NVDA)",
            help="For Japanese stocks, enter just the number (e.g., 7203 for Toyota, 7974 for Nintendo)"
        )

        if search_term:
            # Process the search term
            search_term = search_term.upper().strip()

            # Check if it's a Japanese stock number (4 digits)
            if search_term.isdigit() and len(search_term) == 4:
                # Add Tokyo Stock Exchange suffix
                selected_symbol = f"{search_term}.T"
                st.info(f"🇯🇵 Japanese stock detected: {selected_symbol}")
            elif search_term.isdigit() and len(search_term) in [1, 2, 3]:
                # Short number, likely Japanese ETF or special stock
                selected_symbol = f"{search_term}.T"
                st.info(f"🇯🇵 Japanese ticker: {selected_symbol}")
            else:
                # US or already formatted ticker
                selected_symbol = search_term
        else:
            # Select from popular stocks
            selected_symbol = st.selectbox(
                "Or select from popular stocks",
                available_symbols,
                format_func=lambda x: x  # Just show the symbol, don't fetch names
            )

        # Date range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )

        # Prediction settings
        st.subheader("🎯 Prediction Settings")
        prediction_days = st.slider(
            "Prediction Horizon (days)",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of days ahead to predict"
        )

        if 'previous_model_type' not in st.session_state:
            st.session_state.previous_model_type = None

        model_type = st.selectbox(
            "Model Type",
            ["xgboost", "lightgbm", "random_forest", "ensemble", "chronos", "patchtst"],
            help="Select the ML model to use:\n- XGBoost/LightGBM: Fast gradient boosting\n- Random Forest: Robust ensemble method\n- Ensemble: Combines multiple models\n- Chronos: Amazon's zero-shot transformer (no training needed)\n- PatchTST: State-of-the-art transformer for time series",
            key="model_selector"
        )

        if st.session_state.previous_model_type != model_type:
            st.session_state.previous_model_type = model_type
            # Don't clear prediction results, but mark model as changed
            # This allows switching between models without losing data
            if selected_symbol and selected_symbol in st.session_state.predictions_cache:
                if model_type in st.session_state.predictions_cache[selected_symbol]:
                    st.session_state.prediction_results = st.session_state.predictions_cache[selected_symbol][model_type]

        # Action buttons
        st.subheader("🚀 Actions")
        fetch_button = st.button("📥 Fetch Data", use_container_width=True, type="primary")
        fetch_news_button = st.button("📰 Fetch News", use_container_width=True)
        predict_button = st.button("🔮 Make Prediction", use_container_width=True)

        # Cache management
        with st.expander("🗑️ Cache Management"):
            if st.button("Clear All Caches", use_container_width=True):
                st.session_state.data_cache = {}
                st.session_state.news_cache = {}
                st.session_state.predictions_cache = {}
                st.success("✅ All caches cleared")

    # Main content area
    col1, col2, col3 = st.columns([2, 3, 2])

    with col1:
        st.subheader("📈 Current Data")

        # Check if we need to fetch data or use cached
        cache_key = f"{selected_symbol}_{start_date}_{end_date}"

        if fetch_button or (selected_symbol and (st.session_state.current_data is None or st.session_state.current_symbol != selected_symbol)):
            # Check cache first
            if cache_key in st.session_state.data_cache and not fetch_button:
                st.session_state.current_data = st.session_state.data_cache[cache_key]
                st.session_state.current_symbol = selected_symbol
                st.success(f"Loaded cached data for {selected_symbol}")
            else:
                with st.spinner(f"Fetching data for {selected_symbol}..."):
                    # Fetch data
                    data = st.session_state.fetcher.fetch_stock_data(
                        selected_symbol,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )

                    if not data.empty:
                        st.session_state.current_data = data
                        st.session_state.data_cache[cache_key] = data
                        st.session_state.current_symbol = selected_symbol
                        st.success(f"✅ Fetched {len(data)} days of data")
                        # Clear news data when fetching new stock data
                        st.session_state.news_data = None
                    else:
                        st.error(f"❌ No data found for {selected_symbol}")

        # Display current data stats
        if st.session_state.current_data is not None and not st.session_state.current_data.empty:
            data = st.session_state.current_data

            # Current price and stats
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100

            # Check if Japanese stock (ends with .T)
            is_japanese = selected_symbol.endswith('.T')
            currency_symbol = "¥" if is_japanese else "$"
            decimal_places = 0 if is_japanese and current_price > 100 else 2

            # Display metrics
            st.metric(
                "Current Price",
                f"{currency_symbol}{current_price:.{decimal_places}f}",
                f"{price_change:.{decimal_places}f} ({price_change_pct:.2f}%)"
            )

            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("High", f"{currency_symbol}{data['High'].iloc[-1]:.{decimal_places}f}")
                st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
            with col1_2:
                st.metric("Low", f"{currency_symbol}{data['Low'].iloc[-1]:.{decimal_places}f}")
                st.metric("Avg Volume", f"{data['Volume'].mean():,.0f}")

    with col2:
        st.subheader("📊 Price Chart")

        if st.session_state.current_data is not None and not st.session_state.current_data.empty:
            # Create candlestick chart
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=st.session_state.current_data.index,
                open=st.session_state.current_data['Open'],
                high=st.session_state.current_data['High'],
                low=st.session_state.current_data['Low'],
                close=st.session_state.current_data['Close'],
                name='Price'
            ))

            # Add moving averages
            if len(st.session_state.current_data) >= 20:
                sma_20 = st.session_state.current_data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=st.session_state.current_data.index,
                    y=sma_20,
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))

            if len(st.session_state.current_data) >= 50:
                sma_50 = st.session_state.current_data['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=st.session_state.current_data.index,
                    y=sma_50,
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ))

            # Update layout with better scaling
            fig.update_layout(
                title=f"{selected_symbol} Price Chart",
                yaxis_title="Price ($)",
                xaxis_title="Date",
                template="plotly_dark",
                height=400,
                xaxis_rangeslider_visible=False,
                yaxis=dict(
                    autorange=True,
                    fixedrange=False,
                    type='linear'
                ),
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            # Update y-axis to better show price movements
            fig.update_yaxes(automargin=True)

            st.plotly_chart(fig, use_container_width=True)

            # Volume chart
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(
                x=st.session_state.current_data.index,
                y=st.session_state.current_data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ))

            fig_vol.update_layout(
                title="Trading Volume",
                yaxis_title="Volume",
                xaxis_title="Date",
                template="plotly_dark",
                height=200
            )

            st.plotly_chart(fig_vol, use_container_width=True)

    # News fetching section
    if fetch_news_button and st.session_state.current_data is not None:
        # Check cache first
        if selected_symbol in st.session_state.news_cache:
            news_sentiment = st.session_state.news_cache[selected_symbol]
            st.info(f"Using cached news analysis for {selected_symbol}")
        else:
            with st.spinner("🔍 Searching news across Bloomberg, Reuters, Financial Times, WSJ, CNBC, and more..."):
                try:
                    # Use multi-source news analyzer
                    from src.data.news_analyzer import EnhancedNewsAnalyzer
                    news_analyzer = EnhancedNewsAnalyzer()
                    # Use symbol as company name to avoid Yahoo Finance API calls
                    company_name = selected_symbol
                    news_sentiment = news_analyzer.search_and_analyze_news(selected_symbol, company_name)
                    # Cache the result
                    st.session_state.news_cache[selected_symbol] = news_sentiment

                    # Log debug info
                    logger.info(f"News fetch result for {selected_symbol}: status={news_sentiment.get('status')}, articles={news_sentiment.get('articles_count', 0)}")

                    st.session_state.news_data = news_sentiment
                    if news_sentiment.get('status') == 'success':
                        articles_count = news_sentiment.get('articles_count', 0)
                        sources = news_sentiment.get('sources_searched', [])
                        sources_count = len(sources)

                        # Show appropriate message based on stock type
                        if selected_symbol.endswith('.T'):
                            # Japanese stock
                            st.success(f"✅ Analyzed {sources_count} Japanese financial sources including Nikkei Asia, Japan Times, and major securities firms")
                        else:
                            st.success(f"✅ Analyzed {sources_count} major financial news sources")

                        if articles_count > 0:
                            st.info(f"📰 Found {articles_count} relevant articles")
                            sentiment = news_sentiment.get('sentiment', 'neutral').upper()
                            if sentiment == 'POSITIVE':
                                st.success(f"📈 Overall Sentiment: {sentiment}")
                            elif sentiment == 'NEGATIVE':
                                st.error(f"📉 Overall Sentiment: {sentiment}")
                            else:
                                st.info(f"➡️ Overall Sentiment: {sentiment}")
                        else:
                            # For Japanese stocks, provide more context
                            if selected_symbol.endswith('.T'):
                                st.info("📰 Generated market analysis based on Japanese market conditions, yen movements, and BOJ policy")
                            else:
                                st.info("📰 Generated comprehensive market analysis based on current trends and fundamentals")

                            # Still show any market outlook if available
                            if news_sentiment.get('market_outlook'):
                                st.info(f"📊 Market Analysis: {news_sentiment.get('market_outlook')}")
                    elif news_sentiment.get('status') == 'no_api':
                        st.warning("⚠️ Please configure OpenAI API key for comprehensive news analysis")
                    else:
                        st.warning(f"📰 News search status: {news_sentiment.get('status', 'unknown')}")
                except Exception as e:
                    logger.error(f"Failed to fetch news: {e}")
                    st.warning("⚠️ News service temporarily unavailable")
                    st.session_state.news_data = None

    with col3:
        st.subheader("🔮 Prediction Results")

        if predict_button and st.session_state.current_data is not None:
            # Check cache first
            if selected_symbol not in st.session_state.predictions_cache:
                st.session_state.predictions_cache[selected_symbol] = {}

            # Check if we have cached prediction for this model
            if model_type in st.session_state.predictions_cache[selected_symbol]:
                # Use cached prediction
                prediction = st.session_state.predictions_cache[selected_symbol][model_type]
                st.info(f"Using cached {model_type} prediction for {selected_symbol}")
            else:
                with st.spinner("Making prediction..."):
                    # Make prediction
                    prediction = make_prediction(
                        selected_symbol,
                        st.session_state.current_data,
                        prediction_days,
                        model_type
                    )

                    if 'error' not in prediction:
                        # Cache the prediction
                        st.session_state.predictions_cache[selected_symbol][model_type] = prediction

            if 'error' not in prediction:
                st.session_state.prediction_results = prediction

                # Display prediction with range
                predicted_price = prediction['predicted_price']
                current_price = prediction['current_price']
                confidence = prediction['confidence'] * 100

                # Check if Japanese stock
                is_japanese = selected_symbol.endswith('.T')
                currency_symbol = "¥" if is_japanese else "$"
                decimal_places = 0 if is_japanese and current_price > 100 else 2

                # Check if we have adjusted predictions (with news sentiment)
                if 'predicted_price_adjusted' in prediction:
                    predicted_price = prediction['predicted_price_adjusted']
                    price_lower = prediction['price_range_lower_adjusted']
                    price_upper = prediction['price_range_upper_adjusted']
                else:
                    price_lower = prediction.get('price_range_lower', predicted_price * 0.98)
                    price_upper = prediction.get('price_range_upper', predicted_price * 1.02)

                # Calculate predicted return based on final predicted price (adjusted or original)
                predicted_return = ((predicted_price - current_price) / current_price) * 100

                # Main prediction display
                st.metric(
                    f"Predicted Price ({prediction_days} days)",
                    f"{currency_symbol}{predicted_price:.{decimal_places}f}",
                    f"{predicted_return:.2f}%"
                )

                # Price range display
                st.info(f"**Price Range:** {currency_symbol}{price_lower:.{decimal_places}f} - {currency_symbol}{price_upper:.{decimal_places}f}")

                # Model type indicator
                st.caption(f"Model: {prediction.get('model_type', 'xgboost')}")

                # Prediction reasoning
                if 'prediction_reasoning' in prediction:
                    st.markdown("### 📝 Analysis")
                    # Use text_area to show full content with adjustable height
                    st.text_area(
                        "Model Analysis",
                        value=prediction['prediction_reasoning'],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )

                # Prediction class with color
                pred_class = prediction['predicted_class']
                if pred_class == 'UP':
                    st.markdown(f"<h3 class='prediction-up'>📈 {pred_class}</h3>", unsafe_allow_html=True)
                elif pred_class == 'DOWN':
                    st.markdown(f"<h3 class='prediction-down'>📉 {pred_class}</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 class='prediction-neutral'>➡️ {pred_class}</h3>", unsafe_allow_html=True)

                # Enhanced confidence explanation
                st.markdown("### 📊 Confidence Analysis")
                confidence_explanation = f"""
                **Confidence Level: {confidence:.1f}%**

                This confidence score represents:
                - **Model Certainty**: How consistent the predictions are across different model iterations
                - **Prediction Strength**: {abs(predicted_return):.2f}% expected move indicates {'strong' if abs(predicted_return) > 2 else 'moderate' if abs(predicted_return) > 1 else 'weak'} signal
                - **Volatility Factor**: Price range of {currency_symbol}{price_lower:.{decimal_places}f} - {currency_symbol}{price_upper:.{decimal_places}f}
                - **Model Agreement**: {prediction.get('model_type', 'xgboost').upper()} model convergence

                {'✅ **High Confidence**: Strong signals align across multiple indicators' if confidence > 70 else '⚠️ **Moderate Confidence**: Mixed signals, exercise caution' if confidence > 50 else '❌ **Low Confidence**: Weak or conflicting signals, high uncertainty'}
                """
                st.info(confidence_explanation)

                # Confidence gauge with better sizing
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    title={'text': "Confidence Level", 'font': {'size': 20}},
                    delta={'reference': 50, 'relative': False},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ff4444"},
                            {'range': [40, 60], 'color': "#ffaa00"},
                            {'range': [60, 80], 'color': "#44ff44"},
                            {'range': [80, 100], 'color': "#00ff00"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 2},
                            'thickness': 0.75,
                            'value': confidence
                        }
                    }
                ))

                fig_gauge.update_layout(
                    height=300,  # Increased height
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=14)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            else:
                st.error(f"❌ Prediction failed: {prediction['error']}")

    # Technical indicators section
    with st.expander("📈 Technical Indicators", expanded=False):
        if st.session_state.current_data is not None and not st.session_state.current_data.empty:
            # Calculate indicators
            engineer = FeatureEngineer()
            indicators_df = engineer.calculate_technical_indicators(st.session_state.current_data)

            # Display latest indicators
            latest = indicators_df.iloc[-1]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("RSI (14)", f"{latest.get('RSI', 0):.2f}")
                st.metric("MACD", f"{latest.get('MACD', 0):.4f}")

            with col2:
                st.metric("BB Upper", f"${latest.get('BB_Upper', 0):.2f}")
                st.metric("BB Lower", f"${latest.get('BB_Lower', 0):.2f}")

            with col3:
                st.metric("SMA 20", f"${latest.get('SMA_20', 0):.2f}")
                st.metric("SMA 50", f"${latest.get('SMA_50', 0):.2f}")

            with col4:
                st.metric("ATR", f"{latest.get('ATR', 0):.2f}")
                st.metric("Volume Ratio", f"{latest.get('Volume_Ratio', 0):.2f}")

    # News Sentiment Section
    with st.expander("📰 News Sentiment Analysis", expanded=False):
        # Check for OpenAI API key
        has_openai = bool(os.getenv('OPENAI_API_KEY'))

        if not has_openai:
            st.warning(
                "**📝 OpenAI News Analysis Available**\n\n"
                "Enable AI-powered news sentiment analysis:\n"
                "1. Edit the `.env` file in the project root\n"
                "2. Add: `OPENAI_API_KEY=your_key_here`\n"
                "3. Get your key from [OpenAI](https://platform.openai.com/api-keys)\n"
                "4. Restart the app\n\n"
                "*Currently using: Yahoo Finance news only (without sentiment analysis)*"
            )
        else:
            requested_model = os.getenv('OPENAI_MODEL')
            if requested_model and requested_model != 'gpt-5-mini':
                st.info(
                    "Detected OPENAI_MODEL=%s but the application only supports gpt-5-mini; "
                    "defaulting to gpt-5-mini for news analysis." % requested_model
                )
            st.success("✅ **Using gpt-5-mini for intelligent news analysis**")

        # Show news from session state
        news_data = st.session_state.get('news_data', None)

        if news_data and news_data.get('status') == 'success':
            col1, col2, col3 = st.columns(3)

            with col1:
                sentiment = news_data['sentiment']
                # Handle dict sentiment properly
                if isinstance(sentiment, dict):
                    sentiment = sentiment.get('sentiment', 'neutral')
                sentiment_str = str(sentiment) if sentiment else 'neutral'

                if sentiment_str == 'positive':
                    st.success(f"📈 **Overall: {sentiment_str.upper()}**")
                elif sentiment_str == 'negative':
                    st.error(f"📉 **Overall: {sentiment_str.upper()}**")
                else:
                    st.info(f"➡️ **Overall: {sentiment_str.upper()}**")

            with col2:
                st.metric(
                    "Sentiment Score",
                    f"{news_data['sentiment_score']:.2f}",
                    f"{news_data['confidence']*100:.0f}% confidence"
                )

            with col3:
                dist = news_data.get('sentiment_distribution', {})
                if dist:
                    st.write("**Distribution:**")
                    st.write(f"✅ Positive: {dist.get('positive', 0)}")
                    st.write(f"❌ Negative: {dist.get('negative', 0)}")
                    st.write(f"⚪ Neutral: {dist.get('neutral', 0)}")
                else:
                    st.write("**Articles Found:**")
                    st.write(f"📰 {news_data.get('articles_count', 0)} articles")

            # Display key factors if from OpenAI
            if news_data.get('source') == 'OpenAI Analysis' and news_data.get('key_factors'):
                st.subheader("🔍 Key Market Factors:")
                for factor in news_data['key_factors']:
                    st.write(f"• {factor}")
                st.markdown("---")

            # Display recent news
            st.subheader("Recent News & Events:")
            articles = news_data.get('articles', [])
            if not articles:
                st.info("No specific news articles found in the analysis. The sentiment analysis is based on overall market conditions.")
            else:
                for idx, article in enumerate(articles[:5]):
                    with st.container():
                        # Clean up the title - remove URL-like patterns
                        title = article.get('title', '')
                        # If title looks like a URL path, try to extract meaningful parts
                        if '/' in title and '?' in title:
                            # This looks like a URL fragment, clean it up
                            title = title.split('/')[-1].split('?')[0].replace('-', ' ').title()
                        elif title.startswith('http'):
                            # Extract domain and path
                            title = title.split('/')[-1].replace('-', ' ').replace('_', ' ').title()

                        # Article title with numbering
                        st.markdown(f"### {idx + 1}. {title}")

                        # Sentiment analysis with prominent score display
                        if 'sentiment' in article:
                            sentiment_data = article['sentiment']
                            if isinstance(sentiment_data, dict):
                                sentiment_str = str(sentiment_data.get('sentiment', 'neutral'))
                                score = sentiment_data.get('score', 0)
                                reasoning = sentiment_data.get('reasoning', 'Analysis based on content')

                                # Create columns for better layout
                                col1, col2 = st.columns([1, 3])

                                with col1:
                                    # Display sentiment with emoji and score prominently
                                    if sentiment_str == 'positive':
                                        st.success(f"✅ POSITIVE\n**Score: {score:+.2f}**")
                                    elif sentiment_str == 'negative':
                                        st.error(f"❌ NEGATIVE\n**Score: {score:+.2f}**")
                                    else:
                                        st.info(f"⚪ NEUTRAL\n**Score: {score:+.2f}**")

                                with col2:
                                    # Show reasoning for the score
                                    st.write("**Why this score?**")
                                    st.caption(f"{reasoning}")
                            else:
                                # Fallback for simple sentiment string
                                sentiment_str = str(sentiment_data)
                                if sentiment_str == 'positive':
                                    st.success(f"✅ POSITIVE")
                                elif sentiment_str == 'negative':
                                    st.error(f"❌ NEGATIVE")
                                else:
                                    st.info(f"⚪ NEUTRAL")

                        # Show summary if available
                        if article.get('summary') and article['summary'] != title:
                            st.write("**Summary:**")
                            st.caption(f"{article['summary']}")

                        # Source with clickable link
                        source = article.get('source', 'Financial News')
                        link = article.get('link', article.get('url'))
                        published_date = article.get('publishedAt', '')[:10]
                        relevance = article.get('relevance', 'medium')
                        relevance_emoji = "🔴" if relevance == 'high' else "🟡" if relevance == 'medium' else "🔵"

                        # Create source info with link using HTML for new tab
                        if link and link != '#':
                            # Ensure the link has proper protocol
                            if not link.startswith(('http://', 'https://')):
                                link = f"https://{link}"
                            # Use HTML with target="_blank" to open in new tab
                            source_html = f"""
                            <p>📰 <strong>Source:</strong> <a href="{link}" target="_blank" style="color: #1f77b4; text-decoration: underline;">{source}</a> | {relevance_emoji} Relevance: {relevance.upper()} | 📅 {published_date}</p>
                            """
                        else:
                            source_html = f"""
                            <p>📰 <strong>Source:</strong> {source} | {relevance_emoji} Relevance: {relevance.upper()} | 📅 {published_date}</p>
                            """

                        st.markdown(source_html, unsafe_allow_html=True)
                        st.markdown("---")

            # Impact on prediction
            impact = news_data.get('impact_on_prediction', {})
            if impact:
                st.info(f"**News Impact:** {impact.get('suggested_action', 'Neutral outlook')}")
        elif st.session_state.get('current_data') is not None:
            st.info("News sentiment data not available - try fetching data again")
        else:
            st.info("Fetch stock data first to see news sentiment analysis")


    # Footer
    st.markdown("---")
    st.markdown("**Note:** This tool is for fun and educational purposes only. It is not financial advice, and we cannot be held responsible for any outcomes.")

@st.cache_data
def get_stock_name(symbol: str) -> str:
    """Get stock name from symbol - simplified to avoid Yahoo API calls."""
    # Simply return the symbol to avoid 401 errors from Yahoo Finance
    return symbol

def make_prediction(symbol: str, data: pd.DataFrame,
                   prediction_days: int, model_type: str) -> dict:
    """Make stock prediction with news sentiment."""
    try:
        # Initialize prediction service
        service = st.session_state.prediction_service

        # Make prediction with specified model type
        result = service.predict_stock(
            symbol=symbol,
            data=data,
            prediction_horizon=prediction_days,
            model_type=model_type  # Pass the model type
        )

        # Use cached news sentiment from session state
        news_sentiment = st.session_state.get('news_data', None)

        # Only fetch news if it wasn't already fetched manually
        # This prevents redundant API calls when the user already clicked "Fetch News"
        if news_sentiment is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                try:
                    from src.data.news_analyzer import EnhancedNewsAnalyzer
                    news_analyzer = EnhancedNewsAnalyzer()
                    # Use symbol as company name to avoid Yahoo Finance API calls
                    company_name = symbol

                    with st.spinner("🔍 Fetching news for prediction analysis..."):
                        news_sentiment = news_analyzer.search_and_analyze_news(symbol, company_name)
                        st.session_state.news_data = news_sentiment
                except Exception as e:
                    logger.warning(f"Failed to fetch news during prediction: {e}")
                    news_sentiment = None

        # Check if prediction was successful before adjusting for sentiment
        if 'error' not in result and 'current_price' in result:
            # Adjust prediction based on news sentiment
            if news_sentiment and news_sentiment.get('status') == 'success':
                sentiment_impact = news_sentiment['impact_on_prediction']

                # Adjust predicted price based on sentiment
                adjustment = result['current_price'] * sentiment_impact['price_adjustment']
                result['predicted_price_adjusted'] = result['predicted_price'] + adjustment
                result['price_range_lower_adjusted'] = result.get('price_range_lower', result['predicted_price'] * 0.98) + adjustment
                result['price_range_upper_adjusted'] = result.get('price_range_upper', result['predicted_price'] * 1.02) + adjustment

                # Add news data to result
                result['news_sentiment'] = news_sentiment

        return result

    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    main()
