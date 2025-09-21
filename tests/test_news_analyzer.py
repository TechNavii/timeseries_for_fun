"""Unit tests for the NewsAnalyzer module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from src.data.news_analyzer import NewsAnalyzer

class TestNewsAnalyzer(unittest.TestCase):
    """Test cases for NewsAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = NewsAnalyzer()

    def test_analyzer_initialization(self):
        """Test NewsAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        # Check if API key is loaded from environment
        api_key = os.getenv('OPENAI_API_KEY')
        self.assertEqual(self.analyzer.openai_api_key, api_key)

    @patch('src.data.news_analyzer.yf.Ticker')
    def test_fetch_news_yahoo_success(self, mock_ticker):
        """Test successful Yahoo Finance news fetching."""
        # Mock Yahoo Finance response
        mock_news = [
            {
                'title': 'Test News 1',
                'publisher': 'Test Publisher',
                'link': 'https://example.com/news1',
                'providerPublishTime': 1640000000,
                'summary': 'Test summary 1'
            },
            {
                'title': 'Test News 2',
                'publisher': 'Test Publisher 2',
                'link': 'https://example.com/news2',
                'providerPublishTime': 1640000100,
                'summary': 'Test summary 2'
            }
        ]

        mock_ticker_instance = Mock()
        mock_ticker_instance.news = mock_news
        mock_ticker.return_value = mock_ticker_instance

        # Fetch news
        articles = self.analyzer.fetch_news_yahoo('AAPL')

        # Assertions
        self.assertIsInstance(articles, list)
        self.assertEqual(len(articles), 2)
        self.assertEqual(articles[0]['title'], 'Test News 1')
        self.assertIn('publishedAt', articles[0])
        self.assertIn('source', articles[0])

    @patch('src.data.news_analyzer.yf.Ticker')
    def test_fetch_news_yahoo_empty(self, mock_ticker):
        """Test Yahoo Finance news fetching with no news."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.news = []
        mock_ticker.return_value = mock_ticker_instance

        articles = self.analyzer.fetch_news_yahoo('AAPL')

        self.assertIsInstance(articles, list)
        self.assertEqual(len(articles), 0)

    @patch('src.data.news_analyzer.yf.Ticker')
    def test_fetch_news_yahoo_error(self, mock_ticker):
        """Test Yahoo Finance news fetching error handling."""
        mock_ticker.side_effect = Exception("API Error")

        articles = self.analyzer.fetch_news_yahoo('AAPL')

        self.assertIsInstance(articles, list)
        self.assertEqual(len(articles), 0)

    @patch('src.data.news_analyzer.OpenAI')
    def test_analyze_sentiment_success(self, mock_openai):
        """Test successful sentiment analysis."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "overall_sentiment": "positive",
            "sentiment_score": 0.7,
            "confidence": 0.85,
            "key_factors": ["strong earnings", "product launch"],
            "articles_analyzed": 5
        }
        """

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Set API key for test
        self.analyzer.openai_api_key = 'test-key'

        # Sample articles
        articles = [
            {'title': 'Company reports record earnings', 'summary': 'Positive news'},
            {'title': 'New product launch announced', 'summary': 'Exciting development'}
        ]

        # Analyze sentiment
        result = self.analyzer.analyze_sentiment(articles, 'AAPL')

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(result['overall_sentiment'], 'positive')
        self.assertEqual(result['sentiment_score'], 0.7)
        self.assertEqual(result['confidence'], 0.85)
        self.assertIn('key_factors', result)
        self.assertEqual(len(result['key_factors']), 2)

    def test_analyze_sentiment_no_api_key(self):
        """Test sentiment analysis without API key."""
        self.analyzer.openai_api_key = None

        articles = [
            {'title': 'Test news', 'summary': 'Test summary'}
        ]

        result = self.analyzer.analyze_sentiment(articles, 'AAPL')

        self.assertIsNone(result)

    def test_analyze_sentiment_empty_articles(self):
        """Test sentiment analysis with empty articles."""
        self.analyzer.openai_api_key = 'test-key'

        result = self.analyzer.analyze_sentiment([], 'AAPL')

        self.assertIsNone(result)

    @patch('src.data.news_analyzer.OpenAI')
    def test_analyze_sentiment_api_error(self, mock_openai):
        """Test sentiment analysis with API error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        self.analyzer.openai_api_key = 'test-key'

        articles = [
            {'title': 'Test news', 'summary': 'Test summary'}
        ]

        result = self.analyzer.analyze_sentiment(articles, 'AAPL')

        self.assertIsNone(result)

    @patch.object(NewsAnalyzer, 'fetch_news_yahoo')
    @patch.object(NewsAnalyzer, 'analyze_sentiment')
    def test_get_news_sentiment_success(self, mock_analyze, mock_fetch):
        """Test complete news sentiment workflow."""
        # Mock Yahoo news
        mock_fetch.return_value = [
            {'title': 'News 1', 'summary': 'Summary 1'},
            {'title': 'News 2', 'summary': 'Summary 2'}
        ]

        # Mock sentiment analysis
        mock_analyze.return_value = {
            'overall_sentiment': 'positive',
            'sentiment_score': 0.8,
            'confidence': 0.9,
            'key_factors': ['growth'],
            'articles_analyzed': 2
        }

        self.analyzer.openai_api_key = 'test-key'

        # Get news sentiment
        result = self.analyzer.get_news_sentiment('AAPL')

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['sentiment'], 'positive')
        self.assertEqual(result['sentiment_score'], 0.8)
        self.assertEqual(result['articles_count'], 2)
        self.assertIn('impact_on_prediction', result)

    @patch.object(NewsAnalyzer, 'fetch_news_yahoo')
    def test_get_news_sentiment_no_articles(self, mock_fetch):
        """Test news sentiment with no articles found."""
        mock_fetch.return_value = []

        result = self.analyzer.get_news_sentiment('AAPL')

        self.assertEqual(result['status'], 'no_data')
        self.assertIn('No news articles found', result['message'])

    def test_get_news_sentiment_no_api_key(self):
        """Test news sentiment without API key (Yahoo only)."""
        self.analyzer.openai_api_key = None

        with patch.object(self.analyzer, 'fetch_news_yahoo') as mock_fetch:
            mock_fetch.return_value = [
                {'title': 'News 1', 'summary': 'Summary 1'}
            ]

            result = self.analyzer.get_news_sentiment('AAPL')

            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['sentiment'], 'neutral')
            self.assertEqual(result['source'], 'Yahoo Finance')

    def test_calculate_impact_on_prediction(self):
        """Test calculation of sentiment impact on prediction."""
        # Test positive sentiment
        impact_positive = self.analyzer._calculate_impact_on_prediction(0.8, 0.9)
        self.assertIn('price_adjustment', impact_positive)
        self.assertIn('confidence_adjustment', impact_positive)
        self.assertIn('suggested_action', impact_positive)
        self.assertGreater(impact_positive['price_adjustment'], 0)
        self.assertIn('Bullish', impact_positive['suggested_action'])

        # Test negative sentiment
        impact_negative = self.analyzer._calculate_impact_on_prediction(-0.8, 0.9)
        self.assertLess(impact_negative['price_adjustment'], 0)
        self.assertIn('Bearish', impact_negative['suggested_action'])

        # Test neutral sentiment
        impact_neutral = self.analyzer._calculate_impact_on_prediction(0.1, 0.5)
        self.assertAlmostEqual(impact_neutral['price_adjustment'], 0.002, places=3)
        self.assertIn('Neutral', impact_neutral['suggested_action'])

    def test_model_selection(self):
        """Test correct model selection from environment."""
        # Test with environment variable set
        with patch.dict(os.environ, {'OPENAI_MODEL': 'legacy-model'}):
            analyzer = NewsAnalyzer()
            self.assertEqual(analyzer.model, 'gpt-5-mini')

        # Test with default
        with patch.dict(os.environ, {}, clear=True):
            analyzer = NewsAnalyzer()
            self.assertEqual(analyzer.model, 'gpt-5-mini')

if __name__ == '__main__':
    unittest.main()
