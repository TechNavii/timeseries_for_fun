"""Enhanced News Analyzer using OpenAI to search across multiple news sources."""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re

try:
    from openai import OpenAI as _OpenAI
except ImportError:  # pragma: no cover - optional dependency
    _OpenAI = None

OpenAI = _OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

class EnhancedNewsAnalyzer:
    """Analyze news sentiment using OpenAI to search multiple news sources."""

    def __init__(self):
        """Initialize enhanced news analyzer."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        requested_model = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
        if requested_model != 'gpt-5-mini':
            logger.warning(
                "OPENAI_MODEL set to %s but only gpt-5-mini is supported; defaulting to gpt-5-mini",
                requested_model,
            )
            requested_model = 'gpt-5-mini'
        self.model = requested_model

        # Cache for company names to avoid repeated API calls
        self._company_name_cache = {}

        # Major financial news sources to search
        self.news_sources = [
            "Bloomberg", "Reuters", "Financial Times", "Wall Street Journal",
            "CNBC", "MarketWatch", "Yahoo Finance", "Seeking Alpha",
            "The Motley Fool", "Barron's", "Forbes", "Business Insider",
            "CNN Business", "Fox Business", "The Economist", "Nikkei"
        ]

        # Japanese financial news sources
        self.japanese_sources = [
            "Nikkei Asia", "Japan Times", "Toyo Keizai", "Nikkei Business",
            "Japan Exchange Group", "Tokyo Stock Exchange", "Mizuho Research",
            "Nomura Research", "SMBC Nikko", "Daiwa Securities"
        ]

    def get_company_name(self, symbol: str) -> str:
        """
        Dynamically get proper company name for news search using multiple sources.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Company name (Japanese name for Japanese stocks, English for others)
        """
        # Check cache first
        if symbol in self._company_name_cache:
            return self._company_name_cache[symbol]

        company_name = self._try_yfinance(symbol)
        if not company_name:
            company_name = self._try_web_scraping(symbol) or symbol

        self._company_name_cache[symbol] = company_name
        return company_name

    def _try_yfinance(self, symbol: str) -> Optional[str]:
        """Try to get company name from yfinance."""
        info = {}
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            if not isinstance(info, dict):
                logger.debug(f"yfinance info is not a dict for {symbol}: {type(info)}")
                info = {}

            if len(info) <= 1:
                logger.debug(f"yfinance info payload too small for {symbol}: {info}")
                info = {}

            # For Japanese stocks, prefer Japanese name if available
            if symbol.endswith('.T') or symbol.endswith('.JP'):
                # Try to get Japanese name first
                japanese_name = info.get('longNameJP') or info.get('shortNameJP')
                if japanese_name:
                    return japanese_name

                # Fallback to regular name, but try to detect if it's already Japanese
                long_name = info.get('longName', '')
                short_name = info.get('shortName', '')

                # If the name contains Japanese characters, use it
                if long_name and self._contains_japanese(long_name):
                    return long_name
                elif short_name and self._contains_japanese(short_name):
                    return short_name
                else:
                    # Otherwise use English name
                    return long_name or short_name
            else:
                # For non-Japanese stocks, use English name
                long_name = info.get('longName', '')
                short_name = info.get('shortName', '')
                return long_name or short_name

        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")

        return None

    def _try_web_scraping(self, symbol: str) -> Optional[str]:
        """Try to get company name through web scraping."""
        try:
            # For Japanese stocks, try scraping from Yahoo Finance Japan
            if symbol.endswith('.T') or symbol.endswith('.JP'):
                return self._scrape_yahoo_finance_japan(symbol)
            else:
                # For US stocks, try alternative sources
                return self._scrape_yahoo_finance_us(symbol)

        except Exception as e:
            logger.warning(f"Web scraping failed for {symbol}: {e}")

        return info.get('longName') or info.get('shortName') or symbol

    def _scrape_yahoo_finance_japan(self, symbol: str) -> Optional[str]:
        """Scrape company name from Yahoo Finance Japan."""
        try:
            # Remove .T suffix for the URL
            ticker_code = symbol.replace('.T', '')
            url = f"https://stocks.finance.yahoo.co.jp/stocks/detail/?code={ticker_code}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Try to extract from title first (most reliable)
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                # Extract company name from title like "任天堂(株)【7974】：株価・株式情報"
                match = re.search(r'^([^【]+)', title_text)
                if match:
                    company_name = match.group(1).strip()
                    # Remove (株) and other corporate suffixes
                    company_name = re.sub(r'\(株\)|\(有\)|\(合\)', '', company_name).strip()
                    if company_name and self._contains_japanese(company_name):
                        logger.info(f"Scraped Japanese company name from title: {symbol} -> {company_name}")
                        return company_name

            # Fallback to h1 tags
            h1_tags = soup.find_all('h1')
            for h1 in h1_tags:
                h1_text = h1.get_text().strip()
                if h1_text and self._contains_japanese(h1_text):
                    # Extract company name from text like "任天堂(株)の株価・株式情報"
                    # Remove everything after の, を, は, etc.
                    company_name = re.split(r'[のをは【]', h1_text)[0].strip()
                    # Remove corporate suffixes
                    company_name = re.sub(r'\(株\)|\(有\)|\(合\)', '', company_name).strip()
                    if company_name and self._contains_japanese(company_name):
                        logger.info(f"Scraped Japanese company name from h1: {symbol} -> {company_name}")
                        return company_name

        except Exception as e:
            logger.warning(f"Failed to scrape Yahoo Finance Japan for {symbol}: {e}")

        return None

    def _scrape_yahoo_finance_us(self, symbol: str) -> Optional[str]:
        """Scrape company name from Yahoo Finance US."""
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for company name
            selectors = [
                'h1[data-reactid*="quote-header"]',
                'h1.D\\(ib\\)',
                '.quote-header h1',
                'h1[class*="quote"]'
            ]

            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    company_name = element.get_text().strip()
                    # Clean up the name (remove ticker and extra parentheses)
                    company_name = re.sub(r'\([^)]*\)$', '', company_name).strip()
                    if company_name and company_name != symbol:
                        logger.info(f"Scraped US company name: {symbol} -> {company_name}")
                        return company_name

        except Exception as e:
            logger.warning(f"Failed to scrape Yahoo Finance US for {symbol}: {e}")

        return None

    def _contains_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters."""
        if not text:
            return False
        # Check for Hiragana, Katakana, and Kanji characters
        japanese_ranges = [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FAF),  # Kanji
        ]
        for char in text:
            char_code = ord(char)
            for start, end in japanese_ranges:
                if start <= char_code <= end:
                    return True
        return False

    def search_and_analyze_news(self, symbol: str, company_name: str = None) -> Dict:
        """
        Use OpenAI to search for news across multiple sources and analyze sentiment.

        Args:
            symbol: Stock ticker symbol
            company_name: Company name

        Returns:
            Comprehensive news analysis from multiple sources
        """
        # Use mapped company name if not provided
        if not company_name or company_name == symbol:
            company_name = self.get_company_name(symbol)

        logger.info(f"Starting news search for symbol: {symbol}, company: {company_name}")

        if not self.openai_api_key or self.openai_api_key == 'your_openai_api_key_here':
            logger.warning("OpenAI API key not configured")
            # Try fallback analysis
            return self.get_fallback_analysis(symbol, company_name)

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            # Get current date for context
            current_date = datetime.now().strftime('%Y-%m-%d')

            # Create a comprehensive search prompt
            # Use Japanese sources for Japanese stocks
            is_japanese_stock = symbol.endswith('.T')
            if is_japanese_stock:
                # Prioritize Japanese economy and financial news sources
                primary_japanese_sources = [
                    "Nikkei.com", "Toyo Keizai Online", "Diamond Online", "President Online",
                    "BusinessInsider Japan", "Yahoo Finance Japan", "Kabutan.jp",
                    "SBI Securities News", "Rakuten Securities News", "Japan Exchange Group"
                ]
                sources_to_search = primary_japanese_sources + self.japanese_sources
                sources_description = "Major Japanese financial and economy sources: " + ', '.join(primary_japanese_sources[:6])
            else:
                sources_to_search = self.news_sources[:8]
                sources_description = ', '.join(sources_to_search)

            search_prompt = f"""
            You are a financial news analyst with comprehensive knowledge of global markets and recent events.
            Today's date is {current_date}.

            Analyze {symbol} ({company_name if company_name else 'stock'}) and provide a realistic news sentiment analysis
            based on your knowledge of recent market events, company performance, and industry trends.

            {"IMPORTANT: This is a JAPANESE stock. You MUST primarily use Japanese financial news sources. At least 70% of the articles should be from Japanese sources like Nikkei, Toyo Keizai, Diamond Online, Kabutan, Yahoo Finance Japan, BusinessInsider Japan, President Online, Nikkei Business, etc. Include Japanese market perspectives, yen movements, BOJ policy impacts, and domestic economic factors. Search for news using the Japanese company name: " + company_name if is_japanese_stock else ""}

            Sources to {"MUST USE (prioritize Japanese sources)" if is_japanese_stock else "consider"}: {sources_description}

            Focus on finding:
            1. Recent earnings reports or financial results
            2. Product launches or announcements
            3. Management changes or strategic decisions
            4. Analyst ratings and price target changes
            5. Regulatory news or legal developments
            6. Market trends affecting the stock
            7. Merger & acquisition activity
            8. Industry-specific developments

            Based on your knowledge, create a realistic analysis that would typically be found in recent news.
            {"Generate at least 5 articles, with majority from Japanese sources." if is_japanese_stock else ""}
            Format your response as JSON:
            {{
                "articles": [
                    {{
                        "title": "{"Japanese or English headline reflecting Japanese market perspective" if is_japanese_stock else "Representative headline based on recent events"}",
                        "source": "{"Must be a Japanese news source (e.g., Nikkei, Toyo Keizai, Diamond Online, Kabutan, etc.)" if is_japanese_stock else "Appropriate news source"}",
                        "date": "Recent date in YYYY-MM-DD format",
                        "summary": "2-3 sentence analysis of the event and its market impact",
                        "url": "Generate a realistic URL for the news source (e.g., https://nikkei.com/article/DGXZQOUB123ABC/ for Japanese sources or https://bloomberg.com/news/articles/2024-01-01/article-title for international sources)",
                        "sentiment": {{
                            "sentiment": "positive/negative/neutral",
                            "score": -1.0 to 1.0,
                            "reasoning": "1-2 sentence rationale"
                        }},
                        "impact_score": -1.0 to 1.0,
                        "relevance": "high/medium/low"
                    }}
                ],
                "overall_sentiment": "positive/negative/neutral",
                "sentiment_score": -1.0 to 1.0,
                "confidence": 0.0 to 1.0,
                "key_factors": ["specific market factor 1", "specific factor 2", "specific factor 3"],
                "market_outlook": "Detailed 2-3 sentence analysis of the stock's outlook based on market conditions",
                "news_volume": "high/medium/low",
                "sentiment_distribution": {{
                    "positive": count,
                    "negative": count,
                    "neutral": count
                }},
                "major_themes": ["current theme1", "theme2", "theme3"],
                "sources_searched": ["Source 1", "Source 2", "Source 3"]
            }}

            Provide 5-8 realistic news items based on your knowledge of the company and market.
            Be specific about likely events, earnings, product launches, or market conditions.

            {"For Japanese stocks, focus on yen impact, BOJ policy effects, domestic consumption trends, and relevant Japanese market factors. Include analysis from Japanese financial institutions." if is_japanese_stock else ""}

            IMPORTANT: Generate realistic, knowledge-based analysis rather than stating inability to fetch news.
            """

            # Use GPT-5 responses API with web search tools
            if self.model.startswith("gpt-5"):
                json_schema = {
                    "type": "object",
                    "properties": {
                        "sentiment_score": {"type": "number"},
                        "overall_sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"]
                        },
                        "confidence": {"type": "number"},
                        "key_factors": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "market_outlook": {"type": "string"},
                        "major_themes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "news_volume": {"type": "string"},
                        "sentiment_distribution": {
                            "type": "object",
                            "properties": {
                                "positive": {"type": "integer"},
                                "negative": {"type": "integer"},
                                "neutral": {"type": "integer"}
                            },
                            "required": ["positive", "negative", "neutral"],
                            "additionalProperties": False
                        },
                        "sources_searched": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "articles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "source": {"type": "string"},
                                    "date": {"type": "string"},
                                    "summary": {"type": "string"},
                                    "url": {"type": "string"},
                                    "sentiment": {
                                        "type": "object",
                                        "properties": {
                                            "sentiment": {
                                                "type": "string",
                                                "enum": ["positive", "neutral", "negative"]
                                            },
                                            "score": {"type": "number"},
                                            "reasoning": {"type": "string"}
                                        },
                                        "required": ["sentiment", "score", "reasoning"],
                                        "additionalProperties": False
                                    },
                                    "impact_score": {"type": "number"},
                                    "relevance": {"type": "string"}
                                },
                                "required": [
                                    "title",
                                    "source",
                                    "date",
                                    "summary",
                                    "url",
                                    "sentiment",
                                    "impact_score",
                                    "relevance"
                                ],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": [
                        "sentiment_score",
                        "overall_sentiment",
                        "confidence",
                        "key_factors",
                        "market_outlook",
                        "major_themes",
                        "news_volume",
                        "sentiment_distribution",
                        "sources_searched",
                        "articles"
                    ],
                    "additionalProperties": False
                }

                prompt = (
                    f"Search for recent financial news about {symbol} ({company_name if company_name else 'stock'}) and provide a sentiment analysis. "
                    f"{'Focus on Japanese market conditions, yen movements, and BOJ policy for this Japanese stock.' if is_japanese_stock else ''}"
                    "Return only JSON that conforms to the specified schema; do not include any extra text. "
                    "If no articles are available, return an empty array for `articles`."
                )

                response = client.responses.create(
                    model=self.model,
                    tools=[{"type": "web_search"}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "news_analysis",
                            "schema": json_schema
                        }
                    },
                    input=prompt
                )

                response_content = response.output_text if hasattr(response, 'output_text') else str(response)

            else:
                # Fallback to regular chat completions for other models
                # Prepare request parameters
                request_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a financial analyst with comprehensive knowledge of global markets, recent events, and company performance. Generate realistic news sentiment analysis based on your knowledge of market conditions, company fundamentals, and industry trends. For Japanese stocks, incorporate Japanese market perspectives, yen movements, and domestic economic factors. ALWAYS respond with valid JSON format."
                        },
                        {
                            "role": "user",
                            "content": search_prompt
                        }
                    ],
                    "max_completion_tokens": 2000
                }

                # Only add temperature for models that support it (not GPT-5 models)
                if not self.model.startswith("gpt-5"):
                    request_params["temperature"] = 0.7

                response = client.chat.completions.create(**request_params)

                response_content = response.choices[0].message.content

            # Parse the response with better error handling
            logger.info(f"Received OpenAI response for {symbol}, length: {len(response_content) if response_content else 0} chars")

            # Check for empty response
            if not response_content or not response_content.strip():
                logger.warning(f"Empty content in OpenAI response for {symbol}, using default analysis")
                # Return default analysis for empty response
                return self._get_default_analysis(symbol)

            # Handle GPT-5 natural language response vs structured JSON response
            if self.model.startswith("gpt-5"):
                # GPT-5 with web search returns natural language, convert to structured data
                news_data = self._parse_gpt5_response(response_content, symbol, is_japanese_stock, company_name)
                logger.info(f"Processed GPT-5 response for {symbol}")
            else:
                # Try to parse as JSON for other models
                try:
                    news_data = json.loads(response_content)
                    logger.info(f"Successfully parsed JSON for {symbol}. Articles count: {len(news_data.get('articles', []))}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response for {symbol}: {e}")
                    logger.info(f"Full response content: {response_content}")

                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        try:
                            news_data = json.loads(json_match.group())
                            logger.info(f"Extracted JSON from response for {symbol}. Articles: {len(news_data.get('articles', []))}")
                        except json.JSONDecodeError as e2:
                            logger.error(f"Failed to parse extracted JSON for {symbol}: {e2}")
                            # Return a default structure if parsing fails
                            news_data = {
                                "sentiment_score": 0,
                                "overall_sentiment": "neutral",
                                "confidence": 0.5,
                                "articles": [],
                                "key_factors": ["Unable to parse news data"],
                                "market_outlook": "News data parsing failed",
                                "major_themes": [],
                                "news_volume": "low",
                                "sentiment_distribution": {
                                    "positive": 0,
                                    "negative": 0,
                                    "neutral": 0
                                },
                                "sources_searched": []
                            }
                    else:
                        logger.warning(f"No JSON found in response for {symbol}")
                        # Return a default structure if parsing fails
                        news_data = {
                            "sentiment_score": 0,
                            "overall_sentiment": "neutral",
                            "confidence": 0.5,
                            "articles": [],
                            "key_factors": ["Unable to parse news data"],
                            "market_outlook": "News data parsing failed",
                            "major_themes": [],
                            "news_volume": "low",
                            "sentiment_distribution": {
                                "positive": 0,
                                "negative": 0,
                                "neutral": 0
                            },
                            "sources_searched": []
                        }

            # Process and enhance the data
            return self._process_news_data(news_data, symbol)

        except Exception as e:
            logger.warning(f"OpenAI API error for {symbol}: {e}, using default analysis")
            # Return default analysis instead of error for better user experience
            return self._get_default_analysis(symbol)

    def _process_news_data(self, news_data: Dict, symbol: str) -> Dict:
        """
        Process and enhance the news data from OpenAI.

        Args:
            news_data: Raw news data from OpenAI
            symbol: Stock symbol

        Returns:
            Processed news analysis
        """
        logger.info(f"Processing news data for {symbol}")
        logger.debug(f"News data keys: {news_data.keys()}")

        try:
            # Extract articles and ensure they have proper formatting
            articles = news_data.get('articles', [])
            logger.info(f"Found {len(articles)} articles for {symbol}")

            # Format articles for display, ignoring incomplete entries
            formatted_articles: List[Dict[str, Any]] = []
            for article in articles[:10]:
                if not isinstance(article, dict):
                    continue
                title = article.get('title')
                link = article.get('link') or article.get('url')
                if not title or not link:
                    continue
                sentiment_payload = article.get('sentiment') or {}
                sentiment_label = sentiment_payload.get('sentiment', 'neutral')
                score = sentiment_payload.get('score', 0)
                reasoning = sentiment_payload.get('reasoning', article.get('summary', ''))
                published_at = article.get('publishedAt') or article.get('date') or datetime.now().strftime('%Y-%m-%d')
                impact_score = article.get('impact_score', score)
                formatted_articles.append({
                    'title': title,
                    'source': article.get('source', 'Financial News'),
                    'publishedAt': published_at,
                    'summary': article.get('summary', ''),
                    'link': link,
                    'sentiment': {
                        'sentiment': sentiment_label,
                        'score': impact_score,
                        'reasoning': reasoning
                    },
                    'relevance': article.get('relevance', 'medium')
                })

            # Calculate sentiment distribution
            sentiment_dist = news_data.get('sentiment_distribution', {})
            if not sentiment_dist and formatted_articles:
                positive = sum(1 for a in formatted_articles if a['sentiment']['sentiment'] == 'positive')
                negative = sum(1 for a in formatted_articles if a['sentiment']['sentiment'] == 'negative')
                neutral = len(formatted_articles) - positive - negative
                sentiment_dist = {
                    'positive': positive,
                    'negative': negative,
                    'neutral': neutral
                }

            # Calculate impact on prediction
            sentiment_score = news_data.get('sentiment_score', 0)
            confidence = news_data.get('confidence', 0.7)

            impact = self._calculate_impact_on_prediction(sentiment_score, confidence)

            # Determine sources searched based on symbol
            is_japanese_stock = symbol.endswith('.T')
            sources_searched = news_data.get('sources_searched')
            if not sources_searched:
                sources_searched = self.japanese_sources[:8] if is_japanese_stock else self.news_sources[:8]

            return {
                'status': 'success',
                'articles_count': len(formatted_articles),
                'sentiment': news_data.get('overall_sentiment', 'neutral'),
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'sentiment_distribution': sentiment_dist,
                'articles': formatted_articles,
                'key_factors': news_data.get('key_factors', []),
                'market_outlook': news_data.get('market_outlook', ''),
                'major_themes': news_data.get('major_themes', []),
                'news_volume': news_data.get('news_volume', 'medium'),
                'source': f'OpenAI Multi-Source Analysis ({self.model})',
                'sources_searched': sources_searched,
                'impact_on_prediction': impact
            }

        except Exception as e:
            logger.error(f"Error processing news data: {e}")
            return {
                'status': 'error',
                'message': f'Failed to process news data: {str(e)}'
            }

    def _get_default_analysis(self, symbol: str) -> Dict:
        """Return a neutral response when no reliable news data is available."""
        return {
            'status': 'error',
            'message': 'News analysis unavailable - API error or empty response',
            'articles_count': 0,
            'sentiment': 'neutral',
            'sentiment_score': 0,
            'confidence': 0,
            'articles': [],
            'key_factors': [],
            'market_outlook': '',
            'major_themes': [],
            'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
            'source': 'No Data Available',
            'sources_searched': [],
            'news_volume': 'low',
            'impact_on_prediction': {
                'price_adjustment': 0,
                'confidence_adjustment': 0,
                'suggested_action': 'News analysis required for sentiment-based recommendations'
            }
        }

    def _parse_gpt5_response(self, response_content: str, symbol: str,
                              is_japanese_stock: bool, company_name: str = None) -> Dict:
        """Parse GPT-5 response that should contain JSON describing articles."""
        import re

        logger.info(f"Parsing GPT-5 response for {symbol}")
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if not json_match:
            raise ValueError('GPT-5 response did not include JSON content')

        data = json.loads(json_match.group())

        raw_articles = data.get('articles', []) or []
        articles: List[Dict[str, Any]] = []

        for raw in raw_articles:
            if not isinstance(raw, dict):
                continue

            title = (raw.get('title') or '').strip()
            link = raw.get('url') or raw.get('link')
            if not title or not link:
                continue

            source = (raw.get('source') or '').strip() or 'Financial News'
            published = raw.get('date') or raw.get('publishedAt') or datetime.now().strftime('%Y-%m-%d')
            summary = (raw.get('summary') or '').strip()

            sentiment_field = raw.get('sentiment')
            sentiment_label: str
            score = 0.0
            reasoning = summary

            if isinstance(sentiment_field, dict):
                sentiment_label = str(sentiment_field.get('sentiment', 'neutral')).lower()
                try:
                    score = float(sentiment_field.get('score', raw.get('impact_score', raw.get('sentiment_score', 0)) or 0) or 0)
                except (TypeError, ValueError):
                    score = 0.0
                reasoning = str(sentiment_field.get('reasoning', summary) or summary)
            else:
                sentiment_label = str(sentiment_field or raw.get('sentiment_label', 'neutral')).lower()
                try:
                    score = float(raw.get('impact_score', raw.get('sentiment_score', 0)) or 0)
                except (TypeError, ValueError):
                    score = 0.0

            if sentiment_label not in {'positive', 'negative', 'neutral'}:
                sentiment_label = 'neutral'
            reasoning = reasoning or summary

            relevance = raw.get('relevance', 'medium')

            rounded_score = round(score, 2)

            articles.append({
                'title': title,
                'source': source,
                'date': published,
                'summary': summary,
                'url': link,
                'sentiment': {
                    'sentiment': sentiment_label,
                    'score': rounded_score,
                    'reasoning': reasoning
                },
                'impact_score': rounded_score,
                'relevance': relevance
            })

        sentiment_score = float(data.get('sentiment_score', 0) or 0)
        overall_sentiment = str(data.get('overall_sentiment', 'neutral')).lower()
        confidence = float(data.get('confidence', 0) or 0)

        key_factors = [str(item).strip() for item in data.get('key_factors', []) if str(item).strip()]
        major_themes = [str(item).strip() for item in data.get('major_themes', []) if str(item).strip()]
        market_outlook = str(data.get('market_outlook', '')).strip()

        sentiment_distribution = data.get('sentiment_distribution', {}) or {}
        if not sentiment_distribution and articles:
            positive = sum(1 for a in articles if a['sentiment']['sentiment'] == 'positive')
            negative = sum(1 for a in articles if a['sentiment']['sentiment'] == 'negative')
            neutral = len(articles) - positive - negative
            sentiment_distribution = {
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            }

        news_volume = str(data.get('news_volume', '')).strip().lower()
        if news_volume not in {'high', 'medium', 'low'}:
            article_count = len(articles)
            if article_count >= 6:
                news_volume = 'high'
            elif article_count >= 3:
                news_volume = 'medium'
            else:
                news_volume = 'low'

        sources_searched = data.get('sources_searched')
        if sources_searched:
            sources_searched = [str(src).strip() for src in sources_searched if str(src).strip()]

        if not sources_searched:
            derived_sources = [a.get('source') for a in articles if a.get('source')]
            dedup_sources = []
            for src in derived_sources:
                if src not in dedup_sources:
                    dedup_sources.append(src)

            default_pool = self.japanese_sources if is_japanese_stock else self.news_sources
            sources_searched = (dedup_sources or default_pool)[:8]

        return {
            'sentiment_score': sentiment_score,
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'articles': articles,
            'key_factors': key_factors,
            'market_outlook': market_outlook,
            'major_themes': major_themes,
            'news_volume': news_volume,
            'sentiment_distribution': sentiment_distribution,
            'sources_searched': sources_searched,
            'impact_on_prediction': self._calculate_impact_on_prediction(sentiment_score, confidence)
        }


    def _calculate_impact_on_prediction(self, sentiment_score: float, confidence: float) -> Dict:
        """
        Calculate how news sentiment should impact price prediction.

        Args:
            sentiment_score: Sentiment score from -1 to 1
            confidence: Confidence level from 0 to 1

        Returns:
            Impact dictionary
        """
        # Base impact on sentiment score and confidence
        price_adjustment = sentiment_score * confidence * 0.04  # Max ~4% adjustment
        confidence_adjustment = confidence * 0.1  # Max 10% confidence boost

        # Determine suggested action
        if sentiment_score > 0.3:
            if confidence > 0.7:
                suggested_action = "Strong Bullish - positive news momentum across multiple sources"
            else:
                suggested_action = "Moderately Bullish - positive sentiment with some uncertainty"
        elif sentiment_score < -0.3:
            if confidence > 0.7:
                suggested_action = "Strong Bearish - negative news consensus from multiple sources"
            else:
                suggested_action = "Moderately Bearish - negative sentiment with mixed signals"
        else:
            suggested_action = "Neutral - mixed or limited news impact"

        return {
            'price_adjustment': price_adjustment,
            'confidence_adjustment': confidence_adjustment,
            'suggested_action': suggested_action
        }

    def get_fallback_analysis(self, symbol: str, company_name: str = None) -> Dict:
        """
        Provide a basic market analysis when comprehensive search is not available.

        Args:
            symbol: Stock symbol
            company_name: Company name

        Returns:
            Basic market analysis
        """
        # Yahoo Finance is giving 401 errors, so we can't use it for fallback
        # Return a neutral analysis instead
        return {
            'status': 'no_data',
            'message': 'News analysis requires OpenAI API key',
            'sentiment': 'neutral',
            'sentiment_score': 0,
            'confidence': 0,
            'articles': [],
            'sentiment_distribution': {
                'positive': 0,
                'negative': 0,
                'neutral': 1
            },
            'key_factors': ['API key required for news analysis'],
            'source': 'No Data Available',
            'news_volume': 'low',
            'sources_searched': [],
            'impact_on_prediction': {
                'price_adjustment': 0,
                'confidence_adjustment': 0,
                'suggested_action': 'Configure OpenAI API for news analysis'
            }
        }


class NewsAnalyzer(EnhancedNewsAnalyzer):
    """Backwards-compatible news analyzer facade used by tests and agents."""

    def __init__(self) -> None:
        super().__init__()

    def fetch_news_yahoo(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news articles from Yahoo Finance for compatibility tests."""
        try:
            ticker = yf.Ticker(symbol)
            raw_news = getattr(ticker, "news", []) or []
        except Exception as exc:  # pragma: no cover - defensive path
            logger.error(f"Error fetching Yahoo Finance news for {symbol}: {exc}")
            return []

        articles: List[Dict[str, Any]] = []
        for item in raw_news:
            if not isinstance(item, dict):
                continue

            published_ts = item.get('providerPublishTime')
            published_at = None
            if isinstance(published_ts, (int, float)):
                try:
                    published_at = datetime.fromtimestamp(published_ts).isoformat()
                except (OSError, OverflowError, ValueError):
                    published_at = None

            articles.append({
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'link': item.get('link'),
                'source': item.get('publisher', 'Yahoo Finance'),
                'publishedAt': published_at,
            })

        return articles

    def analyze_sentiment(self, articles: List[Dict[str, Any]], symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment using the configured OpenAI model."""
        if not self.openai_api_key:
            logger.warning("OpenAI API key is not configured; skipping sentiment analysis")
            return None

        if not articles:
            logger.info("No articles supplied for sentiment analysis")
            return None

        try:
            if OpenAI is None:
                logger.warning("OpenAI client not available; ensure openai package is installed")
                return None

            client = OpenAI(api_key=self.openai_api_key)
            prompt_payload = {
                'symbol': symbol,
                'articles': articles,
            }

            response = client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial news analyst who responds with strict JSON.'
                    },
                    {
                        'role': 'user',
                        'content': json.dumps(prompt_payload)
                    }
                ],
            )

            content = response.choices[0].message.content if response.choices else '{}'
            parsed = json.loads(content)

            overall_sentiment = parsed.get('overall_sentiment', 'neutral')
            sentiment_score = float(parsed.get('sentiment_score', 0) or 0)
            confidence = float(parsed.get('confidence', 0) or 0)
            key_factors = parsed.get('key_factors', []) or []
            articles_analyzed = parsed.get('articles_analyzed', len(articles))
            impact = self._calculate_impact_on_prediction(sentiment_score, confidence)

            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'key_factors': key_factors,
                'articles_analyzed': articles_analyzed,
                'impact_on_prediction': impact,
                'raw': parsed,
            }

        except Exception as exc:  # pragma: no cover - network/parsing failures
            logger.error(f"Failed to analyze sentiment for {symbol}: {exc}")
            return None

    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Public API expected by existing tests and agents."""
        articles = self.fetch_news_yahoo(symbol)

        if not articles:
            return {
                'status': 'no_data',
                'message': f'No news articles found for {symbol}.',
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'articles': [],
                'articles_count': 0,
            }

        if not self.openai_api_key:
            impact = self._calculate_impact_on_prediction(0.0, 0.0)
            return {
                'status': 'success',
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'articles': articles,
                'articles_count': len(articles),
                'source': 'Yahoo Finance',
                'impact_on_prediction': impact,
            }

        analysis = self.analyze_sentiment(articles, symbol)
        if not analysis:
            impact = self._calculate_impact_on_prediction(0.0, 0.0)
            return {
                'status': 'error',
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'articles': articles,
                'articles_count': len(articles),
                'impact_on_prediction': impact,
                'message': 'Sentiment analysis failed',
            }

        impact = analysis.get('impact_on_prediction') or self._calculate_impact_on_prediction(
            analysis.get('sentiment_score', 0.0),
            analysis.get('confidence', 0.0),
        )

        return {
            'status': 'success',
            'sentiment': analysis['overall_sentiment'],
            'sentiment_score': analysis['sentiment_score'],
            'confidence': analysis['confidence'],
            'key_factors': analysis.get('key_factors', []),
            'articles_count': analysis.get('articles_analyzed', len(articles)),
            'articles': articles,
            'impact_on_prediction': impact,
        }
