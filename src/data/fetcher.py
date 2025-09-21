"""Market data fetching module for US and Japanese stocks."""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

import yfinance as yf
import pandas as pd
import numpy as np
from aiohttp import ClientSession

from src.core.database import db_manager, StockPrice, StockInfo
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Fetch market data from various sources."""

    def __init__(self):
        """Initialize market data fetcher."""
        self.us_symbols = []
        self.jp_symbols = []
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (optional)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)

            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch historical data
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Add symbol column
            df['Symbol'] = symbol

            # Determine market based on symbol pattern or exchange
            market = self._determine_market(symbol)
            df['Market'] = market

            logger.info(f"Fetched {len(df)} rows for {symbol} from {start_date} to {end_date}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _determine_market(self, symbol: str) -> str:
        """Determine if stock is US or Japanese market."""
        # Japanese stocks often have .T suffix on Yahoo Finance
        if symbol.endswith('.T') or symbol.endswith('.JP'):
            return 'JP'
        # Check if it's a number (common for Japanese stocks)
        elif symbol.replace('.', '').isdigit():
            return 'JP'
        else:
            return 'US'

    async def fetch_batch_async(self, symbols: List[str], start_date: str, end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols asynchronously.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date (optional)

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        loop = asyncio.get_event_loop()
        tasks = []

        for symbol in symbols:
            task = loop.run_in_executor(
                None,
                self.fetch_stock_data,
                symbol,
                start_date,
                end_date
            )
            tasks.append((symbol, task))

        results = {}
        for symbol, task in tasks:
            try:
                df = await task
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Error in batch fetch for {symbol}: {e}")

        return results

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get stock information/metadata.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with stock info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'exchange': info.get('exchange', ''),
                'country': info.get('country', ''),
                'currency': info.get('currency', 'USD')
            }
        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {e}")
            return {}

    def save_to_database(self, df: pd.DataFrame, symbol: str):
        """
        Save stock data to database.

        Args:
            df: DataFrame with stock data
            symbol: Stock symbol
        """
        if df.empty:
            logger.info(f"No records to save for {symbol}")
            return

        try:
            with db_manager.get_session() as session:
                indexed_rows = []
                for idx in df.index:
                    python_ts = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
                    indexed_rows.append(python_ts)

                existing_rows = session.query(StockPrice.timestamp).filter(
                    StockPrice.symbol == symbol,
                    StockPrice.timestamp.in_(indexed_rows)
                ).all()

                existing_timestamps = {row[0] for row in existing_rows}

                new_records = []
                for idx, row in zip(indexed_rows, df.itertuples(index=False)):
                    if idx in existing_timestamps:
                        continue

                    new_records.append(StockPrice(
                        symbol=symbol,
                        timestamp=idx,
                        open=getattr(row, 'Open', None),
                        high=getattr(row, 'High', None),
                        low=getattr(row, 'Low', None),
                        close=getattr(row, 'Close', None),
                        volume=getattr(row, 'Volume', None),
                        adjusted_close=getattr(row, 'Close', None),
                        market=getattr(row, 'Market', 'US') if hasattr(row, 'Market') else 'US'
                    ))

                if new_records:
                    session.bulk_save_objects(new_records)
                    logger.info(f"Saved {len(new_records)} records for {symbol} to database")
                else:
                    logger.info(f"All {len(df)} records for {symbol} already exist")

        except Exception as e:
            logger.error(f"Error saving data to database: {e}")

    def get_popular_stocks(self) -> Dict[str, List[str]]:
        """Get lists of popular stocks for US and Japanese markets."""
        us_stocks = [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'CVS',
            # Consumer
            'WMT', 'HD', 'DIS', 'NKE', 'MCD',
            # Index ETFs
            'SPY', 'QQQ', 'DIA', 'IWM'
        ]

        jp_stocks = [
            # Major Japanese companies (Tokyo Stock Exchange)
            '7203.T',  # Toyota
            '6758.T',  # Sony
            '9984.T',  # SoftBank
            '6861.T',  # Keyence
            '8306.T',  # Mitsubishi UFJ
            '9432.T',  # NTT
            '7267.T',  # Honda
            '6902.T',  # Denso
            '6501.T',  # Hitachi
            '7974.T',  # Nintendo
            '4063.T',  # Shin-Etsu Chemical
            '9433.T',  # KDDI
            '6098.T',  # Recruit Holdings
            '8035.T',  # Tokyo Electron
            '4502.T',  # Takeda Pharmaceutical
        ]

        return {
            'US': us_stocks,
            'JP': jp_stocks
        }

    async def update_all_stocks(self, days_back: int = 30):
        """
        Update all popular stocks with recent data.

        Args:
            days_back: Number of days of historical data to fetch
        """
        stocks = self.get_popular_stocks()
        all_symbols = stocks['US'] + stocks['JP']

        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Updating {len(all_symbols)} stocks from {start_date} to {end_date}")

        # Fetch data in batches
        batch_size = 10
        for i in range(0, len(all_symbols), batch_size):
            batch = all_symbols[i:i+batch_size]
            results = await self.fetch_batch_async(batch, start_date, end_date)

            # Save to database
            for symbol, df in results.items():
                if not df.empty:
                    self.save_to_database(df, symbol)

                    # Also save stock info
                    info = self.get_stock_info(symbol)
                    if info:
                        self._save_stock_info(info)

            # Rate limiting
            await asyncio.sleep(1)

        logger.info("Stock update completed")

    def _save_stock_info(self, info: Dict[str, Any]):
        """Save stock information to database."""
        try:
            with db_manager.get_session() as session:
                existing = session.query(StockInfo).filter_by(
                    symbol=info['symbol']
                ).first()

                if existing:
                    # Update existing record
                    for key, value in info.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.last_updated = datetime.utcnow()
                else:
                    # Create new record
                    stock_info = StockInfo(**info)
                    session.add(stock_info)

                session.commit()

        except Exception as e:
            logger.error(f"Error saving stock info: {e}")

# Example usage
