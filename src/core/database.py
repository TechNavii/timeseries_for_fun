"""Database models and connection management."""
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class StockPrice(Base):
    """Historical stock price data."""
    __tablename__ = 'stock_prices'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)
    market = Column(String(10))  # US or JP

    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class StockInfo(Base):
    """Stock metadata and information."""
    __tablename__ = 'stock_info'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    exchange = Column(String(20))
    country = Column(String(50))
    currency = Column(String(10))
    active = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database connection and session management."""

    def __init__(self, db_url: Optional[str] = None):
        """Initialize database manager."""
        if not db_url:
            db_url = os.getenv('DATABASE_URL', 'sqlite:///stock_prediction.db')

        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            echo=False
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Initialize Redis for caching
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        print("Database tables created successfully")

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
        print("Database tables dropped")

    @contextmanager
    def get_session(self) -> Session:
        """Get database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False

# Initialize global database manager
db_manager = DatabaseManager()
