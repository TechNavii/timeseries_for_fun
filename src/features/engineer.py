"""Feature engineering module for stock prediction."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engineer features from raw stock data."""

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with technical indicators
        """
        df = df.copy()

        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # ATR (Average True Range)
        df['ATR'] = self._calculate_atr(df)

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = self._calculate_obv(df)

        # Stochastic Oscillator
        df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)

        return df

    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price features
        """
        df = df.copy()

        # Returns
        df['Return_1'] = df['Close'].pct_change(1)
        df['Return_5'] = df['Close'].pct_change(5)
        df['Return_10'] = df['Close'].pct_change(10)
        df['Return_20'] = df['Close'].pct_change(20)

        # Log returns
        df['Log_Return_1'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Return_5'] = np.log(df['Close'] / df['Close'].shift(5))

        # Volatility
        df['Volatility_5'] = df['Return_1'].rolling(window=5).std()
        df['Volatility_20'] = df['Return_1'].rolling(window=20).std()
        df['Volatility_60'] = df['Return_1'].rolling(window=60).std()

        # Price ratios
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Price position
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

        # Gaps
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = df['Gap'] / df['Close'].shift(1)

        return df

    def calculate_market_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market microstructure features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with microstructure features
        """
        df = df.copy()

        # Spread approximation
        df['Spread'] = df['High'] - df['Low']
        df['Spread_Pct'] = df['Spread'] / df['Close']

        # Liquidity proxy
        df['Amihud_Illiquidity'] = np.abs(df['Return_1']) / df['Volume']

        # Volume features
        df['Dollar_Volume'] = df['Close'] * df['Volume']
        df['Volume_Volatility'] = df['Volume'].rolling(window=20).std()

        # Intraday features
        df['Intraday_Volatility'] = (df['High'] - df['Low']) / df['Open']
        df['Overnight_Return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        return df

    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.

        Args:
            df: DataFrame with features
            columns: List of column names to lag
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f'{col}_Lag_{lag}'] = df[col].shift(lag)

        return df

    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window statistics.

        Args:
            df: DataFrame with features
            columns: List of column names
            windows: List of window sizes

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        for col in columns:
            for window in windows:
                df[f'{col}_Rolling_Mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_Rolling_Std_{window}'] = df[col].rolling(window=window).std()
                df[f'{col}_Rolling_Min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_Rolling_Max_{window}'] = df[col].rolling(window=window).max()

        return df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features from raw OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all engineered features
        """
        # Calculate all feature groups
        df = self.calculate_technical_indicators(df)
        df = self.calculate_price_features(df)
        df = self.calculate_market_microstructure(df)

        # Create lag features for important columns
        lag_columns = ['Close', 'Volume', 'RSI', 'MACD']
        df = self.create_lag_features(df, lag_columns, [1, 2, 3, 5, 10])

        # Create rolling features
        rolling_columns = ['Return_1', 'Volume']
        df = self.create_rolling_features(df, rolling_columns, [5, 10, 20])

        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Market']]

        # Drop NaN rows
        df = df.dropna()

        logger.info(f"Engineered {len(self.feature_names)} features")
        return df

    def prepare_ml_data(self, df: pd.DataFrame, target_days: int = 1) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for ML training.

        Args:
            df: DataFrame with features
            target_days: Number of days ahead to predict

        Returns:
            Tuple of (features, regression_target, classification_target)
        """
        df = df.copy()

        # Create target variable (future return)
        df['Target'] = df['Close'].shift(-target_days) / df['Close'] - 1

        # Create classification target
        df['Target_Class'] = pd.cut(df['Target'], bins=[-np.inf, -0.01, 0.01, np.inf], labels=['DOWN', 'NEUTRAL', 'UP'])

        # Drop NaN targets
        df = df.dropna(subset=['Target'])

        # Select features
        feature_cols = [col for col in self.feature_names if col in df.columns]
        X = df[feature_cols]
        y = df['Target']
        y_class = df['Target_Class']

        return X, y, y_class

    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return obv

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()

        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=smooth).mean()

        return k_percent, d_percent

    def normalize_features(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize features.

        Args:
            X: Feature DataFrame
            method: Normalization method ('standard', 'minmax', 'robust')

        Returns:
            Normalized DataFrame
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        return X_scaled
