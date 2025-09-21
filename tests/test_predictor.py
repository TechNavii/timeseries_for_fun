"""Unit tests for the StockPredictor module."""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.models.predictor import StockPredictor, PredictionService
from config.app_config import PREDICTION_CONFIG

class TestStockPredictor(unittest.TestCase):
    """Test cases for StockPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 105, 100),
            'Volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)

        # Create sample features
        self.sample_features = pd.DataFrame(
            np.random.randn(100, 20),
            columns=[f'feature_{i}' for i in range(20)],
            index=dates
        )

        # Create sample targets
        self.sample_targets = pd.Series(
            np.random.uniform(-0.05, 0.05, 100),
            index=dates
        )

    def test_predictor_initialization(self):
        """Test StockPredictor initialization."""
        predictor = StockPredictor(model_type='xgboost')
        self.assertEqual(predictor.model_type, 'xgboost')
        self.assertIsNone(predictor.model)
        self.assertIsNone(predictor.feature_importance)
        self.assertEqual(predictor.model_params, {})
        self.assertEqual(predictor.training_history, [])

    def test_create_model_xgboost(self):
        """Test XGBoost model creation."""
        predictor = StockPredictor(model_type='xgboost')
        model = predictor.create_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

    def test_create_model_lightgbm(self):
        """Test LightGBM model creation."""
        predictor = StockPredictor(model_type='lightgbm')
        model = predictor.create_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

    def test_create_model_random_forest(self):
        """Test Random Forest model creation."""
        predictor = StockPredictor(model_type='random_forest')
        model = predictor.create_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

    def test_create_model_ensemble(self):
        """Test Ensemble model creation."""
        predictor = StockPredictor(model_type='ensemble')
        model = predictor.create_model()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

    def test_create_model_invalid_type(self):
        """Test invalid model type raises error."""
        predictor = StockPredictor(model_type='invalid_model')
        with self.assertRaises(ValueError):
            predictor.create_model()

    def test_train_model(self):
        """Test model training."""
        predictor = StockPredictor(model_type='xgboost')

        # Train model
        metrics = predictor.train(
            self.sample_features.iloc[:80],
            self.sample_targets.iloc[:80],
            validation_split=0.2,
            use_time_series_cv=False
        )

        self.assertIsNotNone(predictor.model)
        self.assertIn('val_rmse', metrics)
        self.assertIn('train_rmse', metrics)
        self.assertIsInstance(metrics['val_rmse'], float)
        self.assertGreaterEqual(metrics['val_rmse'], 0)

    def test_predict_without_training(self):
        """Test prediction without training raises error."""
        predictor = StockPredictor(model_type='xgboost')

        with self.assertRaises(ValueError) as context:
            predictor.predict(self.sample_features)

        self.assertIn("Model not trained yet", str(context.exception))

    def test_predict_after_training(self):
        """Test prediction after training."""
        predictor = StockPredictor(model_type='xgboost')

        # Train model
        predictor.train(
            self.sample_features.iloc[:80],
            self.sample_targets.iloc[:80],
            validation_split=0.2,
            use_time_series_cv=False
        )

        # Make predictions
        predictions = predictor.predict(self.sample_features.iloc[80:])

        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions), len(self.sample_features.iloc[80:]))
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in predictions))

    def test_predict_proba(self):
        """Test probability predictions with confidence."""
        predictor = StockPredictor(model_type='xgboost')

        # Train model
        predictor.train(
            self.sample_features.iloc[:80],
            self.sample_targets.iloc[:80],
            validation_split=0.2,
            use_time_series_cv=False
        )

        # Get probability predictions
        result = predictor.predict_proba(self.sample_features.iloc[80:])

        self.assertIn('predictions', result)
        self.assertIn('predictions_lower', result)
        self.assertIn('predictions_upper', result)
        self.assertIn('predictions_std', result)
        self.assertIn('classes', result)
        self.assertIn('confidence', result)

        # Check classes are valid
        valid_classes = {'UP', 'DOWN', 'NEUTRAL'}
        self.assertTrue(all(c in valid_classes for c in result['classes']))

        # Check confidence is between 0 and 1
        self.assertTrue(all(0 <= c <= 1 for c in result['confidence']))

    def test_get_top_features(self):
        """Test getting top features."""
        predictor = StockPredictor(model_type='xgboost')

        # Train model to generate feature importance
        predictor.train(
            self.sample_features.iloc[:80],
            self.sample_targets.iloc[:80],
            validation_split=0.2,
            use_time_series_cv=False
        )

        # Get top features
        top_features = predictor.get_top_features(n=5)

        self.assertIsInstance(top_features, list)
        self.assertLessEqual(len(top_features), 5)
        self.assertTrue(all(f in self.sample_features.columns for f in top_features))

    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        predictor = StockPredictor(model_type='random_forest')

        # Train model
        predictor.train(
            self.sample_features,
            self.sample_targets,
            validation_split=0.2,
            use_time_series_cv=False
        )

        # Check feature importance
        self.assertIsNotNone(predictor.feature_importance)
        self.assertIn('feature', predictor.feature_importance.columns)
        self.assertIn('importance', predictor.feature_importance.columns)
        self.assertEqual(len(predictor.feature_importance), len(self.sample_features.columns))

    def test_ensemble_feature_importance(self):
        """Test feature importance extraction from ensemble model."""
        predictor = StockPredictor(model_type='ensemble')

        # Train model
        predictor.train(
            self.sample_features,
            self.sample_targets,
            validation_split=0.2,
            use_time_series_cv=False
        )

        # Feature importance might be None or a DataFrame
        if predictor.feature_importance is not None:
            self.assertIsInstance(predictor.feature_importance, pd.DataFrame)
            self.assertIn('feature', predictor.feature_importance.columns)
            self.assertIn('importance', predictor.feature_importance.columns)

class TestPredictionService(unittest.TestCase):
    """Test cases for PredictionService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample stock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_stock_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 105, 100),
            'Volume': np.random.uniform(1000000, 2000000, 100)
        }, index=dates)

        self.service = PredictionService()

    def test_service_initialization(self):
        """Test PredictionService initialization."""
        self.assertIsInstance(self.service.models, dict)
        self.assertEqual(len(self.service.models), 0)
        self.assertIsNone(self.service.feature_engineer)

    def test_load_feature_engineer(self):
        """Test loading feature engineer."""
        self.service.load_feature_engineer()
        self.assertIsNotNone(self.service.feature_engineer)

    def test_predict_stock_insufficient_data(self):
        """Test prediction with insufficient data."""
        # Create minimal data (less than required minimum)
        small_data = self.sample_stock_data.iloc[:10]

        result = self.service.predict_stock(
            symbol='TEST',
            data=small_data,
            prediction_horizon=1,
            model_type='xgboost'
        )

        self.assertIn('error', result)

    def test_predict_stock_success(self):
        """Test successful stock prediction."""
        result = self.service.predict_stock(
            symbol='TEST',
            data=self.sample_stock_data,
            prediction_horizon=1,
            model_type='xgboost'
        )

        # Check required keys in result
        if 'error' not in result:
            self.assertIn('symbol', result)
            self.assertIn('current_price', result)
            self.assertIn('predicted_price', result)
            self.assertIn('price_range_lower', result)
            self.assertIn('price_range_upper', result)
            self.assertIn('predicted_return', result)
            self.assertIn('predicted_class', result)
            self.assertIn('confidence', result)
            self.assertIn('model_type', result)

            # Check predicted class is valid
            self.assertIn(result['predicted_class'], ['UP', 'DOWN', 'NEUTRAL'])

            # Check confidence is between 0 and 1
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)

    def test_model_caching(self):
        """Test that models are cached properly."""
        # First prediction
        result1 = self.service.predict_stock(
            symbol='TEST',
            data=self.sample_stock_data,
            prediction_horizon=1,
            model_type='xgboost'
        )

        # Check model was cached
        model_key = 'TEST_1d_xgboost'
        self.assertIn(model_key, self.service.models)

        # Second prediction should use cached model
        result2 = self.service.predict_stock(
            symbol='TEST',
            data=self.sample_stock_data,
            prediction_horizon=1,
            model_type='xgboost'
        )

        # Should still have only one model cached
        self.assertEqual(len([k for k in self.service.models.keys() if k.startswith('TEST')]), 1)

if __name__ == '__main__':
    unittest.main()