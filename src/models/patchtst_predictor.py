"""PatchTST-based time series prediction for stocks."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# Critical environment settings for macOS stability
if platform.system() == "Darwin":
    # Disable MPS and force CPU usage
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Disable parallelism to avoid threading issues
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    # Disable torch compile
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
else:
    # Enable MPS fallback for other systems
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

logger = logging.getLogger(__name__)

class PatchTSTPredictor:
    """Stock price predictor using PatchTST transformer model."""

    def __init__(self,
                 patch_len: int = 16,
                 stride: int = 8,
                 hidden_size: int = 256,
                 n_heads: int = 8):
        """
        Initialize PatchTST predictor.

        Args:
            patch_len: Length of each patch
            stride: Stride for creating patches
            hidden_size: Hidden dimension size
            n_heads: Number of attention heads
        """
        self.patch_len = patch_len
        self.stride = stride
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.model = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            # Environment variables already set at module level
            if platform.system() == "Darwin":
                logger.info("macOS detected - using CPU with optimized settings for PatchTST")

            # Try NeuralForecast first (easiest integration)
            try:
                from neuralforecast import NeuralForecast
                from neuralforecast.models import PatchTST
                self.use_neuralforecast = True
                self._initialized = True
                logger.info("PatchTST will use NeuralForecast implementation")
            except ImportError:
                pass

            # Try Hugging Face transformers fallback
            if not self._initialized:
                try:
                    from transformers import PatchTSTForPrediction, PatchTSTConfig
                    self.use_transformers = True
                    self._initialized = True
                    logger.info("PatchTST will use Hugging Face Transformers implementation")
                except ImportError:
                    raise ImportError(
                        "Neither NeuralForecast nor Transformers with PatchTST is installed. "
                        "Install with: pip install neuralforecast or pip install transformers"
                    )

        except Exception as e:
            logger.error(f"Failed to initialize PatchTST: {e}")
            raise

    def predict(self,
                data: pd.DataFrame,
                prediction_horizon: int = 5,
                lookback_window: int = 120) -> Dict:
        """
        Make stock price predictions using PatchTST.

        Args:
            data: Historical stock data with at least 'Close' column
            prediction_horizon: Number of days to predict ahead
            lookback_window: Number of historical days to use

        Returns:
            Dictionary with predictions and metrics
        """
        self._lazy_init()

        try:
            if 'Close' not in data.columns:
                raise ValueError("Data must have 'Close' column")

            if hasattr(self, 'use_neuralforecast') and self.use_neuralforecast:
                return self._predict_with_neuralforecast(data, prediction_horizon, lookback_window)
            elif hasattr(self, 'use_transformers') and self.use_transformers:
                return self._predict_with_transformers(data, prediction_horizon, lookback_window)
            else:
                raise RuntimeError("No PatchTST implementation available")

        except Exception as e:
            logger.error(f"PatchTST prediction failed: {e}")
            raise

    def _predict_with_neuralforecast(self,
                                    data: pd.DataFrame,
                                    prediction_horizon: int,
                                    lookback_window: int) -> Dict:
        """Make predictions using NeuralForecast's PatchTST."""
        try:
            import platform
            import os

            # Force CPU usage on macOS
            if platform.system() == "Darwin":
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

            from neuralforecast import NeuralForecast
            from neuralforecast.models import PatchTST

            # Prepare data in NeuralForecast format
            # Need columns: unique_id, ds, y
            ts_data = pd.DataFrame({
                'unique_id': 'stock',
                'ds': data.index,
                'y': data['Close'].values
            })

            # Ensure we have enough data
            if len(ts_data) < lookback_window + prediction_horizon:
                lookback_window = max(30, len(ts_data) - prediction_horizon - 10)

            # Configure PatchTST model - use high quality settings but force CPU on macOS
            if platform.system() == 'Darwin':
                logger.info("Using high-quality PatchTST on CPU for macOS")
                models = [
                    PatchTST(
                        h=prediction_horizon,          # Forecast horizon
                        input_size=lookback_window,    # Look-back window
                        patch_len=16,                  # Standard patch length for quality
                        stride=8,                      # Standard stride
                        hidden_size=256,               # Large hidden size for better patterns
                        n_heads=8,                     # More attention heads for quality
                        scaler_type='robust',          # Robust scaling for financial data
                        max_steps=1000,                # More training for better accuracy
                        batch_size=32,                 # Larger batch for better gradients
                        random_seed=42,
                        early_stop_patience_steps=50,
                        val_check_steps=30,
                        learning_rate=5e-4,            # Optimized learning rate
                        dropout=0.1,                   # Standard dropout
                        encoder_layers=4,              # Deeper model for quality
                        accelerator='cpu',             # Force CPU to avoid MPS issues
                        num_workers_loader=0,          # Single-threaded loading for stability
                        activation='gelu',             # Better activation
                        loss=None,                     # Use default loss
                        valid_loss=None                # Use default valid loss
                    )
                ]
            else:
                models = [
                    PatchTST(
                        h=prediction_horizon,          # Forecast horizon
                        input_size=lookback_window,    # Look-back window
                        patch_len=16,                  # Standard patch length
                        stride=8,                      # Standard stride
                        hidden_size=256,               # Large hidden size
                        n_heads=8,                     # More attention heads
                        scaler_type='robust',          # Robust scaling for financial data
                        max_steps=1000,                # More training steps
                        batch_size=32,                 # Larger batch size
                        random_seed=42,
                        early_stop_patience_steps=50,
                        val_check_steps=30,
                        learning_rate=5e-4,            # Optimized learning rate
                        dropout=0.1,                   # Add dropout for regularization
                        encoder_layers=4,              # Deeper encoder
                        accelerator='auto',            # Auto select best device
                        activation='gelu',             # Better activation
                        loss=None,                     # Use default loss
                        valid_loss=None                # Use default valid loss
                    )
                ]

            # Create and train model - force CPU on macOS to avoid segmentation faults
            import platform
            if platform.system() == "Darwin":
                # Force CPU usage on macOS
                nf = NeuralForecast(models=models, freq='D')
                # Disable GPU/MPS training
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                logger.info("Using CPU for PatchTST on macOS to avoid segmentation faults")
            else:
                nf = NeuralForecast(models=models, freq='D')

            # Split data for training
            train_size = len(ts_data) - prediction_horizon
            train_data = ts_data.iloc[:train_size].copy()

            # Fit the model
            nf.fit(df=train_data, val_size=min(30, len(train_data) // 5))

            # Generate predictions
            forecasts = nf.predict()

            # Extract predictions
            predictions = forecasts['PatchTST'].values[:prediction_horizon]

            # Calculate metrics
            current_price = data['Close'].iloc[-1]
            predicted_price = predictions[-1]
            predicted_return = (predicted_price - current_price) / current_price

            # Generate more realistic confidence intervals
            # Use a smaller multiplier for financial predictions
            residuals_std = np.std(predictions) * 0.05 if len(predictions) > 1 else current_price * 0.01
            predictions_lower = predictions - 1.96 * residuals_std
            predictions_upper = predictions + 1.96 * residuals_std

            # Calculate confidence based on prediction variance
            prediction_variance = np.var(predictions) / (current_price ** 2) if current_price != 0 else 0.1
            confidence = max(0.5, min(0.85, 1.0 - prediction_variance * 10))  # Dynamic confidence

            return {
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'predicted_return': float(predicted_return),
                'confidence': float(confidence),
                'predictions_mean': predictions.tolist(),
                'predictions_lower': predictions_lower.tolist(),
                'predictions_upper': predictions_upper.tolist(),
                'price_range_lower': float(predictions_lower[-1]),
                'price_range_upper': float(predictions_upper[-1]),
                'model_type': 'patchtst',
                'prediction_horizon': prediction_horizon,
                'lookback_window': lookback_window
            }

        except Exception as e:
            logger.error(f"NeuralForecast PatchTST failed: {e}")
            # Fallback to simple prediction if model fails
            return self._simple_fallback(data, prediction_horizon)

    def _predict_with_transformers(self,
                                  data: pd.DataFrame,
                                  prediction_horizon: int,
                                  lookback_window: int) -> Dict:
        """Make predictions using Hugging Face Transformers PatchTST."""
        try:
            import torch
            from transformers import PatchTSTForPrediction, PatchTSTConfig

            # Prepare data
            prices = data['Close'].values
            context_length = min(len(prices), lookback_window)
            context = prices[-context_length:]

            # Normalize data
            mean = np.mean(context)
            std = np.std(context) if np.std(context) > 0 else 1.0
            context_normalized = (context - mean) / std

            # Configure model if not already done
            if self.model is None:
                config = PatchTSTConfig(
                    prediction_length=prediction_horizon,
                    context_length=context_length,
                    patch_length=self.patch_len,
                    patch_stride=self.stride,
                    num_input_channels=1,
                    d_model=self.hidden_size,
                    num_attention_heads=self.n_heads,
                )
                self.model = PatchTSTForPrediction(config)

            # Convert to tensor
            past_values = torch.tensor(
                context_normalized.reshape(1, -1, 1),
                dtype=torch.float32
            )

            # Generate predictions
            with torch.no_grad():
                outputs = self.model(past_values=past_values)
                predictions_normalized = outputs.prediction_outputs.squeeze().numpy()

            # Denormalize predictions
            predictions = predictions_normalized * std + mean

            # Ensure predictions have correct length
            if len(predictions) < prediction_horizon:
                # Pad with last value if needed
                predictions = np.pad(
                    predictions,
                    (0, prediction_horizon - len(predictions)),
                    'edge'
                )
            elif len(predictions) > prediction_horizon:
                predictions = predictions[:prediction_horizon]

            # Calculate metrics
            current_price = prices[-1]
            predicted_price = predictions[-1]
            predicted_return = (predicted_price - current_price) / current_price

            # Generate confidence intervals
            predictions_std = np.std(predictions) * 0.1 if len(predictions) > 1 else current_price * 0.02
            predictions_lower = predictions - 1.96 * predictions_std
            predictions_upper = predictions + 1.96 * predictions_std

            confidence = 0.65  # Base confidence for transformer model

            return {
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'predicted_return': float(predicted_return),
                'confidence': float(confidence),
                'predictions_mean': predictions.tolist(),
                'predictions_lower': predictions_lower.tolist(),
                'predictions_upper': predictions_upper.tolist(),
                'price_range_lower': float(predictions_lower[-1]),
                'price_range_upper': float(predictions_upper[-1]),
                'model_type': 'patchtst-transformers',
                'prediction_horizon': prediction_horizon,
                'lookback_window': context_length
            }

        except Exception as e:
            logger.error(f"Transformers PatchTST failed: {e}")
            return self._simple_fallback(data, prediction_horizon)

    def _simple_fallback(self, data: pd.DataFrame, prediction_horizon: int) -> Dict:
        """Simple fallback prediction using moving average."""
        prices = data['Close'].values
        current_price = prices[-1]

        # Use more conservative trend estimation
        if len(prices) > 20:
            ma_20 = np.mean(prices[-20:])
            ma_5 = np.mean(prices[-5:])
            # Combine short and long term trends
            trend = ((ma_5 - ma_20) / ma_20) * 0.5  # Dampen the trend
        else:
            trend = 0

        # Project trend forward with dampening
        predicted_price = current_price * (1 + trend * min(prediction_horizon, 10) / 30)
        predicted_return = (predicted_price - current_price) / current_price

        # Generate simple linear predictions
        predictions = np.linspace(current_price, predicted_price, prediction_horizon)
        predictions_std = abs(predicted_price - current_price) * 0.1
        predictions_lower = predictions - predictions_std
        predictions_upper = predictions + predictions_std

        return {
            'predicted_price': float(predicted_price),
            'current_price': float(current_price),
            'predicted_return': float(predicted_return),
            'confidence': 0.4,  # Low confidence for fallback
            'predictions_mean': predictions.tolist(),
            'predictions_lower': predictions_lower.tolist(),
            'predictions_upper': predictions_upper.tolist(),
            'price_range_lower': float(predictions_lower[-1]),
            'price_range_upper': float(predictions_upper[-1]),
            'model_type': 'patchtst-fallback',
            'prediction_horizon': prediction_horizon
        }