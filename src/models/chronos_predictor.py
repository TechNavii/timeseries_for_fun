"""Chronos-based time series prediction for stocks using Amazon's pre-trained models."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
import os
import platform

warnings.filterwarnings('ignore')

# Critical environment settings for macOS stability
if platform.system() == "Darwin":
    # Disable MPS and force CPU usage
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Disable parallelism to avoid threading issues that cause segfaults
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    # Disable torch compile
    os.environ['TORCHDYNAMO_DISABLE'] = '1'

logger = logging.getLogger(__name__)

class ChronosPredictor:
    """Stock price predictor using Amazon's Chronos transformer models."""

    def __init__(self, model_size: str = "base"):
        """
        Initialize Chronos predictor.

        Args:
            model_size: Size of Chronos model - 'tiny', 'mini', 'small', 'base'
                       Default is 'base' for best quality predictions
                       Standard T5-based Chronos models for reliable predictions
        """
        self.model_size = model_size
        self.pipeline = None
        self.device = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of the model to avoid loading it at startup."""
        if self._initialized:
            return

        try:
            import torch

            # Force CPU on macOS with special settings
            if platform.system() == "Darwin":
                self.device = "cpu"
                logger.info("macOS detected - using CPU with optimized settings")
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            logger.info(f"Chronos will use device: {self.device}")

            # Try to import and initialize Chronos
            try:
                from chronos import ChronosPipeline

                # Use standard Chronos models (not Bolt) to avoid config issues
                model_name = f"amazon/chronos-t5-{self.model_size}"

                logger.info(f"Loading Chronos model: {model_name}")

                if self.device == "cuda":
                    self.pipeline = ChronosPipeline.from_pretrained(
                        model_name,
                        device_map="cuda",
                        dtype=torch.float16,
                    )
                else:
                    # Use CPU with optimized settings for macOS
                    self.pipeline = ChronosPipeline.from_pretrained(
                        model_name,
                        device_map="cpu",
                        dtype=torch.float32,
                        low_cpu_mem_usage=True  # Reduce memory usage on CPU
                    )

                self._initialized = True
                self.use_simple = False
                logger.info("Chronos model loaded successfully")

            except Exception as e:
                # Re-raise the error - no fallback
                logger.error(f"Chronos initialization failed: {e}")
                raise RuntimeError(f"Could not initialize Chronos model: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize Chronos: {e}")
            raise RuntimeError(f"Could not initialize Chronos model: {e}")

    def _init_with_autogluon(self):
        """Initialize using AutoGluon's Chronos integration."""
        try:
            from autogluon.timeseries import TimeSeriesPredictor
            self.use_autogluon = True
            self._initialized = True
            logger.info("Will use AutoGluon's Chronos integration")
        except ImportError:
            raise ImportError(
                "Neither Chronos nor AutoGluon is installed. "
                "Install with: pip install chronos-forecasting or pip install autogluon.timeseries"
            )

    def predict(self,
                data: pd.DataFrame,
                prediction_horizon: int = 5,
                num_samples: int = 100) -> Dict:
        """
        Make stock price predictions using Chronos.

        Args:
            data: Historical stock data with at least 'Close' column
            prediction_horizon: Number of days to predict ahead
            num_samples: Number of probabilistic samples for uncertainty

        Returns:
            Dictionary with predictions and confidence intervals
        """
        self._lazy_init()

        # No fallback - Chronos must be properly initialized
        if not self.pipeline and not hasattr(self, 'use_autogluon'):
            raise RuntimeError("Chronos model not initialized")

        try:
            # Extract close prices
            if 'Close' not in data.columns:
                raise ValueError("Data must have 'Close' column")

            prices = data['Close'].values

            # Use at most last 500 days for context (Chronos works best with reasonable context)
            context_length = min(len(prices), 500)
            context = prices[-context_length:]

            if hasattr(self, 'use_autogluon') and self.use_autogluon:
                return self._predict_with_autogluon(data, prediction_horizon)

            import torch

            # Convert to tensor - for MPS, keep on CPU to avoid device conflicts
            context_tensor = torch.tensor(context, dtype=torch.float32)
            # Only move to CUDA, keep on CPU for MPS to avoid searchsorted errors
            if self.device == "cuda":
                context_tensor = context_tensor.to(self.device)

            # Generate probabilistic forecast
            with torch.no_grad():
                forecast = self.pipeline.predict(
                    context=context_tensor,
                    prediction_length=prediction_horizon,
                    num_samples=num_samples
                )

            # Convert forecast to numpy for processing
            if isinstance(forecast, torch.Tensor):
                # Move to CPU first if on MPS or CUDA
                if forecast.device.type != 'cpu':
                    forecast = forecast.cpu()
                forecast = forecast.numpy()

            # Squeeze batch dimension if present (shape should be [batch=1, num_samples, pred_len])
            if forecast.ndim == 3 and forecast.shape[0] == 1:
                forecast = forecast.squeeze(0)  # Now shape is [num_samples, pred_len]

            # Calculate statistics over samples (axis=0)
            predictions_mean = np.mean(forecast, axis=0)  # Shape: [pred_len]
            predictions_std = np.std(forecast, axis=0)
            predictions_lower = np.percentile(forecast, 5, axis=0)
            predictions_upper = np.percentile(forecast, 95, axis=0)

            # Calculate trend and confidence
            current_price = float(prices[-1])
            predicted_price = float(predictions_mean[-1])  # Last day of prediction
            predicted_return = float((predicted_price - current_price) / current_price)

            # Confidence based on prediction spread
            avg_spread = float(np.mean((predictions_upper - predictions_lower) / predictions_mean))
            confidence = float(max(0.3, min(0.9, 1.0 - avg_spread)))  # Normalize to 0.3-0.9

            return {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'predicted_return': predicted_return,
                'confidence': confidence,
                'predictions_mean': predictions_mean.tolist(),
                'predictions_lower': predictions_lower.tolist(),
                'predictions_upper': predictions_upper.tolist(),
                'predictions_std': predictions_std.tolist(),
                'price_range_lower': float(predictions_lower[-1]),
                'price_range_upper': float(predictions_upper[-1]),
                'model_type': f'chronos-t5-{self.model_size}',
                'prediction_horizon': prediction_horizon,
                'context_length': context_length
            }

        except Exception as e:
            logger.error(f"Chronos prediction failed: {e}")
            raise

    def _predict_with_autogluon(self, data: pd.DataFrame, prediction_horizon: int) -> Dict:
        """Make predictions using AutoGluon's Chronos integration."""
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

            # Prepare data in AutoGluon format
            ts_data = pd.DataFrame({
                'item_id': ['stock'] * len(data),
                'timestamp': data.index,
                'target': data['Close'].values
            })

            train_data = TimeSeriesDataFrame.from_dataframe(
                ts_data,
                id_column='item_id',
                timestamp_column='timestamp'
            )

            # Create predictor with Chronos preset
            predictor = TimeSeriesPredictor(
                prediction_length=prediction_horizon,
                eval_metric='MASE',
                verbosity=0
            )

            # Use standard Chronos preset (not Bolt to avoid issues)
            preset = 'best_quality'  # Use AutoGluon's best quality preset which includes Chronos

            predictor.fit(
                train_data=train_data,
                presets=preset,
                time_limit=60  # 1 minute time limit
            )

            # Generate predictions
            predictions = predictor.predict(train_data, quantile_levels=[0.05, 0.5, 0.95])

            # Extract prediction values
            pred_mean = predictions['mean'].values[-prediction_horizon:]
            pred_lower = predictions['0.05'].values[-prediction_horizon:]
            pred_upper = predictions['0.95'].values[-prediction_horizon:]

            current_price = data['Close'].iloc[-1]
            predicted_price = pred_mean[-1]
            predicted_return = (predicted_price - current_price) / current_price

            # Calculate confidence
            avg_spread = np.mean((pred_upper - pred_lower) / pred_mean)
            confidence = max(0.3, min(0.9, 1.0 - avg_spread))

            return {
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'predicted_return': float(predicted_return),
                'confidence': float(confidence),
                'predictions_mean': pred_mean.tolist(),
                'predictions_lower': pred_lower.tolist(),
                'predictions_upper': pred_upper.tolist(),
                'price_range_lower': float(pred_lower[-1]),
                'price_range_upper': float(pred_upper[-1]),
                'model_type': f'chronos-t5-{self.model_size}',
                'prediction_horizon': prediction_horizon
            }

        except Exception as e:
            logger.error(f"AutoGluon Chronos prediction failed: {e}")
            raise

