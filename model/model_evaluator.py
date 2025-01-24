import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and reporting"""
    @staticmethod
    def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = model.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
    
    @staticmethod
    def log_results(model: Pipeline, metrics: dict, category: str, dataset_type: str) -> None:
        logger.info(f"\nCategory: {category} - {dataset_type} Results")
        logger.info(f"Best Model: {model.named_steps['regressor'].regressor.__class__.__name__}")
        logger.info(f"RMSE: {metrics['rmse']:.2f}")
        logger.info(f"MAE: {metrics['mae']:.2f}")
        logger.info(f"R²: {metrics['r2']:.2f}")
    
    @staticmethod
    def log_examples(model: Pipeline, X: pd.DataFrame, y: pd.Series, num_samples: int = 5) -> None:
        """Log example predictions with actual values"""
        samples = X.sample(num_samples)
        actuals = y.loc[samples.index]
        preds = model.predict(samples)
        
        logger.info("\nExample Predictions:")
        for idx, (actual, pred) in enumerate(zip(actuals, preds)):
            logger.info(f"Sample {idx+1}:")
            logger.info(f"  Actual: {actual:.2f}")
            logger.info(f"  Predicted: {pred:.2f}")
            logger.info(f"  Difference: {abs(actual - pred):.2f} ({abs((actual - pred)/actual)*100:.1f}%)")
            