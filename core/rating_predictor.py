import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data_preprocessor import RatingDataPreprocessor
from model.model_trainer import ModelTrainer
from model.model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class RatingPredictor:
    """Orchestration class for product rating prediction"""
    def __init__(self, category: str):
        self.category = category
        self.data_path = f'./dataset/{category}.csv'
        self.model_path = f'best_models/best_rating_model_{category}.joblib'
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.model = None

    def _split_data(self, data: pd.DataFrame) -> None:
        """Split data for rating prediction"""
        y = data['ratings']
        X = data.drop(columns=['ratings'])
        X = X[y.notna()]
        y = y[y.notna()]
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    def execute(self) -> None:
        try:
            # Data processing
            preprocessor = RatingDataPreprocessor(self.data_path)
            processed_data = preprocessor.process()
            
            # Data splitting
            self._split_data(processed_data)
            
            # Model training
            trainer = ModelTrainer(self.X_train, self.y_train)
            self.model = trainer.train()
            
            # Model evaluation
            val_metrics = ModelEvaluator.evaluate(self.model, self.X_val, self.y_val)
            ModelEvaluator.log_results(self.model, val_metrics, self.category, 'Validation')
            
            test_metrics = ModelEvaluator.evaluate(self.model, self.X_test, self.y_test)
            ModelEvaluator.log_results(self.model, test_metrics, self.category, 'Test')
            
            ModelEvaluator.log_examples(self.model, self.X_test, self.y_test)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Rating model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Error processing ratings for {self.category}: {str(e)}")
            raise
