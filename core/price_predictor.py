import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from data.data_preprocessor import PriceDataPreprocessor
from model.model_trainer import ModelTrainer
from model.model_evaluator import ModelEvaluator
from model.model_factory import PriceModelFactory

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self, category: str):
        self.category = category
        self.data_path = f'./dataset/{category}.csv'
        self.model_path = f'best_models/best_model_{category}.joblib'
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.model = None
        self.model_factory = PriceModelFactory()
    
    def _split_data(self, data: pd.DataFrame) -> None:
        y = data['actual_price']
        X = data.drop(columns=['actual_price'])
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
            preprocessor = PriceDataPreprocessor(self.data_path)
            processed_data = preprocessor.process()
            
            # Data splitting
            self._split_data(processed_data)
            
            # Model training
            trainer = ModelTrainer(self.X_train, self.y_train, self.model_factory)
            self.model = trainer.train(GradientBoostingRegressor())

            # Model evaluation
            val_metrics = ModelEvaluator.evaluate(self.model, self.X_val, self.y_val)
            ModelEvaluator.log_results(self.model, val_metrics, self.category, 'Validation')
            
            # Test set evaluation
            test_metrics = ModelEvaluator.evaluate(self.model, self.X_test, self.y_test)
            ModelEvaluator.log_results(self.model, test_metrics, self.category, 'Test')
            
            # Example predictions
            # ModelEvaluator.log_examples(self.model, self.X_test, self.y_test)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            
        except Exception as e:
            logger.error(f"Error processing {self.category}: {str(e)}")
            raise