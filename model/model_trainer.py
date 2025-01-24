from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from .model_factory import BaseModelFactory

    
class ModelTrainer:
    def __init__(self, X_train, y_train, factory: BaseModelFactory):
        self.X_train = X_train
        self.y_train = y_train
        self.factory = factory
    
    def train(self, base_model):
        model_pipeline = self.factory.create_pipeline(base_model)
        hyperparams = self.factory.get_hyperparameters()
        
        search = RandomizedSearchCV(
            model_pipeline,
            hyperparams,
            cv=3,
            scoring='neg_mean_squared_error',
            n_iter=30,
            n_jobs=-1,
            random_state=135
        )
        search.fit(self.X_train, self.y_train)
        return search.best_estimator_