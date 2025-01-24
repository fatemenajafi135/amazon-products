from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler, FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import numpy as np

from abc import ABC, abstractmethod

class BaseModelFactory(ABC):
    @abstractmethod
    def create_pipeline(self):
        pass
    
    @abstractmethod
    def get_hyperparameters(self):
        pass

class PriceModelFactory(BaseModelFactory):
    def create_pipeline(self, base_model):
        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )),
            ('svd', TruncatedSVD(n_components=50))
        ])

        preprocessor = ColumnTransformer([
            ('ratings_scaler', RobustScaler(), ['ratings']),
            ('no_of_ratings_scaler', Pipeline([
                ('log1p', FunctionTransformer(np.log1p)),
                ('robust_scaler', RobustScaler())
            ]), ['no_of_ratings']),
            ('name_tfidf', text_pipeline, 'name'),
        ], remainder='drop')

        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', TransformedTargetRegressor(
                regressor=base_model,
                transformer=StandardScaler()
            ))
        ])
    
    def get_hyperparameters(self):
        return [{
            'regressor__regressor': [GradientBoostingRegressor(random_state=42)],
            'regressor__regressor__n_estimators': [100, 200, 300],
            'regressor__regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__regressor__max_depth': [3, 5, 7],
            'regressor__regressor__subsample': [0.8, 1.0],
            'preprocessor__name_tfidf__svd__n_components': [50],
            'preprocessor__name_tfidf__tfidf__max_features': [100, 200]
        # }, {
        #     'regressor__regressor': [XGBRegressor(random_state=42)],
        #     'regressor__regressor__n_estimators': [100, 200, 300],
        #     'regressor__regressor__learning_rate': [0.01, 0.05, 0.1],
        #     'regressor__regressor__max_depth': [3, 5, 7],
        #     'regressor__regressor__subsample': [0.8, 1.0],
        #     'regressor__regressor__colsample_bytree': [0.8, 1.0],
        #     'preprocessor__name_tfidf__svd__n_components': [50],
        #     'preprocessor__name_tfidf__tfidf__max_features': [100, 200]
        }]

class RatingModelFactory(BaseModelFactory):
    def create_pipeline(self, base_model):

        text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('svd', TruncatedSVD(n_components=50))
        ])

        preprocessor = ColumnTransformer([
            ('price_scaler', RobustScaler(), ['actual_price', 'discount_price']),
            ('name_tfidf', text_pipeline, 'name'),
        ], remainder='drop')

        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', base_model)
        ])
    
    def get_hyperparameters(self):
        return [{
            'regressor': [GradientBoostingRegressor()],
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [3, 5]
        }, {
            'regressor': [XGBRegressor()],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__max_depth': [3, 5]
        }]
