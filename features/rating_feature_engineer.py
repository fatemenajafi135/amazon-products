import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RatingFeatureEngineer(TransformerMixin, BaseEstimator):
    """Feature engineering for rating prediction"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Add rating-specific feature engineering
        df = df[df['ratings'].between(0, 5)]  # Ensure valid rating range
        return df
    
class PriceFeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.price_columns = ['actual_price', 'discount_price']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._clean_numeric_features(df)
        df = self._calculate_discount_percentage(df)
        df = self._remove_outliers(df)
        return df.dropna(subset=['actual_price'])
    
    # ... (keep all the methods as in original FeatureEngineer)


    def _clean_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['no_of_ratings'] = df['no_of_ratings'].apply(self._clean_numeric)
        df['ratings'] = df['ratings'].apply(self._clean_numeric).fillna(0)
        return df
    
    @staticmethod
    def _clean_numeric(value) -> float:
        if pd.isna(value):
            return 0.0
        try:
            return float(str(value).replace(',', ''))
        except ValueError:
            return 0.0
    
    def _calculate_discount_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        df['discount_percentage'] = df.apply(
            lambda x: 1 - (x['discount_price'] / x['actual_price']) 
            if x['actual_price'] != 0 else 0, axis=1
        )
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.price_columns:
            df = self._remove_iqr_outliers(df, col)
        return df
    
    @staticmethod
    def _remove_iqr_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
