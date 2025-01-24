import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataCleaner(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in ['actual_price', 'discount_price']:
            df[col] = df[col].apply(self._clean_price)
        df['discount_price'] = df['discount_price'].fillna(df['actual_price'])
        return df
    
    @staticmethod
    def _clean_price(price) -> float:
        if pd.isna(price):
            return price
        return float(str(price).replace('₹', '').replace(',', ''))
    