from abc import ABC, abstractmethod
import pandas as pd

class FeatureEngineer(ABC):
    """Abstract base class for feature engineering modules."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations to the dataframe."""
        pass


class PriceFeatureEngineer(FeatureEngineer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['discount_percentage'] = df.apply(
            lambda x: 1 - (x['discount_price'] / x['actual_price']) if x['actual_price'] != 0 else 0,
            axis=1
        )
        return df


class RatingFeatureEngineer(FeatureEngineer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df['adjusted_rating'] = df['ratings'] * df['no_of_ratings']
        return df
