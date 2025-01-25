import pandas as pd
from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from features.price_feature_engineer import PriceFeatureEngineer
from features.rating_feature_engineer import RatingFeatureEngineer


class BaseDataPreprocessor:
    """Base class for preprocessing data with common functionalities."""
    
    def __init__(self, data_path: str, feature_engineer):
        """
        Initialize the data preprocessor.

        Args:
            data_path (str): Path to the data file.
            feature_engineer: Feature engineering class instance.
        """
        self.data_path = data_path
        self.cleaner = DataCleaner()
        self.feature_engineer = feature_engineer

    def process(self) -> pd.DataFrame:
        """
        Load, clean, and transform the data using feature engineering.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        df = DataLoader.load(self.data_path)
        df = self.cleaner.transform(df)
        return self.feature_engineer.transform(df)


class PriceDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor for handling price data."""
    
    def __init__(self, data_path: str):
        super().__init__(data_path, PriceFeatureEngineer())


class RatingDataPreprocessor(BaseDataPreprocessor):
    """Preprocessor for handling rating data."""
    
    def __init__(self, data_path: str):
        super().__init__(data_path, RatingFeatureEngineer())
