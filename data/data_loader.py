import pandas as pd

class DataLoader:
    @staticmethod
    def load(data_path: str) -> pd.DataFrame:
        return pd.read_csv(data_path)
        