import pandas as pd
from typing import Optional


class DataCleaner:
    def __init__(self, missing_threshold: float = 0.35):
        if not 0 <= missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")
        self.missing_threshold = missing_threshold

    def drop_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(how="all")

    def keep_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.select_dtypes(include="number")

    def drop_high_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_ratio = df.isna().mean()
        cols_to_drop = missing_ratio[missing_ratio > self.missing_threshold].index
        return df.drop(columns=cols_to_drop)

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.drop_empty_rows(df)
        df = self.keep_numeric_columns(df)
        df = self.drop_high_missing(df)
        return df