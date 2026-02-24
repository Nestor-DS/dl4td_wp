import pandas as pd


def nan_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "nan_count": df.isna().sum(),
        "nan_ratio": df.isna().mean()
    })


def zero_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "zero_count": (df == 0).sum(),
        "zero_ratio": (df == 0).mean()
    })


def completeness_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    return pd.DataFrame({
        "non_null_count": df.count(),
        "non_null_ratio": df.count() / total
    })