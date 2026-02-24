import pandas as pd
from pathlib import Path


def load_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path)

    raise ValueError("Unsupported file format")


def save_file(df: pd.DataFrame, path: str | Path):
    path = Path(path)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return

    if path.suffix in [".xls", ".xlsx"]:
        df.to_excel(path, index=False)
        return

    raise ValueError("Unsupported file format")