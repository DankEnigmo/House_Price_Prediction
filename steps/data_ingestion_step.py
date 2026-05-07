import os
from pathlib import Path

import pandas as pd
from zenml import step


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "india_housing_prices.csv"


@step
def data_ingestion() -> pd.DataFrame:
    """
    Data ingestion step to load the India House Price Prediction dataset from local CSV.

    Returns:
        A Pandas DataFrame containing the raw housing data.
    """
    csv_path = Path(os.getenv("HOUSE_PRICE_DATA_PATH", DEFAULT_DATA_PATH))

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path).drop_duplicates()
    print(f"Data ingested. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df
