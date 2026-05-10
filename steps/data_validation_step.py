import pandas as pd
from zenml import step


REQUIRED_COLUMNS = [
    "Price_in_Lakhs",
    "Area_SqFt",
    "City",
    "Property_Type",
    "BHK",
]
QUALITY_COLUMNS = ["Locality", "Building_Name", "Furnish", "Facing", "Amenities"]


def _log_filter(name: str, before_count: int, after_shape: tuple[int, int]) -> None:
    print(f"{name}: dropped {before_count - after_shape[0]} rows. New shape: {after_shape}")


@step
def data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate normalized 99Acres sale listings and remove rows unusable for training.
    """
    print(f"Starting data validation. Initial shape: {df.shape}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df_clean = df.copy()

    before = len(df_clean)
    if "Prop_ID" in df_clean.columns and df_clean["Prop_ID"].notna().any():
        df_clean = df_clean.drop_duplicates(subset=["Prop_ID"]).copy()
        _log_filter("Duplicate Prop_ID removal", before, df_clean.shape)
    else:
        df_clean = df_clean.drop_duplicates().copy()
        _log_filter("Full-row duplicate removal", before, df_clean.shape)

    before = len(df_clean)
    df_clean = df_clean.dropna(subset=REQUIRED_COLUMNS).copy()
    _log_filter("Missing core field removal", before, df_clean.shape)

    filters = [
        ("Price bounds 1-5000 lakhs", df_clean["Price_in_Lakhs"].between(1, 5000)),
        ("Area bounds 100-20000 sqft", df_clean["Area_SqFt"].between(100, 20000)),
        ("BHK bounds 1-10", df_clean["BHK"].between(1, 10)),
    ]
    for name, mask in filters:
        before = len(df_clean)
        df_clean = df_clean[mask.fillna(False)].copy()
        _log_filter(name, before, df_clean.shape)

    floor_mask = (
        df_clean["Floor_No"].notna()
        & df_clean["Total_Floors"].notna()
        & (df_clean["Floor_No"] > 0)
        & (df_clean["Total_Floors"] > 0)
        & (df_clean["Floor_No"] > df_clean["Total_Floors"])
    ).fillna(False)
    before = len(df_clean)
    df_clean = df_clean[~floor_mask].copy()
    _log_filter("Invalid positive floor relationship removal", before, df_clean.shape)

    print("Validation missing-rate summary:")
    for column in QUALITY_COLUMNS:
        if column in df_clean.columns:
            missing_rate = df_clean[column].isna().mean() * 100
            print(f"  - {column}: {missing_rate:.2f}% missing")

    print(f"Final validated shape: {df_clean.shape}")
    return df_clean
