import pandas as pd
from zenml import step

@step
def data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data validation step to clean illogical rows and handle obvious outliers.
    """
    print(f"Starting data validation. Initial shape: {df.shape}")
    
    # 1. Drop rows where Floor_No > Total_Floors
    df_clean = df[df["Floor_No"] <= df["Total_Floors"]].copy()
    print(f"Dropped rows with Floor_No > Total_Floors. New shape: {df_clean.shape}")
    
    # 2. Filter out extreme Price outliers (e.g., houses < 1 Lakh or > 20,000 Lakhs)
    # This prevents the log-mean from being destroyed by garbage placeholders
    df_clean = df_clean[(df_clean["Price_in_Lakhs"] >= 1) & (df_clean["Price_in_Lakhs"] <= 20000)]
    print(f"Dropped price outliers (<1L or >20000L). New shape: {df_clean.shape}")
    
    # 3. Handle BHK and Size outliers
    df_clean = df_clean[(df_clean["BHK"] >= 1) & (df_clean["BHK"] <= 10)]
    df_clean = df_clean[(df_clean["Size_in_SqFt"] >= 100) & (df_clean["Size_in_SqFt"] <= 20000)]
    print(f"Dropped BHK/Size outliers. Final validated shape: {df_clean.shape}")
    
    return df_clean
