# src/data_processing.py

import pandas as pd
import numpy as np
import datetime as dt
import sys
from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    COL_ORDER_NUM_ONLINE,
    COL_ORDER_NUM_OFFLINE,
    COL_VALUE_ONLINE,
    COL_VALUE_OFFLINE,
    DATE_COLUMNS,
    ANALYSIS_DATE,
    OUTLIER_COLUMNS,
    IQR_THRESHOLD
)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the dataset using the IQR (Interquartile Range) method.
    This creates tighter clusters and improves model performance.
    """
    print(f"Removing outliers from {OUTLIER_COLUMNS} with threshold {IQR_THRESHOLD}...")
    df_clean = df.copy()
    initial_count = len(df_clean)

    for col in OUTLIER_COLUMNS:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (IQR_THRESHOLD * IQR)
        upper_bound = Q3 + (IQR_THRESHOLD * IQR)

        # Filter data within bounds
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) &
            (df_clean[col] <= upper_bound)
            ]

    final_count = len(df_clean)
    print(f"Outlier removal complete. Dropped {initial_count - final_count} rows.")
    return df_clean


def load_and_process_data() -> pd.DataFrame:
    """
    v1.3 - Main Data Pipeline:
    1. Load Raw Data
    2. Omnichannel Integration
    3. Date Conversion
    4. Outlier Removal (New Step)
    5. Feature Engineering (Recency, Tenure)
    6. Log Transformation
    7. Export Processed Data
    """
    print(">>> Starting Data Processing Pipeline (v1.3)...")

    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at: {RAW_DATA_PATH}")
        sys.exit(1)

    # 2. Omnichannel Transformation
    df["total_order"] = df[COL_ORDER_NUM_ONLINE] + df[COL_ORDER_NUM_OFFLINE]
    df["total_price"] = df[COL_VALUE_ONLINE] + df[COL_VALUE_OFFLINE]

    # 3. Date Conversion
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col])

    # 4. Outlier Removal (Applied on raw total_order and total_price)
    df = remove_outliers(df)

    # 5. Feature Engineering
    analysis_date = pd.to_datetime(ANALYSIS_DATE)
    df["recency"] = (analysis_date - df["last_order_date"]).dt.days
    df["tenure"] = (analysis_date - df["first_order_date"]).dt.days

    # Select Columns
    rfm_df = df[["master_id", "recency", "total_order", "total_price", "tenure"]]
    rfm_df.columns = ["master_id", "Recency", "Frequency", "Monetary", "Tenure"]

    # 6. Log Transformation
    # Create a copy to avoid SettingWithCopyWarning
    rfm_df = rfm_df.copy()
    print("Applying Log Transformation to reduce skewness...")

    rfm_df["log_Recency"] = np.log1p(rfm_df["Recency"])
    rfm_df["log_Frequency"] = np.log1p(rfm_df["Frequency"])
    rfm_df["log_Monetary"] = np.log1p(rfm_df["Monetary"])
    rfm_df["log_Tenure"] = np.log1p(rfm_df["Tenure"])

    # 7. Save Processed Data
    try:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        rfm_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Processed RFM data saved to: {PROCESSED_DATA_PATH}")
    except Exception as e:
        print(f"ERROR saving processed data: {e}")
        sys.exit(1)

    return rfm_df


if __name__ == "__main__":
    load_and_process_data()