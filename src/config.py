# src/config.py

from pathlib import Path

# === 1. File Paths (Dynamic and Absolute) ===
SRC_ROOT = Path(__file__).parent
PROJECT_ROOT = SRC_ROOT.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flo_data_20k.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rfm_data.csv"

# Model and Reports
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "kmeans_model.joblib"
CLUSTERS_OUTPUT_PATH = PROJECT_ROOT / "reports" / "customer_clusters.csv"

# === 2. Dataset Columns (Omnichannel Strategy) ===
# Columns to be aggregated for total frequency and monetary values
COL_ORDER_NUM_ONLINE = "order_num_total_ever_online"
COL_ORDER_NUM_OFFLINE = "order_num_total_ever_offline"
COL_VALUE_ONLINE = "customer_value_total_ever_online"
COL_VALUE_OFFLINE = "customer_value_total_ever_offline"

# Date columns to be converted to datetime objects
DATE_COLUMNS = [
    "first_order_date",
    "last_order_date",
    "last_order_date_online",
    "last_order_date_offline"
]

# Analysis Date for Recency calculation (2021-06-01)
ANALYSIS_DATE = "2021-06-01"

# === 3. Outlier Detection Settings ===
# Columns to apply IQR cleaning before clustering
OUTLIER_COLUMNS = ["total_order", "total_price"]
IQR_THRESHOLD = 1.5

# === 4. Model Parameters (K-Means Auto-Tuning) ===
# We will test cluster numbers from MIN to MAX and pick the best one.
K_MEANS_RANGE = range(3, 11) # Try 3, 4, ... 10 clusters
RANDOM_STATE = 42
K_MEANS_INIT = 10

# MLflow
MLFLOW_EXPERIMENT_NAME = "FLO Customer Segmentation"
MODEL_REGISTRY_NAME = "FloSegmentationModel"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"