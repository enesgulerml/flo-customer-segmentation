# FLO Customer Segmentation (RFM & K-Means)

This project implements an end-to-end **Unsupervised Learning** pipeline to segment customers based on their Omnichannel (Online + Offline) shopping behavior.

Using the **FLO dataset**, it transforms raw transaction logs into customer-centric RFM (Recency, Frequency, Monetary) features and clusters them using **K-Means**.

* **v1.0: Data Engineering:** Omnichannel integration, IQR Outlier Removal, and Log Transformation.
* **v2.0: Model Training:** K-Means clustering with Auto-Tuning (Elbow Method) and MLflow Tracking.

---

## 🚀 Project Structure

```
flo-customer-segmentation/
│
├── app/
│   └── __init__.py
│
├── dashboard/
│
├── data/
│   ├── raw/
│   │   └── flo_data.csv
│   └── processed/
│       └── rfm_data.csv
│
├── models/
│   └── kmeans_model.joblib
│
├── mlruns/
├── mlartifacts/
│
├── notebooks/
│   └── 01-eda.ipynb
│
├── reports/
│   └── customer_clusters.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   └── train.py
│
├── test/
│   └── __init__.py
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🛠️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/enesgulerml/flo-customer-segmentation.git
    cd flo-customer-segmentation
    ```

2.  **Setup Environment:**
    ```bash
    conda create -n flo-segmentation python=3.10 -y
    conda activate flo-segmentation
    pip install -r requirements.txt
    pip install -e .
    ```

---

## ⚡ Usage

### 1. Start MLflow Server
Keep this terminal running to track experiments and model registry.
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

### 2. Run Data Pipeline & Training
This script will:
* Load and merge Online/Offline data.
* Remove outliers (IQR).
* Apply Log Transformation (np.log1p) to fix skewness.
* Auto-tune K-Means (Test k=3 to 10) and find the best Silhouette Score.
* Register the best model to MLflow.

```bash
python -m src.train
```

### 3. Check Results
* **Logs:** Check http://127.0.0.1:5000 for metrics.
* **Clusters:** Check reports/customer_clusters.csv for the segmented customer list.

---

