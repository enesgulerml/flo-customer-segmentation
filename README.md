# 👟 FLO Customer Segmentation & RFM Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Model-K--Means-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)
![Docker](https://img.shields.io/badge/Docker-Production-2496ED)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688)

## 📖 Overview
This repository implements a production-ready **Unsupervised Machine Learning pipeline** to segment e-commerce customers based on their purchasing behavior. Using real-world data from FLO (a leading shoe retailer), the system identifies distinct customer personas to enable targeted marketing strategies.

**Methodology:**
The project combines **RFM Analysis** (Recency, Frequency, Monetary) with **K-Means Clustering** to classify customers. The resulting model is deployed via a decoupled architecture featuring a FastAPI backend and a Streamlit dashboard.

**Key Features:**
* **Data Engineering:** Automatic calculation of RFM metrics from transaction logs.
* **MLOps Workflow:** End-to-end pipeline with MLflow tracking and Model Registry.
* **Embedded Deployment:** Docker container with "baked-in" model artifacts for immutable deployments.
* **Interactive Intelligence:** Streamlit dashboard for visualizing customer clusters.

---

## 📂 Project Structure

```text
flo-customer-segmentation/
│
├── app/                  # Inference Service
│   ├── main.py           # FastAPI Application
│   ├── schema.py         # Pydantic Models for Input Validation
│   └── model_files/      # (CI/CD Artifacts) Embedded K-Means Model
│
├── dashboard/            # Business Intelligence UI
│   └── app.py            # Streamlit Dashboard Logic
│
├── src/                  # Core ML Pipeline
│   ├── config.py         # Configuration & Path Management
│   ├── data_processing.py# RFM Calculation & Feature Engineering
│   ├── train.py          # K-Means Training & MLflow Logging
│   └── fetch_model.py    # CI/CD Script: Artifact Retrieval
│
├── notebooks/            # Exploratory Data Analysis
│   └── 01-eda.ipynb      # Initial clustering experiments
│
├── tests/                # Quality Assurance
│   └── test_api.py       # API Endpoint Tests
│
├── requirements.txt      # Project Dependencies
└── Dockerfile            # Production Container Setup
```

## 🛠️ Installation & Setup
Prerequisites
* Python 3.10+
* Docker (Optional but recommended)
* Dataset: Ensure [flo_data_20k.csv](https://www.kaggle.com/code/mustafaoz158/flo-cltv-prediction/input) is placed in data/raw/.

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/enesgulerml/flo-customer-segmentation.git
cd flo-customer-segmentation

# Create Virtual Environment
conda create -n flo-segmentation python=3.10 -y
conda activate flo-segmentation

# Install Dependencies
pip install -r requirements.txt
pip install -e .
```

## ⚡ MLOps Workflow
This project follows a strict ETL -> Train -> Register -> Serve lifecycle.

### Phase 1: Data Processing & Training
1. Start MLflow Server:

This step converts raw transaction logs into RFM scores, determines the optimal number of clusters (using Elbow Method/Silhouette Score), and registers the K-Means model.
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```

2. Run Pipeline:

```bash
python -m src.train
```
This script automatically performs RFM feature engineering before training.

### Phase 2: Build & Deployment (CI/CD)
We use the Embedded Model Pattern. The model is fetched from the registry and built into the Docker image, ensuring the container is self-sufficient.

1. Fetch Artifacts: Downloads the latest production model to app/model_files/.

```bash
python -m src.fetch_model
```

2. Build Docker Image:
```bash
docker build -t flo-api:latest .
```

### Phase 3: Serving & Visualization
Deploy the API and connect the dashboard.

1. Run API Container: Runs on port 8005.
```bash
docker run -d --rm -p 8005:80 flo-api:latest
```
👉 API Docs: http://localhost:8005/docs

2. Launch Dashboard: Visualize the segments and analyze customer profiles.
```bash
streamlit run dashboard/app.py
```
👉 Dashboard: http://localhost:8501

## 🧪 Testing
Automated tests ensure the API correctly handles RFM inputs and predicts clusters.
```bash
# Run API integration tests
pytest tests/test_api.py -v
```

## 📊 Business Logic: RFM Analysis
For those interested in the underlying methodology:
* Recency (R): Days since last purchase.
* Frequency (F): Total number of purchases.
* Monetary (M): Total spending.

The model clusters customers into groups (e.g., "Loyal", "Hibernating", "New Customers") allowing marketing teams to take specific actions for each group.