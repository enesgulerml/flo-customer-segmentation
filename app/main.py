import mlflow.pyfunc
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schema import CustomerInput, PredictionResponse

models = {}
MODEL_DIR = "app/model_files"

CLUSTER_NAMES = {
    0: "Hibernating",
    1: "Champions",
    2: "Loyal Customers",
    3: "At Risk",
    4: "New Customers"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Initializing API... Model Source: {MODEL_DIR}")
    try:
        models["kmeans"] = mlflow.pyfunc.load_model(MODEL_DIR)
        print(f"✅ Model has successfully loaded!")
    except Exception as e:
        print(f"❌ Model Download Error: {e}")
        print("Please make sure you run the command 'python -m src.fetch_model'.")
        models["kmeans"] = None
    yield
    models.clear()


app = FastAPI(title="FLO Customer Segmentation API", lifespan=lifespan)


def preprocess_input(data: CustomerInput) -> pd.DataFrame:
    """
    It converts the incoming raw data (Pydantic) into a DataFrame and applies the Log Transform (log1p) on which the model was trained.
    """
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # --- Feature Engineering (Log Transform) ---

    df["log_Recency"] = np.log1p(df["recency_days"])
    df["log_Frequency"] = np.log1p(df["total_orders"])
    df["log_Monetary"] = np.log1p(df["total_price"])

    df["log_Tenure"] = np.log1p(df["tenure_days"])

    expected_cols = ["log_Recency", "log_Frequency", "log_Monetary", "log_Tenure"]
    df = df[expected_cols]

    return df


@app.get("/")
def health_check():
    status = "Active" if models["kmeans"] else "Inactive"
    return {"status": status, "service": "FLO Segmentation API"}


@app.post("/predict", response_model=PredictionResponse)
def predict_segment(customer: CustomerInput):
    if not models["kmeans"]:
        raise HTTPException(status_code=503, detail="Model yüklenemedi.")

    try:
        processed_df = preprocess_input(customer)

        cluster_id = int(models["kmeans"].predict(processed_df)[0])


        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Segment {cluster_id}")

        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "model_version": "Embedded v1.0"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))