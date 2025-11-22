import mlflow
import shutil
import os
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.config import MLFLOW_TRACKING_URI, MODEL_REGISTRY_NAME


def fetch_best_model():
    """
    It downloads the best model from the MLflow Registry and puts it in app/model_files/.
    This script should be run before the Docker build.
    """
    print(f"🔌 Connecting to MLflow Registry: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Find the Latest Version
    try:
        versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["None", "Staging", "Production"])
        if not versions:
            print(f"❌ Error: No models found for '{MODEL_REGISTRY_NAME}'!")
            return

        # Sort by version and get the latest
        latest_model = max(versions, key=lambda x: int(x.version))
        print(f"🎯 Latest Model Found: {MODEL_REGISTRY_NAME} - Version {latest_model.version}")

        # 2. Download Path
        dest_path = Path("app/model_files")
        if dest_path.exists():
            shutil.rmtree(dest_path)

        # 3. Download Model
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{latest_model.version}"
        print(f"⬇️ Downloading: {model_uri}")

        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dest_path))
        print(f"✅ The model has successfully downloaded to: {dest_path}")

    except Exception as e:
        print(f"❌ Download Error: {e}")


if __name__ == "__main__":
    fetch_best_model()