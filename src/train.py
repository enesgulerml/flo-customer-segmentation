# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_OUTPUT_PATH,
    CLUSTERS_OUTPUT_PATH,
    K_MEANS_RANGE,  # <-- YENİ
    RANDOM_STATE,
    K_MEANS_INIT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_REGISTRY_NAME
)
from src.data_processing import load_and_process_data


def train_kmeans_model():
    print(">>> Starting K-Means Auto-Tuning Process (v2.0)...")

    # 1. Load Data
    df = load_and_process_data()
    features_to_use = ["log_Recency", "log_Frequency", "log_Monetary", "log_Tenure"]
    X = df[features_to_use]

    # 2. Scale Data (StandardScaler is crucial)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. MLflow Setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- OPTIMIZATION LOOP (Kaos Avı) ---
    best_score = -1
    best_k = -1
    best_model = None
    best_labels = None

    print(f"Optimizing Cluster Count (Testing k={list(K_MEANS_RANGE)})...")

    # MLflow'da her denemeyi ayrı bir "Child Run" olarak değil,
    # tek bir büyük "Optimization Run" içinde loglayabiliriz veya
    # sadece kazananı loglarız. Temizlik için SADECE KAZANANI loglayacağız.

    for k in K_MEANS_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=K_MEANS_INIT)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        score = silhouette_score(X_scaled, labels)
        print(f"  k={k} -> Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            best_labels = labels

    print(f"\n🏆 WINNER: k={best_k} with Silhouette Score: {best_score:.4f}")

    # --- 4. Log the Winner to MLflow ---
    with mlflow.start_run(run_name=f"KMeans_Best_k{best_k}"):
        # Log Params
        mlflow.log_param("n_clusters", best_k)
        mlflow.log_param("features", features_to_use)
        mlflow.log_param("outlier_removal", "True (IQR 1.5)")

        # Log Metrics
        ch_score = calinski_harabasz_score(X_scaled, best_labels)
        mlflow.log_metric("silhouette_score", best_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)

        # 5. Save Model Locally
        # We need to save the SCALER + MODEL together as a pipeline
        # otherwise the API won't know how to scale new data!
        from sklearn.pipeline import Pipeline
        final_pipeline = Pipeline([
            ('scaler', scaler),
            ('kmeans', best_model)
        ])

        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, MODEL_OUTPUT_PATH)
        print(f"Model saved locally to: {MODEL_OUTPUT_PATH}")

        # 6. Register to MLflow
        print(f"Registering model to: {MODEL_REGISTRY_NAME}...")
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )

        # 7. Save Clusters
        df["cluster"] = best_labels + 1
        CLUSTERS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLUSTERS_OUTPUT_PATH, index=False)
        print(f"Customer Segments saved to: {CLUSTERS_OUTPUT_PATH}")

        print(">>> Training & Registration Finished Successfully.")


if __name__ == "__main__":
    train_kmeans_model()