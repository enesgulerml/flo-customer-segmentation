# End-to-End FLO Customer Segmentation (v5.0)

This project implements a production-grade **Unsupervised Learning** pipeline to segment customers based on their Omnichannel (Online + Offline) shopping behavior.

It moves beyond simple notebooks by implementing a robust **MLOps** architecture:
* **v1.0: Data Engineering:** Omnichannel integration, Outlier Removal (IQR), and RFM Feature Engineering.
* **v2.0: Model Training:** K-Means clustering with Auto-Tuning (Elbow Method) and MLflow Tracking.
* **v3.1: API Serving:** A Dockerized **FastAPI** service using an "Embedded Model" strategy (CI/CD simulation).
* **v4.0: Dashboard:** An interactive **Streamlit** client for real-time segmentation.
* **v5.0: Testing:** A full `pytest` suite for quality assurance.

---

## 🚀 Project Structure



---

## 🛠️ Installation & Setup

Follow these steps to set up the project environment on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/enesgulerml/flo-customer-segmentation.git
cd flo-customer-segmentation
```

### 2. Setup Environment (v5 Strategy - Pip)
We use Conda for Python management and Pip for package management to avoid solver issues.
```bash
conda create -n flo-segmentation python=3.10 -y
conda activate flo-segmentation
pip install -r requirements.txt
pip install -e .
```

### 3. Add Raw Data
Place your flo_data.csv file into the data/raw/ directory. (Note: Data is not tracked by Git due to privacy/size).

## ⚡ Workflow & Usage
### Phase 1: Data Engineering & Training (v1.0 - v2.0)
This step processes raw data, calculates RFM metrics, removes outliers, applies log transformation, and auto-tunes K-Means (testing k=3 to 10) to find the best cluster count.
1. **Start MLflow Server (Terminal 1):** Keep this terminal open.
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000
```
2. **Run Training (Terminal 2):**
```bash
python -m src.train
```
Check results at http://127.0.0.1:5000. The winning model is automatically registered.

### Phase 2: Build the API Image (v3.1)
We use an Embedded Model Strategy. The model is fetched from the Registry and baked into the Docker image at build time, making the container self-sufficient.

1. **Fetch Model:** Downloads the latest production model to app/model_files/.
```bash
python -m src.fetch_model
```

2. **Build Docker Image:**
```bash
docker build -t flo-api:v1 .
```

### Phase 3: Serve & Demo (v4.0)
Now run the microservices architecture.

1. **Run API Motor (Terminal 2):** Runs the container on port 8005. No external volumes required.
```bash
docker run -d --rm -p 8005:80 flo-api:v1
```
Verify API Docs: http://localhost:8005/docs

2. **Run Dashboard (Terminal 3):** Launches the Streamlit interface to interact with the model.
```bash
streamlit run dashboard/app.py
```
Open Dashboard: http://localhost:8501

### Phase 4: Testing (v5.0)
Run the automated test suite to ensure system integrity.
```bash
python -m pytest
```