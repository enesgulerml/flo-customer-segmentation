# v3.1 API Server Dockerfile (FLO Project)
# Strategy: Python 3.10 Slim + Pip (Optimized & Chaos-Free)

FROM python:3.10-slim-bookworm

WORKDIR /app

# 1. Install System Dependencies
# 'curl' is needed for healthchecks or downloads, 'gcc'/'build-essential' for libraries.
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy & Install Dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3. Install Project Package
COPY setup.py setup.py
RUN pip install .

# 4. Copy Application Code
# This copies 'src/', 'app/', and importantly 'app/model_files/' (if fetched)
COPY . .

# 5. Expose Port & Start Server
EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]