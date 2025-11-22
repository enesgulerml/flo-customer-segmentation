from fastapi.testclient import TestClient
from app.main import app

# Create a TestClient using the FastAPI app instance
client = TestClient(app)


def test_health_check():
    """
    v5.0 - Unit Test for API Health Check.
    Verifies that the root endpoint returns 200 OK and correct service name.
    """
    response = client.get("/")

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] in ["Active", "Inactive"]
    assert "service" in json_data
    assert json_data["service"] == "FLO Segmentation API"

# Note: We skip deep prediction tests here because they require
# the model file to be present, which might not be true in a CI environment
# unless we run 'fetch_model.py' first.