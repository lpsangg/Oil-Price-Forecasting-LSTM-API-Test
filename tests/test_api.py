import pytest
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import after adding to path
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_without_data():
    """Test prediction with empty data"""
    response = client.post("/predict", json={"data": []})
    assert response.status_code in [400, 503]


def test_predict_with_insufficient_data():
    """Test prediction with insufficient data points"""
    response = client.post("/predict", json={
        "data": [100.0, 101.0, 102.0],
        "window": 60
    })
    assert response.status_code in [400, 503]


def test_predict_with_valid_data():
    """Test prediction with valid data"""
    # Generate dummy data
    data = list(np.random.uniform(50, 150, 60))

    response = client.post("/predict", json={
        "data": data,
        "window": 60
    })

    # May fail if model not loaded, but should not crash
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        result = response.json()
        assert "prediction" in result
        assert "timestamp" in result
        assert isinstance(result["prediction"], (int, float))


def test_predict_with_custom_window():
    """Test prediction with custom window size"""
    data = list(np.random.uniform(50, 150, 100))

    response = client.post("/predict", json={
        "data": data,
        "window": 30
    })

    assert response.status_code in [200, 503]


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    # May fail if model not loaded
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        result = response.json()
        assert "input_shape" in result
        assert "output_shape" in result
        assert "total_params" in result


def test_train_with_invalid_path():
    """Test training with invalid data path"""
    response = client.post("/train", json={
        "csv_path": "nonexistent_file.csv",
        "window": 10,
        "epochs": 1
    })

    assert response.status_code == 404


def test_api_docs_accessible():
    """Test that API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test OpenAPI schema is available"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema