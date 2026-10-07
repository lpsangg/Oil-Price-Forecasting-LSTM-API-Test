import asyncio
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app, limiter, train_lock

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_limiter_and_lock():
    """Ensure rate limiter and lock are clean before and after every test."""
    limiter.reset()
    if train_lock.locked():
        train_lock.release()
    yield
    limiter.reset()
    if train_lock.locked():
        train_lock.release()


def test_payload_validation_data_length():
    """Test that sending data exceeding 1000 items is blocked with 422 Unprocessable Entity"""
    oversized_data = [70.0] * 1050
    response = client.post(
        "/predict",
        json={"data": oversized_data, "window": 60},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_payload_validation_window_bounds():
    """Test that invalid window bounds (< 3 or > 200) trigger 422 validation error"""
    # Test window < 3
    res_under = client.post(
        "/predict",
        json={"data": [70.0] * 60, "window": 2},
    )
    assert res_under.status_code == 422

    # Test window > 200
    res_over = client.post(
        "/predict",
        json={"data": [70.0] * 210, "window": 250},
    )
    assert res_over.status_code == 422


def test_payload_validation_train_parameters():
    """Test that invalid training hyperparameters trigger 422 validation error"""
    # Invalid epochs (< 1 or > 100)
    res_epochs = client.post(
        "/train",
        json={"epochs": 0, "window": 60},
    )
    assert res_epochs.status_code == 422

    # Invalid batch size (< 1 or > 512)
    res_batch = client.post(
        "/train",
        json={"batch_size": 0, "window": 60},
    )
    assert res_batch.status_code == 422


def test_concurrency_lock_train():
    """Test that concurrent training calls are rejected with 409 Conflict"""

    # Simulate an active training job holding the lock
    async def acquire_lock():
        await train_lock.acquire()

    asyncio.run(acquire_lock())

    try:
        response = client.post(
            "/train",
            json={
                "csv_path": "data/preprocess-QDL-OPEC.csv",
                "window": 60,
                "epochs": 1,
            },
        )
        assert response.status_code == 409
        data = response.json()
        assert "detail" in data
        assert "already in progress" in data["detail"]
    finally:
        if train_lock.locked():
            train_lock.release()


def test_rate_limiting_predict_429():
    """Test that rapid spamming on /predict triggers HTTP 429 Too Many Requests"""
    limiter.reset()
    dummy_data = [70.0] * 60

    got_429 = False
    for _ in range(65):
        res = client.post("/predict", json={"data": dummy_data, "window": 60})
        if res.status_code == 429:
            got_429 = True
            data = res.json()
            assert data["error"] == "Too Many Requests"
            assert "Rate limit exceeded" in data["detail"]
            assert "Retry-After" in res.headers
            break

    assert (
        got_429
    ), "Expected HTTP 429 Too Many Requests after exceeding 60 requests/minute"
