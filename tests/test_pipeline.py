"""
Tests for the training pipeline and serving API.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from train import train_model
from app import app as flask_app


def test_train_model_returns_good_accuracy():
    model, accuracy, f1 = train_model()
    assert accuracy > 0.8  # Iris is an easy dataset — should comfortably clear this
    assert f1 > 0.8


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_valid_input(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "confidence" in data


def test_predict_missing_features(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [1, 2, 3]})
    assert response.status_code == 400
