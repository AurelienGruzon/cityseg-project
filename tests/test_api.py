from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app


client = TestClient(app)


def make_test_image_bytes() -> bytes:
    img = Image.fromarray(np.zeros((256, 512, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_overlay(monkeypatch):
    class DummyModel:
        pass

    def fake_get_model():
        return DummyModel()

    def fake_predict_mask(model, x):
        return np.zeros((256, 512), dtype=np.int32)

    monkeypatch.setattr("src.api.app.get_model", fake_get_model)
    monkeypatch.setattr("src.api.app.predict_mask", fake_predict_mask)

    files = {
        "file": ("test.png", make_test_image_bytes(), "image/png")
    }

    response = client.post("/predict?mode=overlay", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_predict_mask(monkeypatch):
    class DummyModel:
        pass

    def fake_get_model():
        return DummyModel()

    def fake_predict_mask(model, x):
        return np.zeros((256, 512), dtype=np.int32)

    monkeypatch.setattr("src.api.app.get_model", fake_get_model)
    monkeypatch.setattr("src.api.app.predict_mask", fake_predict_mask)

    files = {
        "file": ("test.png", make_test_image_bytes(), "image/png")
    }

    response = client.post("/predict?mode=mask", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_predict_rejects_non_image():
    files = {
        "file": ("test.txt", b"hello", "text/plain")
    }

    response = client.post("/predict?mode=overlay", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image."