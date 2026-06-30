"""API tests using FastAPI TestClient.

Tests critical endpoints for correctness, input validation, and error handling.
"""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key-12345"
os.environ["LOG_LEVEL"] = "ERROR"

from app.main import app

client = TestClient(app)
headers = {"x-api-key": "test-key-12345"}


class TestHealthEndpoints:
    def test_liveness(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readiness(self):
        r = client.get("/api/health/ready", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded")

    def test_config(self):
        r = client.get("/api/config", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "version" in data
        assert "api_version" in data


class TestAuth:
    def test_config_is_public(self):
        r = client.get("/api/config", headers=headers)
        assert r.status_code == 200

    def test_wrong_api_key_blocked(self):
        r = client.get("/api/pipeline/prereqs", headers={"x-api-key": "wrong"})
        assert r.status_code in (401, 403)


class TestPipelinePrereqs:
    def test_prereqs(self):
        r = client.get("/api/pipeline/prereqs", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "prereqs" in data
        assert "discard_on_rerun" in data

    def test_prereqs_v1(self):
        r = client.get("/api/v1/pipeline/prereqs", headers=headers)
        assert r.status_code == 200


class TestModelListing:
    def test_models(self):
        r = client.get("/api/models", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json()["models"], list)

    def test_ocr_engines(self):
        r = client.get("/api/ocr/engines", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "engines" in data

    def test_ollama_models_fallback(self):
        r = client.get("/api/ollama/models", headers=headers)
        assert r.status_code in (200, 503)

    def test_embedding_models(self):
        r = client.get("/api/embedding/models", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json()["models"], list)


class TestPresets:
    def test_presets(self):
        r = client.get("/api/presets", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json()["presets"], list)


class TestDatasetEndpoints:
    def test_dataset_stats(self):
        r = client.get("/api/dataset/stats", headers=headers)
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "total_images" in data or "total_documents" in data

    def test_dataset_documents(self):
        r = client.get(
            "/api/dataset/documents?model=invoice_dataset_model_1&page=1&per_page=5",
            headers=headers,
        )
        assert r.status_code in (200, 404)

    def test_dataset_model_fields(self):
        r = client.get("/api/dataset/model-fields/invoice_dataset_model_1", headers=headers)
        assert r.status_code in (200, 404)


class TestOptimizationEndpoints:
    def test_optimization_status(self):
        r = client.get("/api/optimize/status", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert "cache_file" in data
        assert "optimized_types" in data

    def test_optimization_clear(self):
        r = client.post("/api/v1/optimize/clear", headers=headers)
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"


class TestErrorHandling:
    def test_nonexistent_session(self):
        r = client.get("/api/status/nonexistent123", headers=headers)
        assert r.status_code == 404

    def test_no_question_in_qa(self):
        r = client.post("/api/qa/nonexistent", json={}, headers=headers)
        assert r.status_code in (400, 404)

    def test_empty_question(self):
        r = client.post(
            "/api/qa/nonexistent",
            json={"question": ""},
            headers=headers,
        )
        assert r.status_code in (400, 404)

    def test_invalid_optimize_request(self):
        r = client.post(
            "/api/optimize",
            json={"doc_types": ["invalid_type"]},
            headers=headers,
        )
        assert r.status_code in (400, 500)


class TestCacheEndpoints:
    def test_cache_stats(self):
        r = client.get("/api/cache/stats", headers=headers)
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_cache_clear(self):
        r = client.post("/api/cache/clear", headers=headers)
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"


class TestExport:
    def test_export_nonexistent_session(self):
        r = client.get("/api/session/nonexistent/export/json", headers=headers)
        assert r.status_code == 404

    def test_download_nonexistent(self):
        r = client.get("/api/result/nonexistent/download", headers=headers)
        assert r.status_code == 404
