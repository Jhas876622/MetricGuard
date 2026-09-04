"""
MetricGuard - API Endpoint Test Suite
=======================================
Tests every FastAPI route using Starlette's TestClient (synchronous,
no running server needed).

Strategy:
  - The pipeline is MOCKED in every test — we patch _run_pipeline_background
    so no actual embedding/LLM work happens. Tests stay fast (<1s total).
  - results.json is written to a real tmp directory so file-system logic
    (path exists / does not exist) is exercised against real paths.
  - dashboard.html is patched to a minimal valid HTML string.

Run with:
    pytest tests/test_app.py -v
"""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Minimal fake payload — what a real pipeline run would write to results.json
# ---------------------------------------------------------------------------
FAKE_PAYLOAD = {
    "kpis": {
        "total_definitions_scanned": 12,
        "conflicting_definitions_found": 10,
        "conflict_groups": 4,
        "teams_affected": 6,
        "pct_definitions_in_conflict": 83.3,
        "highest_trust_risk": 100,
    },
    "conflicts": [
        {
            "names": ["monthly_revenue", "revenue_monthly"],
            "teams": ["Finance", "Sales"],
            "metric_ids": ["m01", "m02"],
            "conflicts": ["Filter logic differs"],
            "trust_risk": 70,
            "avg_similarity": 0.95,
            "definitions": [
                {
                    "team": "Finance",
                    "name": "monthly_revenue",
                    "description": "Completed orders only.",
                    "sql": "SELECT SUM(amount) FROM orders WHERE status='completed'",
                }
            ],
        }
    ],
}

FAKE_DASHBOARD_HTML = "<html><body><h1>MetricGuard</h1></body></html>"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def results_file(tmp_path):
    """Write fake results.json to a tmp dir and return its path."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps(FAKE_PAYLOAD))
    return p


@pytest.fixture()
def dashboard_file(tmp_path):
    """Write a minimal dashboard.html to a tmp dir and return its path."""
    p = tmp_path / "dashboard.html"
    p.write_text(FAKE_DASHBOARD_HTML)
    return p


@pytest.fixture()
def client(results_file, dashboard_file):
    """
    TestClient with RESULTS_PATH and DASHBOARD_PATH patched to tmp files,
    pipeline state reset to idle, and lifespan suppressed.
    """
    import app as app_module

    with (
        patch.object(app_module, "RESULTS_PATH", results_file),
        patch.object(app_module, "DASHBOARD_PATH", dashboard_file),
    ):
        app_module._pipeline_running.clear()
        app_module._last_run_time = None
        app_module._last_run_error = None

        # Suppress lifespan so no background pipeline starts during tests
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop_lifespan(app):
            yield

        app_module.app.router.lifespan_context = noop_lifespan
        yield TestClient(app_module.app, raise_server_exceptions=True)


@pytest.fixture()
def client_no_results(tmp_path, dashboard_file):
    """Client where results.json does NOT exist yet."""
    import app as app_module

    missing_path = tmp_path / "results.json"   # intentionally not written

    with (
        patch.object(app_module, "RESULTS_PATH", missing_path),
        patch.object(app_module, "DASHBOARD_PATH", dashboard_file),
    ):
        app_module._pipeline_running.clear()
        app_module._last_run_time = None
        app_module._last_run_error = None

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop_lifespan(app):
            yield

        app_module.app.router.lifespan_context = noop_lifespan
        yield TestClient(app_module.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        res = client.get("/health")
        assert res.status_code == 200

    def test_status_field_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_pipeline_not_running_when_idle(self, client):
        data = client.get("/health").json()
        assert data["pipeline_running"] is False

    def test_results_exist_true_when_file_present(self, client):
        data = client.get("/health").json()
        assert data["results_exist"] is True

    def test_results_exist_false_when_file_missing(self, client_no_results):
        data = client_no_results.get("/health").json()
        assert data["results_exist"] is False

    def test_pipeline_running_reflected_in_health(self, client):
        import app as app_module
        app_module._pipeline_running.set()
        try:
            data = client.get("/health").json()
            assert data["pipeline_running"] is True
        finally:
            app_module._pipeline_running.clear()


# ---------------------------------------------------------------------------
# GET /results
# ---------------------------------------------------------------------------

class TestResultsEndpoint:
    def test_returns_200_when_results_exist(self, client):
        assert client.get("/results").status_code == 200

    def test_payload_contains_kpis(self, client):
        data = client.get("/results").json()
        assert "kpis" in data
        assert data["kpis"]["total_definitions_scanned"] == 12

    def test_payload_contains_conflicts(self, client):
        data = client.get("/results").json()
        assert "conflicts" in data
        assert len(data["conflicts"]) == 1

    def test_meta_field_present(self, client):
        data = client.get("/results").json()
        assert "meta" in data
        assert "pipeline_running" in data["meta"]
        assert "last_run_time" in data["meta"]

    def test_404_when_no_results(self, client_no_results):
        assert client_no_results.get("/results").status_code == 404

    def test_loading_state_when_pipeline_running_no_results(self, client_no_results):
        import app as app_module
        app_module._pipeline_running.set()
        try:
            data = client_no_results.get("/results").json()
            assert data["loading"] is True
        finally:
            app_module._pipeline_running.clear()

    def test_conflict_names_present(self, client):
        data = client.get("/results").json()
        names = data["conflicts"][0]["names"]
        assert "monthly_revenue" in names

    def test_trust_risk_in_conflict(self, client):
        data = client.get("/results").json()
        assert data["conflicts"][0]["trust_risk"] == 70


# ---------------------------------------------------------------------------
# GET / (dashboard)
# ---------------------------------------------------------------------------

class TestDashboardEndpoint:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_content_type_is_html(self, client):
        res = client.get("/")
        assert "text/html" in res.headers["content-type"]

    def test_dashboard_content_served(self, client):
        res = client.get("/")
        assert "MetricGuard" in res.text

    def test_404_when_dashboard_missing(self, tmp_path, results_file):
        import app as app_module
        from contextlib import asynccontextmanager
        missing_html = tmp_path / "dashboard.html"   # not written
        with (
            patch.object(app_module, "RESULTS_PATH", results_file),
            patch.object(app_module, "DASHBOARD_PATH", missing_html),
        ):
            @asynccontextmanager
            async def noop_lifespan(app):
                yield
            app_module.app.router.lifespan_context = noop_lifespan
            c = TestClient(app_module.app, raise_server_exceptions=False)
            assert c.get("/").status_code == 404


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------

class TestRunEndpoint:
    def test_returns_202_when_idle(self, client):
        with patch("app._run_pipeline_background"):
            res = client.post("/run?use_llm=false")
        assert res.status_code == 202

    def test_response_contains_message(self, client):
        with patch("app._run_pipeline_background"):
            data = client.post("/run?use_llm=false").json()
        assert "message" in data

    def test_returns_202_already_running(self, client):
        """When pipeline is already running, /run must return 202 with a
        'already running' message without starting a second thread."""
        import app as app_module
        app_module._pipeline_running.set()
        try:
            res = client.post("/run?use_llm=false")
            assert res.status_code == 202
            assert "already running" in res.json()["message"].lower()
        finally:
            app_module._pipeline_running.clear()

    def test_use_llm_false_accepted(self, client):
        with patch("app._run_pipeline_background"):
            res = client.post("/run?use_llm=false")
        assert res.status_code == 202
        assert res.json()["use_llm"] is False

    def test_use_llm_true_default(self, client):
        with patch("app._run_pipeline_background"):
            res = client.post("/run")
        assert res.status_code == 202
        assert res.json()["use_llm"] is True
