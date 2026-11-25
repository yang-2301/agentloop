"""API tests: run, list, detail, tools, health."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client():
    return TestClient(app)


class TestHealthAndTools:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["llm_mode"] in ("mock", "openai")

    def test_tools_listed(self, client):
        r = client.get("/api/tools")
        assert r.status_code == 200
        names = [t["name"] for t in r.json()["tools"]]
        assert {"calculator", "knowledge_search", "http_get", "now"} <= set(names)


class TestRunEndpoints:
    def test_run_mock_task(self, client):
        r = client.post("/api/run", json={"task": "Calculate 15% of 2400"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["answer"]
        assert "360" in body["answer"]
        assert body["run_id"]

    def test_run_detail_roundtrip(self, client):
        created = client.post("/api/run", json={"task": "Search docs for refund policy"}).json()
        r = client.get(f"/api/runs/{created['run_id']}")
        assert r.status_code == 200
        detail = r.json()
        assert detail["task"] == "Search docs for refund policy"
        assert detail["steps"]

    def test_run_detail_404(self, client):
        assert client.get("/api/runs/does-not-exist").status_code == 404

    def test_runs_listed(self, client):
        client.post("/api/run", json={"task": "Calculate 1+1"})
        r = client.get("/api/runs")
        assert r.status_code == 200
        assert any(item["task"] == "Calculate 1+1" for item in r.json())

    def test_run_validation_empty_task(self, client):
        assert client.post("/api/run", json={"task": ""}).status_code == 422

    def test_run_max_steps_bounds(self, client):
        assert client.post("/api/run", json={"task": "x", "max_steps": 0}).status_code == 422
        assert client.post("/api/run", json={"task": "x", "max_steps": 99}).status_code == 422


class TestRunsHistoryAndToolSchema:
    def test_tools_expose_json_schema(self, client):
        r = client.get("/api/tools")
        tools = {t["name"]: t for t in r.json()["tools"]}
        assert "parameters" in tools["calculator"]
        assert tools["calculator"]["parameters"]["properties"]["expression"]["type"] == "string"

    def test_runs_sorted_newest_first_with_timestamp(self, client):
        first = client.post("/api/run", json={"task": "Calculate 1+1"}).json()
        second = client.post("/api/run", json={"task": "Calculate 2+2"}).json()
        runs = client.get("/api/runs").json()
        assert runs[0]["run_id"] == second["run_id"]
        assert runs[0]["created_at"] >= runs[-1]["created_at"]

    def test_run_detail_includes_created_at(self, client):
        created = client.post("/api/run", json={"task": "Calculate 3+3"}).json()
        detail = client.get(f"/api/runs/{created['run_id']}").json()
        assert isinstance(detail["created_at"], float)

    def test_max_steps_guard_is_honored(self, client):
        # two-tool trajectory needs >1 step; capping at 1 must stop the run
        r = client.post("/api/run", json={
            "task": "Search our docs: how much does the team plan cost per year for 10 users?",
            "max_steps": 1,
        })
        assert r.json()["status"] == "max_steps"
        assert r.json()["answer"] is None
