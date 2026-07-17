from pathlib import Path

from fastapi.testclient import TestClient

from shotseek.api import create_app

ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "runs" / "m1c" / "latest" / "search.sqlite3"
MANIFEST = ROOT / "runs" / "m1a" / "latest" / "manifest.json"


def _client() -> TestClient | None:
    if not DATABASE.is_file():
        return None
    app = create_app(
        database_path=DATABASE,
        manifest_path=MANIFEST,
        trace_dir=ROOT / "runs" / "tests" / "m2c-traces",
    )
    return TestClient(app)


def test_read_only_api_surface_and_not_found_contracts() -> None:
    client = _client()
    if client is None:
        return
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["network_enabled"] is False

    videos = client.get("/videos").json()
    assert videos["count"] == 1
    video_id = videos["items"][0]["video_id"]
    assert client.get(f"/videos/{video_id}").status_code == 200
    assert client.get(f"/videos/{video_id}/scenes").json()["count"] == 23
    assert client.get("/scenes/scene_0016").status_code == 200
    assert client.get("/scenes/missing").status_code == 404
    assert client.get("/videos/missing").status_code == 404


def test_search_trace_and_metrics_round_trip() -> None:
    client = _client()
    if client is None:
        return
    response = client.post(
        "/search",
        json={
            "query": "mechanical ocular implant",
            "planner_mode": "rule",
            "verifier_mode": "rule",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["hits"][0]["candidate"]["scene_id"] == "scene_0016"
    trace_id = body["trace"]["trace_id"]
    assert client.get(f"/traces/{trace_id}").status_code == 200

    metrics = client.get("/metrics").json()
    assert metrics["search_count"] == 1
    assert metrics["status_counts"]["RULE"] == 1
    assert metrics["query_p95_ms"] >= 0.0


def test_api_rejects_unknown_or_invalid_request_fields() -> None:
    client = _client()
    if client is None:
        return
    assert client.post(
        "/search", json={"query": "", "unknown": True}
    ).status_code == 422
