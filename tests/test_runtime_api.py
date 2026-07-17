import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from shotseek.runtime.api import create_runtime_app


def _app(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runs" / "tests" / "runtime-api" / tmp_path.name
    shutil.rmtree(runtime, ignore_errors=True)
    return create_runtime_app(
        project_root=root,
        runtime_root=runtime,
        start_worker=False,
    )


def test_raw_upload_creates_one_idempotent_queued_job(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        first = client.post(
            "/api/v1/jobs?filename=demo.mp4",
            content=b"runtime-api-video",
            headers={"content-type": "video/mp4"},
        )
        assert first.status_code == 202
        payload = first.json()
        assert payload["job"]["state"] == "QUEUED"
        assert payload["job_reused"] is False
        assert payload["upload_created"] is True

        second = client.post(
            "/api/v1/jobs?filename=renamed.mp4",
            content=b"runtime-api-video",
            headers={"content-type": "video/mp4"},
        )
        assert second.status_code == 202
        duplicate = second.json()
        assert duplicate["job"]["job_id"] == payload["job"]["job_id"]
        assert duplicate["job_reused"] is True
        assert duplicate["upload_created"] is False
        assert client.get("/api/v1/jobs").json()["count"] == 1
        assert client.get("/api/v1/videos").json()["count"] == 1


def test_cancel_events_and_nonready_result_are_literal(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/jobs?filename=demo.mov",
            content=b"cancel-me",
        ).json()
        job_id = created["job"]["job_id"]
        assert client.get(f"/api/v1/jobs/{job_id}/result").status_code == 409
        cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["job"]["state"] == "CANCELLED"
        events = client.get(f"/api/v1/jobs/{job_id}/events?once=true")
        assert events.status_code == 200
        assert "event: job" in events.text
        assert '"state":"CANCELLED"' in events.text
        assert "event: end" in events.text


def test_upload_rejects_invalid_or_empty_body(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        invalid = client.post(
            "/api/v1/jobs?filename=payload.txt",
            content=b"not-video",
        )
        assert invalid.status_code == 400
        empty = client.post(
            "/api/v1/jobs?filename=empty.mp4",
            content=b"",
        )
        assert empty.status_code == 400
