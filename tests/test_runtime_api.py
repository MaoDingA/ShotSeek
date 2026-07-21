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


def test_built_workbench_is_served_from_runtime(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["expires"] == "0"
        assert "ShotSeek" in response.text
        asset = next(
            path.name for path in (Path(__file__).resolve().parents[1] / "shotseek" / "runtime" / "static" / "assets").glob("*.css")
        )
        assert client.get(f"/assets/{asset}").status_code == 200
        favicon = client.get("/favicon.svg")
        assert favicon.status_code == 200
        assert favicon.headers["content-type"].startswith("image/svg+xml")


def test_ready_video_exports_selected_scenes_in_all_delivery_formats(
    tmp_path: Path,
) -> None:
    import json
    import sqlite3

    app = _app(tmp_path)
    root = Path(__file__).resolve().parents[1]
    database = app.state.runtime_paths.root / "ready-search.sqlite3"
    scene = {
        "scene_id": "scene_0001",
        "start_ms": 1_000,
        "end_ms": 2_000,
        "start_frame": 25,
        "end_frame": 50,
        "summary": "人物进入房间",
        "confidence": 0.9,
        "evidence_refs": [{"kind": "visual", "evidence_id": "event_1"}],
    }
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE scene("
            "scene_id TEXT, start_ms INTEGER, scene_json TEXT, dialogue TEXT)"
        )
        connection.execute(
            "INSERT INTO scene VALUES (?, ?, ?, ?)",
            ("scene_0001", 1_000, json.dumps(scene), ""),
        )
    registry = app.state.runtime_registry
    video, _ = registry.register_video(
        sha256="1" * 64,
        original_filename="demo.mp4",
        source_path=str(database.relative_to(root)),
        bytes=1,
    )
    registry.update_video(
        video.video_id,
        status="READY",
        fps=25.0,
        search_db_path=str(database.relative_to(root)),
    )

    with TestClient(app) as client:
        for format in ("json", "srt", "xml", "edl"):
            response = client.get(
                f"/api/v1/videos/{video.video_id}/export",
                params={"format": format, "scene_id": "scene_0001"},
            )
            assert response.status_code == 200
            assert response.headers["x-shotseek-scene-count"] == "1"
            assert f".{format}" in response.headers["content-disposition"]
        missing = client.get(
            f"/api/v1/videos/{video.video_id}/export",
            params={"format": "json", "scene_id": "scene_9999"},
        )
        assert missing.status_code == 404
