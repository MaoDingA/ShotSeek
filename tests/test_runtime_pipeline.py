import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shotseek.runtime import JobState, RuntimePaths, RuntimeRegistry, RuntimeWorker, store_upload
from shotseek.runtime.api import create_runtime_app
from shotseek.runtime.pipeline import PipelineSettings, ProductionPipeline


def _generate_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_fixture_pipeline_builds_real_media_artifacts_and_search_index(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_root = root / "runs" / "tests" / "runtime-pipeline" / tmp_path.name
    shutil.rmtree(runtime_root, ignore_errors=True)
    paths = RuntimePaths(root, runtime_root)
    source = runtime_root / "source" / "three-seconds.mp4"
    _generate_video(source)
    with source.open("rb") as handle:
        stored = store_upload(paths, handle, source.name)
    registry = RuntimeRegistry(paths.registry)
    video, _ = registry.register_video(
        sha256=stored.sha256,
        original_filename=stored.original_filename,
        source_path=str(stored.path.relative_to(root)),
        bytes=stored.bytes,
    )
    job = registry.create_job(video.video_id)
    registry.transition(job.job_id, JobState.QUEUED)
    pipeline = ProductionPipeline(
        paths=paths,
        registry=registry,
        settings=PipelineSettings(mode="fixture"),
    )
    result = RuntimeWorker(registry, pipeline, max_retries=0).run_once()
    assert result is not None
    assert result.state == JobState.READY
    finished_video = registry.get_video(video.video_id)
    assert finished_video.status == "READY"
    assert finished_video.scene_count >= 1
    assert finished_video.search_db_path
    assert (root / finished_video.search_db_path).is_file()

    video_root = paths.video_root(video.video_id)
    assert (video_root / "media" / "proxy.mp4").is_file()
    assert (video_root / "media" / "audio.mp3").is_file()
    assert (video_root / "timeline" / "shots.json").is_file()
    assert (video_root / "timeline" / "scenes.json").is_file()
    manifest = json.loads(
        (video_root / "chunks" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["chunks"]) == 1
    assert all(item["bytes"] < 110 * 1024 * 1024 for item in manifest["chunks"])
    artifacts = registry.list_artifacts(video.video_id)
    assert {item.kind for item in artifacts} >= {
        "proxy_video",
        "audio",
        "shot_grid",
        "visual_events",
        "utterances",
        "scenes",
        "search_index",
    }

    app = create_runtime_app(
        project_root=root,
        runtime_root=runtime_root,
        start_worker=False,
    )
    with TestClient(app) as client:
        scenes = client.get(f"/api/v1/videos/{video.video_id}/scenes")
        assert scenes.status_code == 200
        assert scenes.json()["count"] >= 1
        search = client.post(
            f"/api/v1/videos/{video.video_id}/search",
            json={
                "query": "red jacket",
                "planner_mode": "rule",
                "verifier_mode": "rule",
            },
        )
        assert search.status_code == 200
        assert search.json()["hits"]
        scene_id = search.json()["hits"][0]["candidate"]["scene_id"]
        evidence = client.get(
            f"/api/v1/videos/{video.video_id}/scenes/{scene_id}/evidence"
        )
        assert evidence.status_code == 200
        assert evidence.json()["boundary"]["strategy"] == "shot_first"
