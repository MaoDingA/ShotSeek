from fractions import Fraction
from pathlib import Path

from shotseek.m1a import run_m1a, verify_m1a
from shotseek.media.schema import Shot, VideoContract
from shotseek.media.shots import build_shot_grid
from shotseek.media.timebase import frame_to_ms, ms_to_ceil_frame, ms_to_floor_frame


def test_exact_timebase_conversions() -> None:
    fps = Fraction(24, 1)
    assert frame_to_ms(75, fps) == 3125
    assert ms_to_floor_frame(3124, fps) == 74
    assert ms_to_ceil_frame(3126, fps) == 76


def test_complete_grid_without_boundaries() -> None:
    video = VideoContract(
        path="sample.mp4",
        sha256="a" * 64,
        bytes=1,
        width=1,
        height=1,
        duration_ms=1000,
        frame_count=24,
        fps_num=24,
        fps_den=1,
        time_base_num=1,
        time_base_den=12288,
        video_codec="h264",
        cfr=True,
    )
    shots = build_shot_grid(video, [])
    assert shots == [
        Shot(
            shot_id="shot_0001",
            start_frame=0,
            end_frame=24,
            start_ms=0,
            end_ms=1000,
            duration_frames=24,
        )
    ]


def test_golden_m1a_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    video = root / "samples" / "golden.mp4"
    if not video.is_file():
        return
    first = run_m1a(
        project_root=root,
        video_path=video,
        output_dir=root / "runs" / "tests" / "m1a-first",
    )
    second = run_m1a(
        project_root=root,
        video_path=video,
        output_dir=root / "runs" / "tests" / "m1a-second",
    )
    assert verify_m1a(first)["status"] == "pass"
    assert verify_m1a(second)["status"] == "pass"
    import json

    first_report = json.loads((first / "run_report.json").read_text())
    second_report = json.loads((second / "run_report.json").read_text())
    assert (
        first_report["deterministic_payload_sha256"]
        == second_report["deterministic_payload_sha256"]
    )
