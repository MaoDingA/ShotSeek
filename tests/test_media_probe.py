import json
import subprocess
from pathlib import Path

import pytest

from shotseek.media.probe import probe_video_contract


def _ffprobe_result(
    *, stream_duration: str, container_duration: str
) -> subprocess.CompletedProcess[str]:
    payload = {
        "streams": [
            {
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "25/1",
                "r_frame_rate": "25/1",
                "time_base": "1/12800",
                "duration": stream_duration,
                "duration_ts": "1868800",
                "nb_frames": "3650",
            }
        ],
        "format": {"duration": container_duration, "size": "4"},
    }
    return subprocess.CompletedProcess(
        [], 0, stdout=json.dumps(payload), stderr=""
    )


def _media_file(root: Path, tmp_path: Path, name: str) -> Path:
    media = (
        root
        / "runs"
        / "tests"
        / "probe-contract"
        / tmp_path.name
        / name
    )
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"test")
    return media


def test_probe_uses_video_timeline_when_audio_padding_extends_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    media = _media_file(root, tmp_path, "audio-padding.mp4")
    monkeypatch.setattr(
        "shotseek.media.probe.subprocess.run",
        lambda *args, **kwargs: _ffprobe_result(
            stream_duration="146.000000",
            container_duration="146.048000",
        ),
    )

    contract = probe_video_contract(root, media)

    assert contract.duration_ms == 146_000
    assert contract.frame_count == 3650
    assert contract.fps_num == 25


def test_probe_rejects_inconsistent_video_stream_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    media = _media_file(root, tmp_path, "bad-stream.mp4")
    monkeypatch.setattr(
        "shotseek.media.probe.subprocess.run",
        lambda *args, **kwargs: _ffprobe_result(
            stream_duration="145.000000",
            container_duration="146.048000",
        ),
    )

    with pytest.raises(ValueError, match="video stream duration"):
        probe_video_contract(root, media)
