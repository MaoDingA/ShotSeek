import json
import xml.etree.ElementTree as ET

import pytest

from shotseek.export.delivery import render_export


SCENES = [
    {
        "scene_id": "scene_0002",
        "start_ms": 2_000,
        "end_ms": 3_500,
        "start_frame": 50,
        "end_frame": 88,
        "summary": "角色离开房间",
        "dialogue": "",
        "confidence": 0.9,
        "evidence_refs": [{"kind": "visual", "evidence_id": "visual_2"}],
    },
    {
        "scene_id": "scene_0001",
        "start_ms": 0,
        "end_ms": 2_000,
        "start_frame": 0,
        "end_frame": 50,
        "summary": "角色说话",
        "dialogue": "这件事不是意外",
        "confidence": 0.95,
        "evidence_refs": [{"kind": "dialogue", "evidence_id": "utt_1"}],
    },
]


@pytest.mark.parametrize("format", ["json", "srt", "xml", "edl"])
def test_delivery_formats_are_deterministic_and_time_ordered(format: str) -> None:
    first = render_export(
        format,
        SCENES,
        video_id="video_demo",
        source_name="样片.mp4",
        fps=25.0,
    )
    second = render_export(
        format,
        SCENES,
        video_id="video_demo",
        source_name="样片.mp4",
        fps=25.0,
    )
    assert first == second
    assert first.extension == format
    text = first.content.decode()

    if format == "json":
        payload = json.loads(text)
        assert payload["scene_count"] == 2
        assert payload["scenes"][0]["scene_id"] == "scene_0001"
    elif format == "srt":
        assert "00:00:00,000 --> 00:00:02,000" in text
        assert "这件事不是意外" in text
    elif format == "xml":
        root = ET.fromstring(first.content)
        assert [node.attrib["id"] for node in root.findall("scene")] == [
            "scene_0001",
            "scene_0002",
        ]
    else:
        assert text.index("scene_0001") < text.index("scene_0002")
        assert "FCM: NON-DROP FRAME" in text
        assert "00:00:00:00 00:00:02:00 01:00:00:00 01:00:02:00" in text


def test_edl_rejects_non_integer_frame_rate() -> None:
    with pytest.raises(ValueError, match="requires 24, 25 or 30 fps"):
        render_export(
            "edl",
            SCENES,
            video_id="video_demo",
            source_name="demo.mp4",
            fps=23.976,
        )
