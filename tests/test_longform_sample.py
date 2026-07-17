from pathlib import Path

from scripts.build_longform_sample import (
    SOURCES,
    encode_command,
    filter_graph,
)


def test_longform_sources_are_frozen_and_attributed() -> None:
    assert [source["title"] for source in SOURCES] == [
        "Tears of Steel",
        "Sintel",
        "Big Buck Bunny",
    ]
    assert len({source["sha256"] for source in SOURCES}) == 3
    assert all(len(source["sha256"]) == 64 for source in SOURCES)
    assert all(source["license"] == "CC BY 3.0" for source in SOURCES)
    assert all(source["license_url"].startswith("https://") for source in SOURCES)


def test_longform_filter_normalizes_and_concatenates_every_source() -> None:
    graph = filter_graph(3)
    assert graph.count("]setpts=PTS-STARTPTS") == 3
    assert graph.count("]asetpts=PTS-STARTPTS") == 3
    assert "concat=n=3:v=1:a=1[v][a]" in graph
    command = encode_command(
        inputs=[Path("one.mp4"), Path("two.mp4"), Path("three.mp4")],
        output=Path("longform.mp4"),
        video_encoder="h264_nvenc",
    )
    assert command.count("-i") == 3
    assert "h264_nvenc" in command
    assert command[-1] == "longform.mp4"
