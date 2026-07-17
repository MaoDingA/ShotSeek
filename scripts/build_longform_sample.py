#!/usr/bin/env python3
"""Build the reproducible 36-minute Blender Open Movies long-form sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shotseek.m0 import ensure_within_project

SOURCES = [
    {
        "title": "Tears of Steel",
        "path": "samples/tears_of_steel_720p.mov",
        "download_url": "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        "license_url": "https://mango.blender.org/about/",
        "license": "CC BY 3.0",
        "sha256": "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f",
    },
    {
        "title": "Sintel",
        "path": "samples/sintel_720p.mkv",
        "download_url": "https://download.blender.org/demo/movies/Sintel.2010.720p.mkv",
        "license_url": "https://durian.blender.org/sharing/",
        "license": "CC BY 3.0",
        "sha256": "60cff51761641626e82eeb4e1c248c471375b2536bb1089f49825b7fb58d8723",
    },
    {
        "title": "Big Buck Bunny",
        "path": "samples/big_buck_bunny_720p.mov",
        "download_url": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov",
        "license_url": "https://video.blender.org/w/pAQiVCgv2CsLg79KKXUoMw",
        "license": "CC BY 3.0",
        "sha256": "45c8bafeb9a53df7f491198d2e71529701bcf1cd51805782089fac1d32869f9b",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def filter_graph(count: int) -> str:
    chains: list[str] = []
    concat_inputs: list[str] = []
    for index in range(count):
        chains.append(
            f"[{index}:v:0]setpts=PTS-STARTPTS,fps=25,"
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,format=yuv420p[v{index}]"
        )
        chains.append(
            f"[{index}:a:0]asetpts=PTS-STARTPTS,"
            "aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,aresample=async=1:first_pts=0[a{index}]"
        )
        concat_inputs.append(f"[v{index}][a{index}]")
    chains.append(
        "".join(concat_inputs) + f"concat=n={count}:v=1:a=1[v][a]"
    )
    return ";".join(chains)


def encode_command(
    *,
    inputs: list[Path],
    output: Path,
    video_encoder: str,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-y"]
    for source in inputs:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex",
            filter_graph(len(inputs)),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            video_encoder,
        ]
    )
    if video_encoder == "h264_nvenc":
        command.extend(
            [
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-b:v",
                "3M",
                "-maxrate",
                "4M",
                "-bufsize",
                "8M",
            ]
        )
    else:
        command.extend(
            [
                "-preset",
                "medium",
                "-crf",
                "23",
                "-maxrate",
                "4M",
                "-bufsize",
                "8M",
            ]
        )
    command.extend(
        [
            "-g",
            "50",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return command


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("samples/shotseek_longform_blender_open_movies_v1.mp4"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = ensure_within_project(root, root / args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.building.mp4")
    inputs: list[Path] = []
    source_records: list[dict[str, Any]] = []
    offset_ms = 0

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe are required")
    for source in SOURCES:
        path = ensure_within_project(root, root / source["path"])
        if not path.is_file():
            raise SystemExit(f"missing source: {path.relative_to(root)}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != source["sha256"]:
            raise SystemExit(f"source checksum mismatch: {path.relative_to(root)}")
        media = probe(path)
        duration_ms = round(float(media["format"]["duration"]) * 1_000)
        source_records.append(
            {
                **source,
                "duration_ms": duration_ms,
                "source_start_ms": offset_ms,
                "source_end_ms": offset_ms + duration_ms,
                "bytes": path.stat().st_size,
            }
        )
        offset_ms += duration_ms
        inputs.append(path)

    temporary.unlink(missing_ok=True)
    encoder = "h264_nvenc"
    completed = subprocess.run(
        encode_command(
            inputs=inputs,
            output=temporary,
            video_encoder=encoder,
        ),
        check=False,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        encoder = "libx264"
        subprocess.run(
            encode_command(
                inputs=inputs,
                output=temporary,
                video_encoder=encoder,
            ),
            check=True,
        )
    temporary.replace(output)
    output_probe = probe(output)
    duration_ms = round(float(output_probe["format"]["duration"]) * 1_000)
    if not 30 * 60 * 1_000 <= duration_ms <= 43 * 60 * 1_000:
        raise RuntimeError(f"unexpected long-form duration: {duration_ms} ms")
    manifest = {
        "schema_version": "shotseek-longform-material-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "output": {
            "path": str(output.relative_to(root)),
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "duration_ms": duration_ms,
            "encoder": encoder,
            "probe": output_probe,
        },
        "sources": source_records,
        "composition": {
            "order": [source["title"] for source in SOURCES],
            "video": "1280x720 CFR 25fps H.264",
            "audio": "48kHz stereo AAC 96kbps",
            "credits_preserved": True,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
