#!/usr/bin/env python3
"""Prepare deterministic WebRTC-safe camera-emulator files with NVIDIA codecs."""

from __future__ import annotations

import subprocess
from pathlib import Path


DATA = Path("/workspace/data")
CACHE = DATA / ".camera_emulator_cache"
SUPPORTED = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
LIMIT = 16


def main() -> int:
    sources = sorted(
        (
            path
            for path in DATA.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED
            and CACHE not in path.parents
        ),
        key=lambda path: str(path.relative_to(DATA)).lower(),
    )[:LIMIT]
    if len(sources) != LIMIT:
        raise RuntimeError(f"Expected {LIMIT} source videos, found {len(sources)}")

    CACHE.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, start=1):
        output = CACHE / f"camera-{index:03d}.mp4"
        if output.is_file() and output.stat().st_size > 1024:
            print(f"GPU cache valid {index}/{LIMIT}: {output.name}")
            continue
        output.unlink(missing_ok=True)
        command = [
            "gst-launch-1.0",
            "-e",
            "-q",
            "nvurisrcbin",
            f"uri={source.as_uri()}",
            "disable-audio=true",
            "file-loop=false",
            "!",
            "video/x-raw(memory:NVMM)",
            "!",
            "nvvideoconvert",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            "bitrate=6000000",
            "iframeinterval=30",
            "idrinterval=30",
            "control-rate=1",
            "preset-id=1",
            "tuning-info-id=3",
            "profile=0",
            "!",
            "h264parse",
            "config-interval=-1",
            "!",
            "mp4mux",
            "faststart=true",
            "!",
            "filesink",
            f"location={output}",
        ]
        print(f"GPU transcoding {index}/{LIMIT}: {source.relative_to(DATA)}")
        subprocess.run(command, check=True)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"GPU transcode produced no output: {output}")

    manifest = CACHE / "manifest.txt"
    manifest.write_text(
        "\n".join(
            f"camera-{index:03d}.mp4|{source.relative_to(DATA)}"
            for index, source in enumerate(sources, start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
