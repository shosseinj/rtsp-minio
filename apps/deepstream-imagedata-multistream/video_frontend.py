#!/usr/bin/env python3
"""Serve every local video in a browser-ready gallery.

This intentionally performs no AI or frame decoding. Future frame processors
can be added behind the existing /api/videos contract without changing the UI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, send_file
from werkzeug.serving import make_server


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def create_app(video_folder: Path) -> Flask:
    app = Flask("deepstream-video-frontend")
    frontend = Path(__file__).with_name("deepstream_demo.html")

    def videos() -> list[Path]:
        return sorted(
            (
                path
                for path in video_folder.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )

    @app.get("/")
    def index() -> Any:
        return send_file(frontend, conditional=True)

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok", "video_folder": str(video_folder)}

    @app.get("/api/videos")
    def list_videos() -> Any:
        return jsonify(
            [
                {
                    "id": index,
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "content_url": f"/videos/{index}",
                }
                for index, path in enumerate(videos())
            ]
        )

    @app.get("/videos/<int:video_id>")
    def video_content(video_id: int) -> Any:
        available = videos()
        if video_id < 0 or video_id >= len(available):
            return {"error": "Video not found"}, 404
        return send_file(available[video_id], conditional=True)

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local video gallery")
    parser.add_argument("--video-folder", default="/workspace/video")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7070)
    args = parser.parse_args()

    video_folder = Path(args.video_folder).resolve()
    if not video_folder.is_dir():
        parser.error(f"video folder does not exist: {video_folder}")

    app = create_app(video_folder)
    server = make_server(args.host, args.port, app, threaded=True)
    print(f"Video frontend: http://HOST:{args.port}/", flush=True)
    print(f"Video folder: {video_folder}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
