#!/usr/bin/env python3
"""
Low-latency DeepStream RTSP/local-file mosaic publisher.

Pipeline:
    RTSP cameras or local video files
      -> NVIDIA hardware decode
      -> nvstreammux
      -> nvmultistreamtiler
      -> nvdsosd
      -> NVIDIA H.264 encoder
      -> rtspclientsink
      -> MediaMTX
      -> WebRTC browser playback

This version intentionally does not use Flask, MJPEG, appsink, OpenCV,
or CPU frame copies. Those stages were the primary source of latency.

Example:
    python3 deepstream_low_latency_webrtc.py \
        --publish-url rtsp://mediamtx:8554/deepstream-mosaic \
        --transport udp \
        rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101 \
        rtsp://user:pass@192.168.1.11:554/Streaming/Channels/101

Local-file example:
    python3 deepstream_low_latency_rtsp_and_file.py \
        --publish-url rtsp://127.0.0.1:8554/deepstream-mosaic \
        file:///workspace/video4.mp4

Browser:
    http://SERVER_IP:8889/deepstream-mosaic
"""

from __future__ import annotations

import argparse
import configparser
import ctypes
import datetime as dt
import json
import math
import os
import queue
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

GST_CAPS_FEATURES_NVMM = "memory:NVMM"
# NvBufSurfaceMemType.NVBUF_MEM_CUDA_DEVICE. Keeping the numeric GStreamer
# property value here avoids a Python-binding dependency in this video-only
# pipeline; pixels never leave GPU device memory.
NVBUF_MEM_CUDA_DEVICE = 2

# GstRtsp.RTSPLowerTrans flags.
RTSP_TRANSPORTS = {
    "udp": 1,
    "tcp": 4,
}


@dataclass(frozen=True)
class Settings:
    uris: list[str]
    cameras: list[dict[str, Any]]
    publish_url: str
    output_width: int
    output_height: int
    mux_width: int
    mux_height: int
    bitrate: int
    gop: int
    rtsp_latency_ms: int
    rtsp_transport: str
    batch_timeout_us: int
    gpu_id: int
    decoder_low_latency: bool
    loop_files: bool
    web_port: int


class LowLatencyDeepStreamPublisher:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline: Gst.Pipeline | None = None
        self.loop: GLib.MainLoop | None = None
        self.streammux: Gst.Element | None = None
        self._source_tees: dict[int, Gst.Element] = {}
        self._fullscreen_source_tees: dict[int, Gst.Element] = {}
        self._demux_request_pads: list[Gst.Pad] = []
        self._fullscreen_branch: dict[str, Any] | None = None
        self._fullscreen_lock = threading.Lock()
        self._fullscreen_state = "IDLE"
        self._fullscreen_overlay_requested: bool | None = None
        self._fullscreen_operation_id = 0
        self._fullscreen_metrics = {
            "fullscreen_create_requests": 0,
            "fullscreen_create_success": 0,
            "fullscreen_create_failures": 0,
            "fullscreen_destroy_requests": 0,
            "fullscreen_destroy_success": 0,
            "fullscreen_destroy_failures": 0,
            "fullscreen_switch_requests": 0,
            "fullscreen_operation_timeouts": 0,
            "fullscreen_stale_callbacks_ignored": 0,
            "fullscreen_overlay_frames": 0,
            "fullscreen_overlay_metadata_missing": 0,
            "fullscreen_encoder_create_count": 0,
            "fullscreen_encoder_reuse_count": 0,
            "fullscreen_encoder_recreate_count": 0,
            "fullscreen_first_buffer_timeout": 0,
            "fullscreen_zero_fps_events": 0,
            "fullscreen_source_switch_count": 0,
            "fullscreen_stale_frames_dropped": 0,
            "overlay_bbox_transform_count": 0,
            "overlay_bbox_transform_errors": 0,
            "overlay_bbox_clamped_count": 0,
            "overlay_coordinate_space_mismatch": 0,
            "overlay_metadata_copy_failures": 0,
        }
        self._requested_mux_pads: list[Gst.Pad] = []
        self._web_server: Any = None
        self._web_thread: threading.Thread | None = None
        self._loop_restarts_pending: set[str] = set()
        self._file_publisher_stop = threading.Event()
        self._file_publisher_processes: list[subprocess.Popen[Any]] = []
        self._file_publisher_threads: list[threading.Thread] = []
        self._fps_lock = threading.Lock()
        now = time.monotonic()
        generation_seed = int(time.time() * 1000)
        self._source_stats: dict[int, dict[str, Any]] = {
            index: {
                "frames": 0,
                "window_frames": 0,
                "window_started": now,
                "fps": 0.0,
                "width": 0,
                "height": 0,
                "last_frame_at": 0.0,
                "last_pts": -1,
                "loop_count": 0,
                "generation": generation_seed,
                "generation_changes": 0,
            }
            for index in range(len(settings.uris))
        }
        self._wall_stats: dict[str, Any] = {
            "window_frames": 0,
            "window_started": now,
            "fps": 0.0,
            "frames": 0,
        }
        self._fullscreen_stats: dict[str, Any] = {
            "camera_id": None,
            "fps": 0.0,
            "frames": 0,
            "window_frames": 0,
            "window_started": now,
        }
        self._last_wall_buffer_at = 0.0
        self._ai_enabled = os.getenv("AI_STREAM_ENABLED", "false").lower() in {
            "1", "true", "yes", "on"
        }
        self._ai_stats: dict[str, Any] = {
            "ai_frames_received": 0,
            "ai_frames_resized_gpu": 0,
            "person_detections_total": 0,
            "pose_keypoints_extracted_total": 0,
            "active_person_tracks": 0,
            "ended_person_tracks": 0,
            "track_candidate_count": 0,
            "track_candidates_dropped": 0,
            "redis_events_published": 0,
            "redis_events_pending": 0,
            "minio_upload_success": 0,
            "minio_upload_pending": 0,
            "full_frame_cpu_copy_count": 0,
            "full_resolution_ndarray_queue_count": 0,
            "window_frames": 0,
            "window_started": now,
            "person_inference_fps": 0.0,
        }
        self._person_tracks: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._finalized_track_keys: set[tuple[str, int, int]] = set()
        self._overlay_global = os.getenv(
            "AI_OVERLAY_DEFAULT_ENABLED", "true"
        ).lower() in {"1", "true", "yes", "on"}
        self._overlay_per_camera: dict[str, bool | None] = {}
        self._overlay_stats = {
            "overlay_frames_rendered": 0,
            "overlay_render_errors": 0,
            "fullscreen_requests_total": 0,
            "fullscreen_route_failures": 0,
            "fullscreen_wrong_source_count": 0,
            "duplicate_track_end_events": 0,
            "sqlite_events_created": 0,
            "sqlite_write_failures": 0,
        }
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=int(os.getenv("EVENT_QUEUE_MAX_SIZE", "1000"))
        )
        self._event_db_path = Path(
            os.getenv(
                "EVENT_SQLITE_PATH", "/workspace/data/events/events.db"
            )
        )
        self._event_worker_stop = threading.Event()
        self._event_worker = threading.Thread(
            target=self._event_db_loop,
            name="person-event-sqlite",
            daemon=True,
        )
        self._event_worker.start()
        self._outbox_path = Path(
            os.getenv("AI_DURABLE_OUTBOX_PATH", "/workspace/spool/events.jsonl")
        )

    @staticmethod
    def _make(factory: str, name: str) -> Gst.Element:
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(
                f"Required GStreamer element is unavailable: {factory}"
            )
        return element

    @staticmethod
    def _set_if_supported(
        element: Gst.Element,
        property_name: str,
        value: Any,
    ) -> bool:
        if element.find_property(property_name) is None:
            return False
        element.set_property(property_name, value)
        return True

    @staticmethod
    def _request_pad(
        element: Gst.Element,
        name: str,
    ) -> Gst.Pad | None:
        pad = element.request_pad_simple(name)
        if pad is None:
            # Compatibility with older GStreamer / DeepStream versions.
            pad = element.get_request_pad(name)
        return pad

    @staticmethod
    def _link_many(elements: list[Gst.Element]) -> None:
        for left, right in zip(elements, elements[1:]):
            if not left.link(right):
                raise RuntimeError(
                    f"Could not link {left.get_name()} -> {right.get_name()}"
                )

    @staticmethod
    def _configure_leaky_queue(queue: Gst.Element) -> None:
        # Realtime presentation: retain only the newest GPU buffer. Allowing a
        # backlog makes late frames appear as freezes/visual echo.
        queue.set_property("max-size-buffers", 1)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)
        queue.set_property("leaky", 2)  # downstream

    def _configure_h264_encoder(
        self,
        encoder: Gst.Element,
        bitrate: int,
    ) -> None:
        """Configure NVIDIA NVENC without any software-encoder fallback."""
        self._set_if_supported(encoder, "gpu-id", self.settings.gpu_id)
        self._set_if_supported(encoder, "bitrate", bitrate)
        self._set_if_supported(encoder, "iframeinterval", self.settings.gop)
        self._set_if_supported(encoder, "idrinterval", self.settings.gop)
        self._set_if_supported(encoder, "insert-sps-pps", True)
        self._set_if_supported(encoder, "control-rate", 1)
        self._set_if_supported(encoder, "preset-id", 1)
        self._set_if_supported(encoder, "tuning-info-id", 3)
        self._set_if_supported(encoder, "num-B-Frames", 0)
        self._set_if_supported(encoder, "profile", 0)
        self._set_if_supported(encoder, "qos", False)

    def _configure_rtsp_publisher(
        self,
        publisher: Gst.Element,
        location: str,
        sync: bool,
    ) -> None:
        publisher.set_property("location", location)
        self._set_if_supported(
            publisher,
            "protocols",
            RTSP_TRANSPORTS["tcp"],
        )
        self._set_if_supported(publisher, "sync", sync)
        self._set_if_supported(publisher, "async", False)
        self._set_if_supported(publisher, "qos", False)

    def _individual_publish_url(self, camera_id: str) -> str:
        parsed = urlsplit(self.settings.publish_url)
        return f"{parsed.scheme}://{parsed.netloc}/{camera_id}"

    def _file_passthrough_loop(
        self,
        uri: str,
        camera_id: str,
    ) -> None:
        path = Path(unquote(urlsplit(uri).path))
        publish_url = self._individual_publish_url(camera_id)
        while not self._file_publisher_stop.is_set():
            command = [
                "gst-launch-1.0",
                "-q",
                "filesrc",
                f"location={path}",
                "!",
                "qtdemux",
                "!",
                "queue",
                "max-size-buffers=4",
                "leaky=downstream",
                "!",
                "h264parse",
                "config-interval=-1",
                "!",
                "rtspclientsink",
                f"location={publish_url}",
                "protocols=tcp",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._file_publisher_processes.append(process)
            while (
                process.poll() is None
                and not self._file_publisher_stop.wait(0.25)
            ):
                pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            if not self._file_publisher_stop.wait(1.0):
                continue

    def _start_file_passthrough_publishers(self) -> None:
        if os.getenv("BENCHMARK_PIPELINE_MODE", "full") == "decode_only":
            return
        if os.getenv(
            "STATIC_CAMERA_PASSTHROUGH",
            "true",
        ).lower() not in {"1", "true", "yes", "on"}:
            return
        camera_by_source = {
            int(camera["source_index"]): camera
            for camera in self.settings.cameras
        }
        for source_index, uri in enumerate(self.settings.uris):
            if not uri.lower().startswith("file://"):
                continue
            camera = camera_by_source.get(source_index)
            if camera is None:
                continue
            if camera.get("requires_gpu_transcode", False):
                continue
            thread = threading.Thread(
                target=self._file_passthrough_loop,
                args=(uri, str(camera["camera_id"])),
                name=f"file-publisher-{source_index:03d}",
                daemon=True,
            )
            self._file_publisher_threads.append(thread)
            thread.start()

    def _check_publish_server(self) -> None:
        """Fail with a useful error before starting the expensive pipeline."""
        parsed = urlsplit(self.settings.publish_url)
        if parsed.scheme.lower() not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError(
                "--publish-url must be a valid rtsp:// or rtsps:// URL"
            )

        port = parsed.port or (322 if parsed.scheme.lower() == "rtsps" else 554)
        last_error: OSError | None = None
        for attempt in range(20):
            try:
                with socket.create_connection(
                    (parsed.hostname, port), timeout=1.0
                ):
                    return
            except OSError as exc:
                last_error = exc
                if attempt < 19:
                    time.sleep(0.25)

        if last_error is not None:
            docker_hint = ""
            if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
                docker_hint = (
                    " In Docker, 127.0.0.1 is this DeepStream container, not "
                    "the host or another container. Start MediaMTX in the same "
                    "Compose network and use rtsp://mediamtx:8554/PATH, or use "
                    "host.docker.internal with a host-published MediaMTX port."
                )
            raise ConnectionError(
                f"Cannot reach the RTSP publishing server at "
                f"{parsed.hostname}:{port}.{docker_hint} Original error: "
                f"{last_error}"
            ) from last_error

    @staticmethod
    def _decodebin_autoplug_continue(
        _decodebin: Gst.Element,
        _pad: Gst.Pad,
        caps: Gst.Caps,
    ) -> bool:
        """Only decode video; compressed audio is irrelevant to this pipeline."""
        if caps is None or caps.get_size() == 0:
            return True
        return caps.get_structure(0).get_name().startswith("video/")

    def _decodebin_child_added(
        self,
        _child_proxy: Any,
        child: Gst.Element,
        name: str,
        source_context: dict[str, Any],
    ) -> None:
        lower_name = name.lower()
        is_live = bool(source_context["is_live"])

        # Decodebins can contain nested decodebins.
        if "decodebin" in lower_name:
            child.connect(
                "child-added",
                self._decodebin_child_added,
                source_context,
            )

        # RTSP-only settings. Do not apply these to file sources.
        if is_live:
            if child.find_property("latency") is not None:
                child.set_property(
                    "latency",
                    self.settings.rtsp_latency_ms,
                )

            if child.find_property("drop-on-latency") is not None:
                child.set_property("drop-on-latency", True)

            if child.find_property("protocols") is not None:
                child.set_property(
                    "protocols",
                    RTSP_TRANSPORTS[self.settings.rtsp_transport],
                )

            if child.find_property("buffer-mode") is not None:
                child.set_property("buffer-mode", 0)

            if child.find_property("do-retransmission") is not None:
                child.set_property("do-retransmission", False)

            if child.find_property("udp-buffer-size") is not None:
                child.set_property("udp-buffer-size", 524288)

        # Keep decoded surfaces in CUDA device memory.
        if "nvv4l2decoder" in lower_name:
            self._set_if_supported(
                child,
                "cudadec-memtype",
                0,
            )

            # NVIDIA documents this mode for streams without B-frames.
            # Keep it opt-in; enabling it for ordinary MP4 files can be unsafe.
            if is_live and self.settings.decoder_low_latency:
                self._set_if_supported(
                    child,
                    "low-latency-mode",
                    True,
                )

    def _fullscreen_fps_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        if info.get_buffer() is None:
            return Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self._fps_lock:
            stats = self._fullscreen_stats
            stats["frames"] += 1
            stats["window_frames"] += 1
            elapsed = now - stats["window_started"]
            if elapsed >= 1.0:
                stats["fps"] = stats["window_frames"] / elapsed
                stats["window_frames"] = 0
                stats["window_started"] = now
        if (
            self._fullscreen_branch is not None
            and self._fullscreen_branch.get("overlay_enabled")
        ):
            self._fullscreen_metrics["fullscreen_overlay_frames"] += 1
        return Gst.PadProbeReturn.OK

    def _fullscreen_retimestamp_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        """Give switched RTSP sources one monotonic live timeline.

        This only edits GstBuffer metadata. The NVMM surface remains on GPU
        and no frame pixels are mapped or copied to CPU memory.
        """
        buffer = info.get_buffer()
        if buffer is None or self.pipeline is None:
            return Gst.PadProbeReturn.OK
        clock = self.pipeline.get_clock()
        if clock is None:
            return Gst.PadProbeReturn.OK
        running_time = max(
            0,
            int(clock.get_time() - self.pipeline.get_base_time()),
        )
        buffer.pts = running_time
        buffer.dts = running_time
        return Gst.PadProbeReturn.OK

    def _wall_rate_limit_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        if info.get_buffer() is None:
            return Gst.PadProbeReturn.OK
        now = time.monotonic()
        target_fps = int(os.getenv("WALL_TARGET_FPS", "12"))
        if (
            self._last_wall_buffer_at > 0
            and now - self._last_wall_buffer_at < (1.0 / target_fps)
        ):
            return Gst.PadProbeReturn.DROP
        self._last_wall_buffer_at = now
        return Gst.PadProbeReturn.OK

    def _set_fullscreen_overlay(self, enabled: bool) -> bool:
        branch = self._fullscreen_branch
        if branch is None:
            return False
        for element in branch["elements"]:
            if element.get_name() == "fullscreen-osd":
                self._set_if_supported(element, "display-bbox", enabled)
                self._set_if_supported(element, "display-text", enabled)
                branch["overlay_enabled"] = enabled
                break
        return False

    def _stop_fullscreen_branch(self) -> bool:
        self._fullscreen_state = "STOPPING"
        with self._fullscreen_lock:
            branch = self._fullscreen_branch
            if branch is None or self.pipeline is None:
                self._fullscreen_state = "STOPPED"
                return False
            tee_pad = branch["tee_pad"]
            queue = branch["elements"][0]
            queue_sink = queue.get_static_pad("sink")
            if queue_sink is not None:
                tee_pad.unlink(queue_sink)
            branch["tee"].release_request_pad(tee_pad)
            for element in reversed(branch["elements"]):
                element.set_state(Gst.State.NULL)
                self.pipeline.remove(element)
            self._fullscreen_branch = None
            self._fullscreen_state = "STOPPED"
            self._fullscreen_metrics["fullscreen_destroy_success"] += 1
            with self._fps_lock:
                self._fullscreen_stats.update(
                    {
                        "camera_id": None,
                        "fps": 0.0,
                        "frames": 0,
                        "window_frames": 0,
                        "window_started": time.monotonic(),
                    }
                )
        return False

    def _start_fullscreen_branch(
        self,
        camera_id: str,
        source_index: int,
    ) -> bool:
        if self.pipeline is None:
            return False
        self._fullscreen_state = "CREATING"
        tee = self._fullscreen_source_tees.get(source_index)
        if tee is None:
            self._fullscreen_state = "FAILED"
            self._fullscreen_metrics["fullscreen_create_failures"] += 1
            return False
        if self._fullscreen_branch is not None:
            self._fullscreen_metrics["fullscreen_switch_requests"] += 1
            self._stop_fullscreen_branch()
        with self._fullscreen_lock:
            branch = self._fullscreen_branch
            if branch is not None:
                queue = branch["elements"][0]
                queue_sink = queue.get_static_pad("sink")
                old_pad = branch["tee_pad"]
                if queue_sink is None:
                    return False
                old_pad.unlink(queue_sink)
                branch["tee"].release_request_pad(old_pad)
                new_pad = self._request_pad(tee, "src_%u")
                if (
                    new_pad is None
                    or new_pad.link(queue_sink) != Gst.PadLinkReturn.OK
                ):
                    return False
                branch.update(
                    {
                        "camera_id": camera_id,
                        "tee": tee,
                        "tee_pad": new_pad,
                    }
                )
                with self._fps_lock:
                    self._fullscreen_stats.update(
                        {
                            "camera_id": camera_id,
                            "fps": 0.0,
                            "frames": 0,
                            "window_frames": 0,
                            "window_started": time.monotonic(),
                        }
                    )
                return False

        queue = self._make("queue", "fullscreen-queue")
        converter = self._make("nvvideoconvert", "fullscreen-rgba-converter")
        rgba_caps = self._make("capsfilter", "fullscreen-rgba-caps")
        fullscreen_osd = self._make("nvdsosd", "fullscreen-osd")
        nv12_converter = self._make(
            "nvvideoconvert", "fullscreen-nv12-converter"
        )
        capsfilter = self._make("capsfilter", "fullscreen-nv12-caps")
        rate = self._make("videorate", "fullscreen-rate")
        rate_caps = self._make(
            "capsfilter", "fullscreen-rate-caps"
        )
        encoder = self._make("nvv4l2h264enc", "fullscreen-encoder")
        parser = self._make("h264parse", "fullscreen-parser")
        publisher = self._make("rtspclientsink", "fullscreen-publisher")
        self._configure_leaky_queue(queue)
        self._set_if_supported(converter, "gpu-id", self.settings.gpu_id)
        self._set_if_supported(
            converter, "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE
        )
        self._set_if_supported(
            nv12_converter, "nvbuf-memory-type", NVBUF_MEM_CUDA_DEVICE
        )
        self._set_if_supported(
            nv12_converter, "gpu-id", self.settings.gpu_id
        )
        rgba_caps.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"),
        )
        overlay_enabled = self._overlay_per_camera.get(camera_id)
        if overlay_enabled is None:
            overlay_enabled = self._overlay_global
        if self._fullscreen_overlay_requested is not None:
            overlay_enabled = self._fullscreen_overlay_requested
        self._set_if_supported(fullscreen_osd, "process-mode", 1)
        self._set_if_supported(
            fullscreen_osd, "display-bbox", bool(overlay_enabled)
        )
        self._set_if_supported(
            fullscreen_osd, "display-text", bool(overlay_enabled)
        )
        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=NV12,"
                f"width={int(os.getenv('FULLSCREEN_OUTPUT_WIDTH', '2944'))},"
                f"height={int(os.getenv('FULLSCREEN_OUTPUT_HEIGHT', '1664'))}"
            ),
        )
        # Never duplicate an NVMM buffer reference. Decoder surfaces can be
        # recycled after presentation; repeating one can show a white/corrupt
        # frame when the same GPU surface is reused.
        self._set_if_supported(rate, "drop-only", True)
        self._set_if_supported(rate, "skip-to-first", True)
        rate_caps.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=NV12,"
                f"width={int(os.getenv('FULLSCREEN_OUTPUT_WIDTH', '2944'))},"
                f"height={int(os.getenv('FULLSCREEN_OUTPUT_HEIGHT', '1664'))},"
                "framerate=20/1"
            ),
        )
        self._configure_h264_encoder(encoder, self.settings.bitrate)
        parser.set_property("config-interval", -1)
        parsed = urlsplit(self.settings.publish_url)
        stream_path = (
            f"fullscreen-{camera_id}-"
            f"{int(self._source_stats[source_index]['generation'])}"
        )
        fullscreen_url = f"{parsed.scheme}://{parsed.netloc}/{stream_path}"
        self._configure_rtsp_publisher(publisher, fullscreen_url, False)
        elements = [
            queue,
            converter,
            rgba_caps,
            fullscreen_osd,
            nv12_converter,
            capsfilter,
            rate,
            rate_caps,
            encoder,
            parser,
            publisher,
        ]
        for element in elements:
            self.pipeline.add(element)
        self._link_many(elements)
        queue_src = queue.get_static_pad("src")
        if queue_src is not None:
            queue_src.add_probe(
                Gst.PadProbeType.BUFFER,
                self._fullscreen_retimestamp_probe,
            )
        tee_pad = self._request_pad(tee, "src_%u")
        queue_sink = queue.get_static_pad("sink")
        if (
            tee_pad is None
            or queue_sink is None
            or tee_pad.link(queue_sink) != Gst.PadLinkReturn.OK
        ):
            for element in reversed(elements):
                element.set_state(Gst.State.NULL)
                self.pipeline.remove(element)
            return False
        parser_src = parser.get_static_pad("src")
        if parser_src is not None:
            parser_src.add_probe(
                Gst.PadProbeType.BUFFER,
                self._fullscreen_fps_probe,
            )
        # Bring the network sink up first. Starting the upstream queue before
        # rtspclientsink is ready can consume one buffer and then stall a
        # dynamically attached branch.
        for element in reversed(elements):
            element.sync_state_with_parent()
        with self._fps_lock:
            self._fullscreen_stats.update(
                {
                    "camera_id": camera_id,
                    "fps": 0.0,
                    "frames": 0,
                    "window_frames": 0,
                    "window_started": time.monotonic(),
                    "stream_path": stream_path,
                }
            )
        with self._fullscreen_lock:
            self._fullscreen_branch = {
                "camera_id": camera_id,
                "tee": tee,
                "tee_pad": tee_pad,
                "elements": elements,
                "overlay_enabled": bool(overlay_enabled),
            }
            self._fullscreen_state = "ACTIVE"
        return False

    def _start_frontend(self) -> None:
        if self.settings.web_port <= 0 or self._web_server is not None:
            return

        try:
            from flask import Flask, jsonify, request, send_file
            from werkzeug.serving import make_server
        except ImportError as exc:
            raise RuntimeError(
                "The demo frontend requires Flask. Install it with: "
                "pip3 install flask"
            ) from exc

        dashboard_path = Path(__file__).with_name("deepstream_demo.html")
        if not dashboard_path.exists():
            raise RuntimeError(f"Frontend file is missing: {dashboard_path}")

        files: list[dict[str, Any]] = []
        for index, uri in enumerate(self.settings.uris):
            parsed = urlsplit(uri)
            if parsed.scheme.lower() != "file":
                continue
            path = Path(unquote(parsed.path))
            if path.exists():
                files.append(
                    {
                        "id": index,
                        "name": path.name,
                        "path": path,
                        "size_bytes": path.stat().st_size,
                    }
                )

        app = Flask("deepstream-video-demo")

        @app.get("/")
        def dashboard() -> Any:
            return send_file(dashboard_path, conditional=True)

        @app.get("/api/videos")
        def videos() -> Any:
            return jsonify(
                [
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "size_bytes": item["size_bytes"],
                        "content_url": f"/videos/{item['id']}",
                    }
                    for item in files
                ]
            )

        @app.get("/api/cameras")
        def cameras() -> Any:
            return jsonify(self.settings.cameras)

        @app.get("/api/runtime")
        def runtime() -> Any:
            now = time.monotonic()
            with self._fps_lock:
                sources = []
                for camera in self.settings.cameras:
                    source_index = int(camera["source_index"])
                    stats = self._source_stats[source_index]
                    sources.append(
                        {
                            "camera_id": camera["camera_id"],
                            "source_index": source_index,
                            "fps": round(float(stats["fps"]), 1),
                            "frames": int(stats["frames"]),
                            "width": int(stats["width"]),
                            "height": int(stats["height"]),
                            "online": (
                                stats["last_frame_at"] > 0
                                and now - stats["last_frame_at"] < 3.0
                            ),
                            "loop_count": int(stats["loop_count"]),
                            "generation": int(stats["generation"]),
                            "generation_changes": int(
                                stats["generation_changes"]
                            ),
                        }
                    )
            return jsonify(
                {
                    "sources": sources,
                    "wall_fps": round(float(self._wall_stats["fps"]), 1),
                    "wall_frames": int(self._wall_stats["frames"]),
                    "fullscreen": {
                        "camera_id": self._fullscreen_stats["camera_id"],
                        "fps": round(
                            float(self._fullscreen_stats["fps"]), 1
                        ),
                        "frames": int(self._fullscreen_stats["frames"]),
                        "active": self._fullscreen_branch is not None,
                    },
                    "fullscreen_controller": {
                        "state": self._fullscreen_state,
                        "operation_id": self._fullscreen_operation_id,
                        **self._fullscreen_metrics,
                        "fullscreen_active_sessions": (
                            1 if self._fullscreen_branch else 0
                        ),
                        "fullscreen_encoder_sessions": (
                            1 if self._fullscreen_branch else 0
                        ),
                        "fullscreen_request_pads_active": (
                            1 if self._fullscreen_branch else 0
                        ),
                    },
                    "wall_nvenc_sessions": 1,
                    "total_nvenc_sessions": (
                        2 if self._fullscreen_branch is not None else 1
                    ),
                    "video_branch_active": True,
                    "active_video_subscribers": -1,
                    "active_video_nvenc_sessions": (
                        2 if self._fullscreen_branch is not None else 1
                    ),
                    "ai": {
                        key: (
                            round(float(value), 2)
                            if isinstance(value, float)
                            else value
                        )
                        for key, value in self._ai_stats.items()
                        if key not in {"window_frames", "window_started"}
                    },
                    "overlay": {
                        "overlay_enabled_global": self._overlay_global,
                        "overlay_enabled_per_camera": self._overlay_per_camera,
                        **self._overlay_stats,
                    },
                    "full_frame_cpu_copy_count": self._ai_stats[
                        "full_frame_cpu_copy_count"
                    ],
                    "full_resolution_ndarray_queue_count": self._ai_stats[
                        "full_resolution_ndarray_queue_count"
                    ],
                }
            )

        @app.get("/api/overlay")
        def overlay_get() -> Any:
            return jsonify(
                {
                    "global_overlay_enabled": self._overlay_global,
                    "camera_overrides": self._overlay_per_camera,
                }
            )

        @app.post("/api/overlay")
        @app.put("/api/overlay")
        def overlay_put() -> Any:
            payload = request.get_json(silent=True) or {}
            if "global_overlay_enabled" in payload:
                self._overlay_global = bool(
                    payload["global_overlay_enabled"]
                )
            camera_id = payload.get("camera_id")
            if camera_id is not None:
                if not any(
                    camera["camera_id"] == camera_id
                    for camera in self.settings.cameras
                ):
                    return {"error": "camera not found"}, 404
                value = payload.get("camera_overlay_enabled")
                self._overlay_per_camera[str(camera_id)] = (
                    None if value is None else bool(value)
                )
            if "fullscreen_overlay_enabled" in payload:
                GLib.idle_add(
                    self._set_fullscreen_overlay,
                    bool(payload["fullscreen_overlay_enabled"]),
                )
            return overlay_get()

        @app.get("/api/events")
        def events_list() -> Any:
            limit = min(500, max(1, int(request.args.get("limit", 100))))
            offset = max(0, int(request.args.get("offset", 0)))
            connection = sqlite3.connect(self._event_db_path)
            connection.row_factory = sqlite3.Row
            clauses, values = [], []
            for field in ("source_id", "status", "track_id"):
                value = request.args.get(field)
                if value is not None:
                    clauses.append(f"{field}=?")
                    values.append(value)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"SELECT * FROM person_track_events{where} "
                "ORDER BY ended_at DESC LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
            connection.close()
            return jsonify([dict(row) for row in rows])

        @app.get("/api/events/<event_id>")
        def event_detail(event_id: str) -> Any:
            connection = sqlite3.connect(self._event_db_path)
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM person_track_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            connection.close()
            return (jsonify(dict(row)), 200) if row else (
                jsonify({"error": "event not found"}), 404
            )

        @app.post("/api/fullscreen/switch")
        @app.post("/api/fullscreen/open")
        @app.post("/api/fullscreen/start")
        def fullscreen_start() -> Any:
            self._overlay_stats["fullscreen_requests_total"] += 1
            self._fullscreen_metrics["fullscreen_create_requests"] += 1
            self._fullscreen_operation_id += 1
            operation_id = self._fullscreen_operation_id
            payload = request.get_json(silent=True) or {}
            camera_id = str(
                payload.get("source_id", payload.get("camera_id", ""))
            )
            camera = next(
                (
                    item
                    for item in self.settings.cameras
                    if item["camera_id"] == camera_id
                ),
                None,
            )
            if camera is None:
                return {"error": "camera not found"}, 404
            now = time.monotonic()
            with self._fps_lock:
                source_stats = self._source_stats[int(camera["source_index"])]
                requested_generation = payload.get("generation")
                if (
                    requested_generation is not None
                    and int(requested_generation)
                    != int(source_stats["generation"])
                ):
                    return {"error": "stale_generation"}, 409
                if (
                    source_stats["last_frame_at"] <= 0
                    or now - source_stats["last_frame_at"] >= 3.0
                ):
                    self._overlay_stats["fullscreen_route_failures"] += 1
                    return {"error": "camera offline"}, 409
            actual_camera_id = str(camera["camera_id"])
            self._fullscreen_overlay_requested = (
                bool(payload["overlay_enabled"])
                if "overlay_enabled" in payload
                else None
            )
            GLib.idle_add(
                self._start_fullscreen_branch,
                actual_camera_id,
                int(camera["source_index"]),
            )
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                with self._fps_lock:
                    ready = (
                        self._fullscreen_stats["camera_id"]
                        == actual_camera_id
                        and self._fullscreen_stats["frames"] >= 2
                    )
                if ready:
                    self._fullscreen_metrics[
                        "fullscreen_create_success"
                    ] += 1
                    return {
                        "status": "ready",
                        "camera_id": actual_camera_id,
                        "requested_camera_id": camera_id,
                        "stream_path": self._fullscreen_stats.get(
                            "stream_path", "fullscreen"
                        ),
                        "operation_id": operation_id,
                    }
                time.sleep(0.1)
            self._fullscreen_metrics["fullscreen_create_failures"] += 1
            self._fullscreen_metrics["fullscreen_operation_timeouts"] += 1
            GLib.idle_add(self._stop_fullscreen_branch)
            cleanup_deadline = time.monotonic() + 4.0
            while (
                self._fullscreen_branch is not None
                and time.monotonic() < cleanup_deadline
            ):
                time.sleep(0.05)
            return {
                "error": "fullscreen stream did not become ready",
                "camera_id": actual_camera_id,
                "requested_camera_id": camera_id,
            }, 504

        @app.post("/api/fullscreen/close")
        @app.post("/api/fullscreen/stop")
        def fullscreen_stop() -> Any:
            self._fullscreen_metrics["fullscreen_destroy_requests"] += 1
            GLib.idle_add(self._stop_fullscreen_branch)
            deadline = time.monotonic() + 4.0
            while (
                self._fullscreen_branch is not None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            if self._fullscreen_branch is not None:
                return {"error": "fullscreen release timeout"}, 504
            return {"status": "stopped", "encoder": "released"}, 200

        @app.get("/api/fullscreen/status")
        def fullscreen_status() -> Any:
            branch = self._fullscreen_branch
            camera_id = (
                str(branch["camera_id"]) if branch is not None else None
            )
            generation = None
            if camera_id is not None:
                camera = next(
                    item for item in self.settings.cameras
                    if item["camera_id"] == camera_id
                )
                generation = self._source_stats[
                    int(camera["source_index"])
                ]["generation"]
            return {
                "state": self._fullscreen_state,
                "source_id": camera_id,
                "generation": generation,
                "operation_id": self._fullscreen_operation_id,
                "overlay_enabled": (
                    branch.get("overlay_enabled") if branch else None
                ),
                "encoder_active": branch is not None,
                "publisher_active": branch is not None,
            }

        @app.get("/health")
        def health() -> Any:
            return {
                "status": "ok",
                "camera_count": len(self.settings.cameras),
                "input_count": len(self.settings.uris),
            }

        @app.get("/videos/<int:video_id>")
        def video_content(video_id: int) -> Any:
            match = next(
                (item for item in files if item["id"] == video_id),
                None,
            )
            if match is None:
                return {"error": "Video not found"}, 404
            return send_file(
                match["path"],
                conditional=True,
                mimetype="video/mp4",
            )

        self._web_server = make_server(
            "0.0.0.0",
            self.settings.web_port,
            app,
            threaded=True,
        )
        self._web_thread = threading.Thread(
            target=self._web_server.serve_forever,
            name="deepstream-demo-web",
            daemon=True,
        )
        self._web_thread.start()

    def _decodebin_pad_added(
        self,
        _decodebin: Gst.Element,
        decoder_src_pad: Gst.Pad,
        source_bin: Gst.Bin,
    ) -> None:
        caps = decoder_src_pad.get_current_caps()
        if caps is None:
            caps = decoder_src_pad.query_caps(None)

        if caps is None or caps.get_size() == 0:
            return

        structure = caps.get_structure(0)
        media_type = structure.get_name()

        # Ignore audio and metadata pads.
        if not media_type.startswith("video/"):
            return

        features = caps.get_features(0)
        if (
            features is None
            or not features.contains(GST_CAPS_FEATURES_NVMM)
        ):
            print(
                "ERROR: uridecodebin did not select NVIDIA hardware decoding "
                "(memory:NVMM is missing).",
                file=sys.stderr,
            )
            return

        try:
            source_index = int(source_bin.get_name().rsplit("-", 1)[-1])
            width = int(structure.get_value("width") or 0)
            height = int(structure.get_value("height") or 0)
            with self._fps_lock:
                self._source_stats[source_index]["width"] = width
                self._source_stats[source_index]["height"] = height
        except (ValueError, TypeError):
            pass

        ghost_pad = source_bin.get_static_pad("src")
        if ghost_pad is None:
            print(
                f"ERROR: source bin {source_bin.get_name()} has no src pad",
                file=sys.stderr,
            )
            return

        if ghost_pad.get_target() is not None:
            return

        if not ghost_pad.set_target(decoder_src_pad):
            print(
                f"ERROR: failed to connect decoded video for "
                f"{source_bin.get_name()}",
                file=sys.stderr,
            )

    def _restart_file_source(self, decodebin: Gst.Element) -> bool:
        name = decodebin.get_name()
        try:
            flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT
            if not decodebin.seek_simple(Gst.Format.TIME, flags, 0):
                print(f"Could not loop {name}", file=sys.stderr)
        finally:
            self._loop_restarts_pending.discard(name)
        return GLib.SOURCE_REMOVE

    def _file_loop_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        decodebin: Gst.Element,
    ) -> Gst.PadProbeReturn:
        event = info.get_event()
        if event is None or event.type != Gst.EventType.EOS:
            return Gst.PadProbeReturn.OK

        name = decodebin.get_name()
        if name not in self._loop_restarts_pending:
            self._loop_restarts_pending.add(name)
            GLib.idle_add(self._restart_file_source, decodebin)
        return Gst.PadProbeReturn.DROP

    def _create_source_bin(
        self,
        index: int,
        uri: str,
    ) -> Gst.Bin:
        source_bin = Gst.Bin.new(f"source-bin-{index:02d}")
        if source_bin is None:
            raise RuntimeError(f"Could not create source bin {index}")

        is_file = uri.lower().startswith("file://")
        is_rtsp = uri.lower().startswith("rtsp://")
        emulate_static = (
            is_file
            and os.getenv("STATIC_CAMERA_EMULATION", "false").lower()
            in {"1", "true", "yes", "on"}
        )
        # nvurisrcbin is DeepStream's GPU-native RTSP source. It guarantees
        # NVIDIA decode/NVMM output instead of allowing uridecodebin to choose
        # a CPU decoder.
        source_factory = (
            "nvurisrcbin" if is_rtsp or emulate_static else "uridecodebin"
        )
        decodebin = self._make(
            source_factory,
            f"uri-decode-bin-{index:02d}",
        )
        decodebin.set_property("uri", uri)

        if is_rtsp:
            self._set_if_supported(decodebin, "disable-audio", True)
            self._set_if_supported(
                decodebin,
                "latency",
                self.settings.rtsp_latency_ms,
            )
            self._set_if_supported(decodebin, "drop-on-latency", True)
            self._set_if_supported(
                decodebin,
                "select-rtp-protocol",
                RTSP_TRANSPORTS[self.settings.rtsp_transport],
            )
            self._set_if_supported(decodebin, "cudadec-memtype", 0)
            self._set_if_supported(decodebin, "rtsp-reconnect-interval", 5)
            self._set_if_supported(decodebin, "rtsp-reconnect-attempts", -1)
        elif emulate_static:
            self._set_if_supported(decodebin, "disable-audio", True)
            self._set_if_supported(decodebin, "cudadec-memtype", 0)
            self._set_if_supported(decodebin, "file-loop", True)
        else:
            self._set_if_supported(
                decodebin,
                "expose-all-streams",
                False,
            )

        decodebin.connect(
            "pad-added",
            self._decodebin_pad_added,
            source_bin,
        )
        if source_factory == "uridecodebin":
            decodebin.connect(
                "autoplug-continue",
                self._decodebin_autoplug_continue,
            )
        source_context = {
            "source_bin": source_bin,
            "is_live": is_rtsp,
        }

        decodebin.connect(
            "child-added",
            self._decodebin_child_added,
            source_context,
        )

        source_bin.add(decodebin)

        ghost_pad = Gst.GhostPad.new_no_target(
            "src",
            Gst.PadDirection.SRC,
        )
        if ghost_pad is None or not source_bin.add_pad(ghost_pad):
            raise RuntimeError(
                f"Could not create source-bin ghost pad for source {index}"
            )

        if is_file and self.settings.loop_files and not emulate_static:
            ghost_pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                self._file_loop_probe,
                decodebin,
            )

        return source_bin

    def _processing_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        _user_data: Any,
    ) -> Gst.PadProbeReturn:
        """
        Add lightweight custom batch processing here.

        This callback receives the batched DeepStream GstBuffer before tiling.
        For minimum latency, attach/read NvDs metadata here and keep pixel
        processing on the GPU. Do not JPEG-encode, call cv2, sleep, or run a
        slow synchronous Python model inside this callback.
        """
        now = time.monotonic()
        with self._fps_lock:
            self._wall_stats["frames"] += 1
            self._wall_stats["window_frames"] += 1
            elapsed = now - self._wall_stats["window_started"]
            if elapsed >= 1.0:
                self._wall_stats["fps"] = (
                    self._wall_stats["window_frames"] / elapsed
                )
                self._wall_stats["window_frames"] = 0
                self._wall_stats["window_started"] = now
        return Gst.PadProbeReturn.OK

    def _append_track_event(self, track: dict[str, Any]) -> None:
        finalized_key = (
            str(track["source_id"]),
            int(track["generation"]),
            int(track["track_id"]),
        )
        if finalized_key in self._finalized_track_keys:
            return
        self._finalized_track_keys.add(finalized_key)
        self._outbox_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event_id": str(uuid4()),
            "event_type": "person_track_ended",
            "source_id": track["source_id"],
            "generation": track["generation"],
            "track_id": track["track_id"],
            "first_seen_pts": track["first_seen_pts"],
            "last_seen_pts": track["last_seen_pts"],
            "best_snapshot_artifact_id": None,
            "candidate_count": len(track["candidates"]),
            "status": "media_pending",
        }
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            self._overlay_stats["sqlite_write_failures"] += 1
            return
        self._ai_stats["ended_person_tracks"] += 1
        self._ai_stats["redis_events_pending"] += 1
        self._ai_stats["minio_upload_pending"] += 1

    @staticmethod
    def _utc_iso() -> str:
        return (
            dt.datetime.now(dt.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _event_db_loop(self) -> None:
        self._event_db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._event_db_path, timeout=5)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS person_track_events (
              event_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
              generation INTEGER NOT NULL, track_id INTEGER NOT NULL,
              first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
              ended_at TEXT NOT NULL, duration_ms INTEGER NOT NULL,
              first_seen_pts INTEGER, last_seen_pts INTEGER,
              max_confidence REAL, detection_count INTEGER NOT NULL DEFAULT 0,
              candidate_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
              snapshot_object_key TEXT, crop_object_key TEXT,
              video_object_key TEXT, snapshot_etag TEXT, crop_etag TEXT,
              video_etag TEXT, error_message TEXT,
              retry_count INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              UNIQUE(source_id,generation,track_id)
            )
            """
        )
        connection.commit()
        while not self._event_worker_stop.is_set():
            try:
                event = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            now = self._utc_iso()
            first_pts = int(event["first_seen_pts"])
            last_pts = int(event["last_seen_pts"])
            try:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO person_track_events
                    (event_id,source_id,generation,track_id,first_seen_at,
                     last_seen_at,ended_at,duration_ms,first_seen_pts,
                     last_seen_pts,candidate_count,status,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        event["event_id"], event["source_id"],
                        event["generation"], event["track_id"], now, now, now,
                        max(0, (last_pts - first_pts) // 1_000_000),
                        first_pts, last_pts, event["candidate_count"],
                        "MEDIA_PENDING", now, now,
                    ),
                )
                connection.commit()
                if cursor.rowcount:
                    self._overlay_stats["sqlite_events_created"] += 1
                    self._outbox_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._outbox_path.open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            json.dumps(event, separators=(",", ":")) + "\n"
                        )
                else:
                    self._overlay_stats["duplicate_track_end_events"] += 1
            except sqlite3.Error:
                self._overlay_stats["sqlite_write_failures"] += 1
            finally:
                self._event_queue.task_done()
        connection.close()

    def _ai_metadata_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        try:
            import pyds
        except ImportError as exc:
            raise RuntimeError(
                "AI_STREAM_ENABLED requires official DeepStream pyds"
            ) from exc

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK
        now = time.monotonic()
        seen: set[tuple[int, int, int]] = set()
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break
            source_index = int(frame_meta.pad_index)
            generation = int(self._source_stats[source_index]["generation"])
            source_id = str(
                next(
                    (
                        camera["camera_id"]
                        for camera in self.settings.cameras
                        if int(camera["source_index"]) == source_index
                    ),
                    source_index,
                )
            )
            self._ai_stats["ai_frames_received"] += 1
            self._ai_stats["ai_frames_resized_gpu"] += 1
            self._ai_stats["window_frames"] += 1
            pose_rows: list[dict[str, Any]] = []
            user_list = frame_meta.frame_user_meta_list
            while user_list is not None:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(user_list.data)
                except StopIteration:
                    break
                if (
                    user_meta.base_meta.meta_type
                    == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(
                        user_meta.user_meta_data
                    )
                    for layer_index in range(
                        int(tensor_meta.num_output_layers)
                    ):
                        layer = tensor_meta.output_layers_info(layer_index)
                        if str(layer.layerName or "") != "output0":
                            continue
                        dims = layer.inferDims
                        shape = [
                            int(dims.d[index])
                            for index in range(int(dims.numDims))
                        ]
                        if shape != [300, 57]:
                            continue
                        try:
                            address = pyds.get_ptr(layer.buffer)
                        except (TypeError, ValueError):
                            continue
                        if address == 0:
                            continue
                        try:
                            values = ctypes.cast(
                                address,
                                ctypes.POINTER(ctypes.c_float * (300 * 57)),
                            ).contents
                        except (TypeError, ValueError):
                            continue
                        for row_index in range(300):
                            offset = row_index * 57
                            confidence = float(values[offset + 4])
                            class_id = int(round(float(values[offset + 5])))
                            if confidence < 0.25 or class_id != 0:
                                continue
                            keypoints = [
                                {
                                    "x": float(values[offset + 6 + kp * 3]),
                                    "y": float(values[offset + 7 + kp * 3]),
                                    "confidence": float(
                                        values[offset + 8 + kp * 3]
                                    ),
                                }
                                for kp in range(17)
                            ]
                            pose_rows.append(
                                {
                                    "bbox_640": [
                                        float(values[offset]),
                                        float(values[offset + 1]),
                                        float(values[offset + 2]),
                                        float(values[offset + 3]),
                                    ],
                                    "confidence": confidence,
                                    "keypoints": keypoints,
                                }
                            )
                            self._ai_stats[
                                "pose_keypoints_extracted_total"
                            ] += 17
                try:
                    user_list = user_list.next
                except StopIteration:
                    break
            obj_list = frame_meta.obj_meta_list
            object_index = 0
            while obj_list is not None:
                try:
                    obj = pyds.NvDsObjectMeta.cast(obj_list.data)
                except StopIteration:
                    break
                if int(obj.class_id) == 0:
                    track_id = int(obj.object_id)
                    overlay_override = self._overlay_per_camera.get(source_id)
                    overlay_enabled = (
                        self._overlay_global
                        if overlay_override is None
                        else overlay_override
                    )
                    if overlay_enabled:
                        confidence = max(
                            float(obj.confidence),
                            float(getattr(obj, "tracker_confidence", 0.0)),
                        )
                        obj.rect_params.border_width = 3
                        obj.text_params.display_text = (
                            f"Person | ID: {track_id} | {confidence:.2f}"
                        )
                        self._overlay_stats[
                            "overlay_frames_rendered"
                        ] += 1
                    else:
                        obj.rect_params.border_width = 0
                        obj.text_params.display_text = ""
                    key = (source_index, generation, track_id)
                    seen.add(key)
                    rect = obj.rect_params
                    bbox = [
                        float(rect.left),
                        float(rect.top),
                        float(rect.left + rect.width),
                        float(rect.top + rect.height),
                    ]
                    pts = int(frame_meta.buf_pts)
                    track = self._person_tracks.setdefault(
                        key,
                        {
                            "source_id": source_id,
                            "generation": generation,
                            "track_id": track_id,
                            "first_seen_pts": pts,
                            "last_seen_pts": pts,
                            "last_seen_at": now,
                            "last_bbox_original": bbox,
                            "detection_count": 0,
                            "candidates": [],
                            "last_candidate_at": 0.0,
                        },
                    )
                    track["last_seen_pts"] = pts
                    track["last_seen_at"] = now
                    track["last_bbox_original"] = bbox
                    track["detection_count"] += 1
                    self._ai_stats["person_detections_total"] += 1
                    interval = int(os.getenv(
                        "PERSON_TRACK_CANDIDATE_INTERVAL_MS", "500"
                    )) / 1000.0
                    if now - track["last_candidate_at"] >= interval:
                        candidate = {
                            "frame_token": (
                                f"{source_id}:{generation}:"
                                f"{int(frame_meta.frame_num)}:{pts}"
                            ),
                            "pts": pts,
                            "bbox_original": bbox,
                            "detection_confidence": float(obj.confidence),
                            "quality_score": (
                                float(rect.width * rect.height)
                                * max(float(obj.confidence), 0.0)
                            ),
                        }
                        if object_index < len(pose_rows):
                            candidate["bbox_640"] = pose_rows[object_index][
                                "bbox_640"
                            ]
                            candidate["keypoints_640"] = pose_rows[
                                object_index
                            ]["keypoints"]
                        track["candidates"].append(candidate)
                        track["candidates"].sort(
                            key=lambda item: item["quality_score"],
                            reverse=True,
                        )
                        maximum = int(os.getenv(
                            "PERSON_TRACK_MAX_CANDIDATES", "8"
                        ))
                        if len(track["candidates"]) > maximum:
                            track["candidates"].pop()
                            self._ai_stats["track_candidates_dropped"] += 1
                        track["last_candidate_at"] = now
                    object_index += 1
                try:
                    obj_list = obj_list.next
                except StopIteration:
                    break
            try:
                frame_list = frame_list.next
            except StopIteration:
                break

        timeout = int(os.getenv("PERSON_TRACK_END_TIMEOUT_MS", "1500")) / 1000.0
        expired = [
            key for key, track in self._person_tracks.items()
            if key not in seen and now - track["last_seen_at"] >= timeout
        ]
        for key in expired:
            self._append_track_event(self._person_tracks.pop(key))
        elapsed = now - self._ai_stats["window_started"]
        if elapsed >= 1.0:
            self._ai_stats["person_inference_fps"] = (
                self._ai_stats["window_frames"] / elapsed
            )
            self._ai_stats["window_frames"] = 0
            self._ai_stats["window_started"] = now
        self._ai_stats["active_person_tracks"] = len(self._person_tracks)
        self._ai_stats["track_candidate_count"] = sum(
            len(track["candidates"]) for track in self._person_tracks.values()
        )
        return Gst.PadProbeReturn.OK

    def _pose_tensor_probe(
        self,
        _pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        """Read only the small 300x57 output tensor before tracker ownership."""
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        import pyds

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK
        frame_list = batch_meta.frame_meta_list
        while frame_list is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            user_list = frame_meta.frame_user_meta_list
            while user_list is not None:
                user_meta = pyds.NvDsUserMeta.cast(user_list.data)
                if (
                    user_meta.base_meta.meta_type
                    == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(
                        user_meta.user_meta_data
                    )
                    for index in range(int(tensor_meta.num_output_layers)):
                        layer = tensor_meta.output_layers_info(index)
                        if str(layer.layerName or "") != "output0":
                            continue
                        dims = layer.inferDims
                        shape = [
                            int(dims.d[dim])
                            for dim in range(int(dims.numDims))
                        ]
                        if shape != [300, 57]:
                            continue
                        try:
                            address = pyds.get_ptr(layer.buffer)
                        except (TypeError, ValueError):
                            continue
                        if address == 0:
                            continue
                        try:
                            values = ctypes.cast(
                                address,
                                ctypes.POINTER(ctypes.c_float * (300 * 57)),
                            ).contents
                        except (TypeError, ValueError):
                            continue
                        detections = sum(
                            1
                            for row in range(300)
                            if float(values[row * 57 + 4]) >= 0.25
                            and int(round(float(values[row * 57 + 5]))) == 0
                        )
                        self._ai_stats[
                            "pose_keypoints_extracted_total"
                        ] += detections * 17
                try:
                    user_list = user_list.next
                except StopIteration:
                    break
            try:
                frame_list = frame_list.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def _source_fps_probe(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        source_index: int,
    ) -> Gst.PadProbeReturn:
        # This counts GstBuffer references only. It never maps, downloads, or
        # reads image pixels from the NVMM/CUDA surface.
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        now = time.monotonic()
        with self._fps_lock:
            stats = self._source_stats[source_index]
            pts = int(buffer.pts)
            if (
                pts != int(Gst.CLOCK_TIME_NONE)
                and stats["last_pts"] >= 0
                and pts < stats["last_pts"]
            ):
                stats["loop_count"] += 1
            if pts != int(Gst.CLOCK_TIME_NONE):
                stats["last_pts"] = pts
            if not stats["width"] or not stats["height"]:
                caps = pad.get_current_caps()
                if caps is not None and caps.get_size() > 0:
                    structure = caps.get_structure(0)
                    try:
                        stats["width"] = int(
                            structure.get_value("width") or 0
                        )
                        stats["height"] = int(
                            structure.get_value("height") or 0
                        )
                    except (ValueError, TypeError):
                        pass
            stats["frames"] += 1
            stats["window_frames"] += 1
            stats["last_frame_at"] = now
            elapsed = now - stats["window_started"]
            if elapsed >= 1.0:
                measured_fps = stats["window_frames"] / elapsed
                # Smooth network-arrival jitter while preserving the measured
                # source rate. This is still based on real decoded buffers.
                stats["fps"] = (
                    measured_fps
                    if stats["fps"] <= 0
                    else (stats["fps"] * 0.7) + (measured_fps * 0.3)
                )
                stats["window_frames"] = 0
                stats["window_started"] = now
        return Gst.PadProbeReturn.OK

    def _on_bus_message(
        self,
        _bus: Gst.Bus,
        message: Gst.Message,
    ) -> None:
        if message.type == Gst.MessageType.EOS:
            if self.settings.loop_files and self.pipeline is not None:
                print("End of file set; restarting from the beginning")
                flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT
                if self.pipeline.seek_simple(Gst.Format.TIME, flags, 0):
                    return
                print("Could not restart file sources", file=sys.stderr)
            print("End of stream")
            if self.loop is not None:
                self.loop.quit()
            return

        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            source_path = message.src.get_path_string()
            print(
                f"GStreamer error from {message.src.get_name()}: "
                f"{error.message}",
                file=sys.stderr,
            )
            if debug:
                print(f"Debug details: {debug}", file=sys.stderr)
            # A live camera is a fault domain. nvurisrcbin owns its reconnect
            # cycle; never terminate healthy cameras because one RTSP source
            # disappears or produces a malformed packet.
            if (
                "source-bin-" in source_path
                or "uri-decode-bin-" in source_path
            ):
                print(
                    "Isolated RTSP source failure; keeping pipeline alive",
                    file=sys.stderr,
                )
                return
            if self.loop is not None:
                self.loop.quit()
            return

        if message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print(
                f"GStreamer warning from {message.src.get_name()}: "
                f"{warning.message}",
                file=sys.stderr,
            )
            if debug:
                print(f"Debug details: {debug}", file=sys.stderr)

    def build(self) -> None:
        Gst.init(None)

        number_sources = len(self.settings.uris)
        if number_sources < 1:
            raise ValueError("At least one RTSP or file source is required")
        decode_only = (
            os.getenv("BENCHMARK_PIPELINE_MODE", "full") == "decode_only"
        )

        pipeline = Gst.Pipeline.new("low-latency-camera-pipeline")
        if pipeline is None:
            raise RuntimeError("Could not create GStreamer pipeline")

        streammux = self._make("nvstreammux", "stream-muxer")
        tiler = self._make("nvmultistreamtiler", "tiler")
        pre_osd_converter = self._make(
            "nvvideoconvert",
            "pre-osd-converter",
        )
        rgba_capsfilter = self._make(
            "capsfilter",
            "rgba-caps",
        )
        osd = self._make("nvdsosd", "on-screen-display")
        output_queue = self._make("queue", "output-queue")
        encoder_converter = self._make(
            "nvvideoconvert",
            "encoder-converter",
        )
        nv12_capsfilter = self._make(
            "capsfilter",
            "nv12-caps",
        )
        encoder = self._make(
            "nvv4l2h264enc",
            "h264-encoder",
        )
        parser = self._make("h264parse", "h264-parser")
        publisher = self._make(
            "rtspclientsink",
            "mediamtx-publisher",
        )

        self.pipeline = pipeline
        self.streammux = streammux

        has_live_source = any(
            uri.lower().startswith("rtsp://")
            for uri in self.settings.uris
        )

        # nvstreammux: enable live mode only when at least one RTSP source exists.
        self._set_if_supported(
            streammux,
            "batch-size",
            number_sources,
        )
        self._set_if_supported(streammux, "live-source", has_live_source)
        self._set_if_supported(
            streammux,
            "batched-push-timeout",
            self.settings.batch_timeout_us,
        )
        self._set_if_supported(
            streammux,
            "width",
            self.settings.mux_width,
        )
        self._set_if_supported(
            streammux,
            "height",
            self.settings.mux_height,
        )
        self._set_if_supported(streammux, "enable-padding", False)
        self._set_if_supported(streammux, "sync-inputs", False)
        self._set_if_supported(streammux, "gpu-id", self.settings.gpu_id)
        self._set_if_supported(
            streammux,
            "nvbuf-memory-type",
            NVBUF_MEM_CUDA_DEVICE,
        )

        # The source tee preserves each native NVMM surface. The batched AI
        # representation is produced downstream by gst-nvinfer on GPU.
        pipeline.add(streammux)
        analytics_tail: Gst.Element = streammux
        if self._ai_enabled:
            engine_path = Path(
                "/workspace/weights/face_recognition/linux_trt10/"
                "yolo26s-pose_dynamic_b26_trt103.engine"
            )
            infer_config = Path(
                "/workspace/apps/deepstream-imagedata-multistream/"
                "ai/person_nvinfer.txt"
            )
            if not engine_path.is_file():
                raise RuntimeError(
                    f"TensorRT person engine is missing: {engine_path}"
                )
            try:
                import pyds  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "AI Stream requires official pyds for DeepStream 7.1"
                ) from exc

            person_infer = self._make("nvinfer", "person-tensorrt")
            person_infer.set_property("config-file-path", str(infer_config))
            self._set_if_supported(person_infer, "batch-size", number_sources)
            self._set_if_supported(
                person_infer, "gpu-id", self.settings.gpu_id
            )

            tracker = self._make("nvtracker", "person-nvdcf-tracker")
            tracker_config = configparser.ConfigParser()
            tracker_config.read(
                "/workspace/apps/deepstream-imagedata-multistream/"
                "ai/tracker_config.txt"
            )
            tracker_values = tracker_config["tracker"]
            tracker_properties: dict[str, Any] = {
                "tracker-width": tracker_values.getint("tracker-width"),
                "tracker-height": tracker_values.getint("tracker-height"),
                "gpu-id": tracker_values.getint("gpu-id"),
                "ll-lib-file": tracker_values["ll-lib-file"],
                "ll-config-file": tracker_values["ll-config-file"],
                "display-tracking-id": tracker_values.getint(
                    "display-tracking-id"
                ),
                "enable-batch-process": tracker_values.getint(
                    "enable-batch-process"
                ),
                "enable-past-frame": tracker_values.getint(
                    "enable-past-frame"
                ),
            }
            for name, value in tracker_properties.items():
                self._set_if_supported(tracker, name, value)
            pipeline.add(person_infer)
            pipeline.add(tracker)
            self._link_many([streammux, person_infer, tracker])
            infer_src = person_infer.get_static_pad("src")
            if infer_src is None:
                raise RuntimeError("Could not get person nvinfer src pad")
            infer_src.add_probe(
                Gst.PadProbeType.BUFFER,
                self._pose_tensor_probe,
            )
            tracker_src = tracker.get_static_pad("src")
            if tracker_src is None:
                raise RuntimeError("Could not get NvDCF tracker src pad")
            tracker_src.add_probe(
                Gst.PadProbeType.BUFFER,
                self._ai_metadata_probe,
            )
            analytics_tail = tracker

        if self._ai_enabled:
            batch_tee = self._make("tee", "post-tracker-batch-tee")
            wall_batch_queue = self._make("queue", "wall-batch-queue")
            demux_queue = self._make("queue", "fullscreen-demux-queue")
            demux = self._make("nvstreamdemux", "tracked-source-demux")
            self._configure_leaky_queue(wall_batch_queue)
            self._configure_leaky_queue(demux_queue)
            for element in (
                batch_tee, wall_batch_queue, demux_queue, demux
            ):
                pipeline.add(element)
            if not analytics_tail.link(batch_tee):
                raise RuntimeError("Could not link tracker to batch tee")
            if not batch_tee.link(wall_batch_queue):
                raise RuntimeError("Could not link batch tee to Wall")
            if not batch_tee.link(demux_queue) or not demux_queue.link(demux):
                raise RuntimeError("Could not link tracked demux branch")
            analytics_tail = wall_batch_queue

            for source_index in range(number_sources):
                source_tee = self._make(
                    "tee", f"tracked-source-tee-{source_index:02d}"
                )
                idle_queue = self._make(
                    "queue", f"tracked-idle-queue-{source_index:02d}"
                )
                idle_sink = self._make(
                    "fakesink", f"tracked-idle-sink-{source_index:02d}"
                )
                self._configure_leaky_queue(idle_queue)
                self._set_if_supported(idle_sink, "sync", False)
                self._set_if_supported(idle_sink, "async", False)
                pipeline.add(source_tee)
                pipeline.add(idle_queue)
                pipeline.add(idle_sink)
                if not source_tee.link(idle_queue) or not idle_queue.link(
                    idle_sink
                ):
                    raise RuntimeError("Could not link tracked idle branch")
                demux_pad = self._request_pad(
                    demux, f"src_{source_index}"
                )
                tee_sink = source_tee.get_static_pad("sink")
                if (
                    demux_pad is None
                    or tee_sink is None
                    or demux_pad.link(tee_sink) != Gst.PadLinkReturn.OK
                ):
                    raise RuntimeError(
                        f"Could not map demux source {source_index}"
                    )
                self._demux_request_pads.append(demux_pad)
                self._fullscreen_source_tees[source_index] = source_tee

        # Build source tees. One GPU-memory branch feeds analytics/mosaic and
        # the native surface remains available for on-demand fullscreen.
        camera_by_source: dict[int, dict[str, Any]] = {}
        for camera in self.settings.cameras:
            camera_by_source.setdefault(int(camera["source_index"]), camera)

        for index, uri in enumerate(self.settings.uris):
            source_bin = self._create_source_bin(index, uri)
            tee = self._make("tee", f"source-tee-{index:02d}")
            mosaic_queue = self._make("queue", f"mosaic-queue-{index:02d}")
            self._configure_leaky_queue(mosaic_queue)
            pipeline.add(source_bin)
            pipeline.add(tee)
            pipeline.add(mosaic_queue)
            self._source_tees[index] = tee
            emulate_static = (
                uri.lower().startswith("file://")
                and os.getenv("STATIC_CAMERA_EMULATION", "false").lower()
                in {"1", "true", "yes", "on"}
            )
            if emulate_static:
                pacer = self._make("identity", f"source-pacer-{index:02d}")
                self._set_if_supported(pacer, "sync", True)
                self._set_if_supported(pacer, "single-segment", True)
                pipeline.add(pacer)
                if not source_bin.link(pacer) or not pacer.link(tee):
                    raise RuntimeError(
                        f"Could not link paced source {index} to tee"
                    )
            elif not source_bin.link(tee):
                raise RuntimeError(f"Could not link source {index} to tee")
            if not tee.link(mosaic_queue):
                raise RuntimeError(f"Could not link source tee {index} to mosaic")

            sink_pad = self._request_pad(
                streammux,
                f"sink_{index}",
            )
            if sink_pad is None:
                raise RuntimeError(
                    f"Could not request nvstreammux sink_{index}"
                )

            source_pad = mosaic_queue.get_static_pad("src")
            if source_pad is None:
                raise RuntimeError(
                    f"Source bin {index} does not expose a src pad"
                )

            source_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._source_fps_probe,
                index,
            )
            result = source_pad.link(sink_pad)
            if result != Gst.PadLinkReturn.OK:
                raise RuntimeError(
                    f"Could not link source {index} to nvstreammux: "
                    f"{result}"
                )

            self._requested_mux_pads.append(sink_pad)

            camera = camera_by_source.get(index)
            # File-backed simulated cameras are published as compressed H.264
            # passthrough processes. This avoids both CPU codecs and one NVENC
            # session per file while the analysis branch remains GPU-decoded.
            if (
                camera is None
                or decode_only
                or os.getenv(
                    "INDIVIDUAL_STREAMS_ENABLED", "true"
                ).lower() not in {"1", "true", "yes", "on"}
                or int(camera.get("independent_copy", 1)) > 1
                or (
                    uri.lower().startswith("file://")
                    and os.getenv(
                        "STATIC_CAMERA_PASSTHROUGH",
                        "true",
                    ).lower() in {"1", "true", "yes", "on"}
                    and not camera.get("requires_gpu_transcode", False)
                )
            ):
                continue

            individual_queue = self._make(
                "queue",
                f"camera-queue-{index:02d}",
            )
            individual_converter = self._make(
                "nvvideoconvert",
                f"camera-converter-{index:02d}",
            )
            individual_caps = self._make(
                "capsfilter",
                f"camera-nv12-caps-{index:02d}",
            )
            individual_encoder = self._make(
                "nvv4l2h264enc",
                f"camera-encoder-{index:02d}",
            )
            individual_parser = self._make(
                "h264parse",
                f"camera-parser-{index:02d}",
            )
            individual_publisher = self._make(
                "rtspclientsink",
                f"camera-publisher-{index:02d}",
            )
            self._configure_leaky_queue(individual_queue)
            self._set_if_supported(
                individual_converter,
                "gpu-id",
                self.settings.gpu_id,
            )
            self._set_if_supported(
                individual_converter,
                "nvbuf-memory-type",
                NVBUF_MEM_CUDA_DEVICE,
            )
            individual_caps.set_property(
                "caps",
                Gst.Caps.from_string(
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={int(os.getenv('INDIVIDUAL_OUTPUT_WIDTH', '1280'))},"
                    f"height={int(os.getenv('INDIVIDUAL_OUTPUT_HEIGHT', '720'))}"
                ),
            )
            self._configure_h264_encoder(
                individual_encoder,
                max(4_000_000, self.settings.bitrate // 2),
            )
            individual_parser.set_property("config-interval", -1)
            self._configure_rtsp_publisher(
                individual_publisher,
                self._individual_publish_url(str(camera["camera_id"])),
                False,
            )
            individual_elements = [
                individual_queue,
                individual_converter,
                individual_caps,
                individual_encoder,
                individual_parser,
                individual_publisher,
            ]
            for element in individual_elements:
                pipeline.add(element)
            if not tee.link(individual_queue):
                raise RuntimeError(
                    f"Could not link source tee {index} to native stream"
                )

            self._link_many(individual_elements)

        if decode_only:
            sink = self._make("fakesink", "decode-only-sink")
            self._set_if_supported(sink, "sync", False)
            self._set_if_supported(sink, "async", False)
            self._set_if_supported(sink, "qos", False)
            pipeline.add(sink)
            if not analytics_tail.link(sink):
                raise RuntimeError("Could not link decode-only sink")
            bus = pipeline.get_bus()
            if bus is None:
                raise RuntimeError("Could not get pipeline bus")
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            return

        # The browser wall is composed from the independent camera streams.
        # Disabling the encoded mosaic reserves its NVENC session for a camera.
        if os.getenv("MOSAIC_PUBLISH_ENABLED", "true").lower() not in {
            "1", "true", "yes", "on"
        }:
            sink = self._make("fakesink", "mosaic-disabled-sink")
            self._set_if_supported(sink, "sync", False)
            self._set_if_supported(sink, "async", False)
            self._set_if_supported(sink, "qos", False)
            pipeline.add(sink)
            if not analytics_tail.link(sink):
                raise RuntimeError("Could not link mosaic-disabled sink")
            mux_src_pad = analytics_tail.get_static_pad("src")
            if mux_src_pad is None:
                raise RuntimeError("Could not get nvstreammux src pad")
            mux_src_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._processing_probe,
                None,
            )
            bus = pipeline.get_bus()
            if bus is None:
                raise RuntimeError("Could not get pipeline bus")
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)
            return

        # Mosaic layout.
        rows = int(os.getenv("WALL_ROWS", "4"))
        columns = int(os.getenv("WALL_COLUMNS", "6"))

        tiler.set_property("rows", rows)
        tiler.set_property("columns", columns)
        tiler.set_property("width", self.settings.output_width)
        tiler.set_property("height", self.settings.output_height)
        self._set_if_supported(tiler, "gpu-id", self.settings.gpu_id)
        self._set_if_supported(
            tiler,
            "nvbuf-memory-type",
            NVBUF_MEM_CUDA_DEVICE,
        )

        self._set_if_supported(
            pre_osd_converter,
            "gpu-id",
            self.settings.gpu_id,
        )
        self._set_if_supported(
            pre_osd_converter,
            "nvbuf-memory-type",
            NVBUF_MEM_CUDA_DEVICE,
        )

        rgba_capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),format=RGBA"
            ),
        )

        self._set_if_supported(osd, "gpu-id", self.settings.gpu_id)
        self._set_if_supported(osd, "process-mode", 1)

        self._configure_leaky_queue(output_queue)
        output_src_pad = output_queue.get_static_pad("src")
        if output_src_pad is None:
            raise RuntimeError("Could not get Wall output queue src pad")
        output_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._wall_rate_limit_probe,
        )

        self._set_if_supported(
            encoder_converter,
            "gpu-id",
            self.settings.gpu_id,
        )
        self._set_if_supported(
            encoder_converter,
            "nvbuf-memory-type",
            NVBUF_MEM_CUDA_DEVICE,
        )

        nv12_capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                "video/x-raw(memory:NVMM),"
                f"format=NV12,"
                f"width={self.settings.output_width},"
                f"height={self.settings.output_height}"
            ),
        )

        # NVIDIA hardware encoder low-latency settings.
        self._configure_h264_encoder(encoder, self.settings.bitrate)

        parser.set_property("config-interval", -1)

        self._configure_rtsp_publisher(
            publisher,
            self.settings.publish_url,
            not has_live_source,
        )

        elements = [
            tiler,
            pre_osd_converter,
            rgba_capsfilter,
            osd,
            output_queue,
            encoder_converter,
            nv12_capsfilter,
            encoder,
            parser,
            publisher,
        ]

        for element in elements:
            pipeline.add(element)

        self._link_many([analytics_tail, *elements])

        # Count the frames actually delivered by the single Wall NVENC path.
        wall_encoded_pad = parser.get_static_pad("src")
        if wall_encoded_pad is None:
            raise RuntimeError("Could not get Wall parser src pad")
        wall_encoded_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._processing_probe,
            None,
        )

        bus = pipeline.get_bus()
        if bus is None:
            raise RuntimeError("Could not get pipeline bus")

        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

    def run(self) -> None:
        self._check_publish_server()

        if self.pipeline is None:
            self.build()

        assert self.pipeline is not None

        self._start_frontend()
        self._start_file_passthrough_publishers()

        self.loop = GLib.MainLoop()

        print("Starting low-latency DeepStream pipeline")
        print(f"Inputs: {len(self.settings.uris)}")
        print(f"Publishing to: {self.settings.publish_url}")
        print(
            "Browser URL: replace rtsp://HOST:8554/PATH with "
            "http://HOST:8889/PATH"
        )
        if self.settings.web_port > 0:
            print(f"Demo frontend: http://HOST:{self.settings.web_port}/")

        state_result = self.pipeline.set_state(Gst.State.PLAYING)
        if state_result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Pipeline refused PLAYING state")

        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self.pipeline is None:
            return

        print("Stopping pipeline")
        self._event_worker_stop.set()
        self._event_worker.join(timeout=2.0)
        self._file_publisher_stop.set()
        for process in self._file_publisher_processes:
            if process.poll() is None:
                process.terminate()
        self._file_publisher_processes.clear()
        self.pipeline.send_event(Gst.Event.new_eos())
        self.pipeline.set_state(Gst.State.NULL)

        if self._web_server is not None:
            self._web_server.shutdown()
            self._web_server = None

        if self.streammux is not None:
            for pad in self._requested_mux_pads:
                try:
                    self.streammux.release_request_pad(pad)
                except Exception:
                    pass

        self._requested_mux_pads.clear()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(
        description=(
            "Publish a low-latency DeepStream RTSP/local-file mosaic to MediaMTX."
        )
    )
    parser.add_argument(
        "uris",
        nargs="*",
        help="One or more rtsp:// URLs, file:// URLs, or local video paths",
    )
    parser.add_argument(
        "--camera-config",
        help=(
            "JSON file containing camera_id, name, enabled, tasks, "
            "source_uri, frame_width, and frame_height. Source URIs are "
            "never returned by the frontend API."
        ),
    )
    parser.add_argument(
        "--video-folder",
        help="Add every supported video file in this directory as an input",
    )
    parser.add_argument(
        "--publish-url",
        default=os.getenv(
            "MEDIAMTX_PUBLISH_URL",
            "rtsp://mediamtx:8554/deepstream-mosaic",
        ),
        help="MediaMTX RTSP publish URL",
    )
    parser.add_argument(
        "--output-width",
        type=positive_int,
        default=1280,
    )
    parser.add_argument(
        "--output-height",
        type=positive_int,
        default=720,
    )
    parser.add_argument(
        "--mux-width",
        type=positive_int,
        default=1280,
    )
    parser.add_argument(
        "--mux-height",
        type=positive_int,
        default=720,
    )
    parser.add_argument(
        "--bitrate",
        type=positive_int,
        default=2_500_000,
        help="H.264 bitrate in bits per second",
    )
    parser.add_argument(
        "--gop",
        type=positive_int,
        default=15,
        help="I/IDR frame interval",
    )
    parser.add_argument(
        "--rtsp-latency",
        type=int,
        default=50,
        help="RTSP jitter-buffer latency in milliseconds",
    )
    parser.add_argument(
        "--transport",
        choices=sorted(RTSP_TRANSPORTS),
        default="udp",
        help="Camera RTSP transport; UDP is lower latency on a reliable LAN",
    )
    parser.add_argument(
        "--batch-timeout-us",
        type=positive_int,
        default=4000,
        help="nvstreammux batch timeout in microseconds",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--decoder-low-latency",
        action="store_true",
        help=(
            "Enable nvv4l2decoder low-latency-mode for RTSP streams. "
            "Use only when camera streams have no B-frames."
        ),
    )
    parser.add_argument(
        "--loop-files",
        action="store_true",
        help="Restart file inputs when the complete file set reaches EOS",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=7070,
        help="Frontend HTTP port; use 0 to disable (default: 7070)",
    )

    args = parser.parse_args()

    if args.rtsp_latency < 0:
        parser.error("--rtsp-latency cannot be negative")

    requested_sources = list(args.uris)
    cameras: list[dict[str, Any]] = []
    if (
        args.camera_config
        and os.getenv("BENCHMARK_STATIC_ONLY", "false").lower()
        not in {"1", "true", "yes", "on"}
    ):
        config_path = Path(args.camera_config)
        if not config_path.is_file():
            parser.error(f"--camera-config does not exist: {config_path}")
        try:
            camera_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --camera-config: {exc}")
        if not isinstance(camera_config, list):
            parser.error("--camera-config must contain a JSON array")
        rtsp_limit = int(
            os.getenv("RTSP_CAMERA_LIMIT", str(len(camera_config)))
        )
        rtsp_copies = int(os.getenv("RTSP_CAMERA_COPIES", "1"))
        shared_rtsp_ingest = os.getenv(
            "RTSP_SHARED_INGEST", "false"
        ).lower() in {"1", "true", "yes", "on"}
        if rtsp_copies < 1:
            parser.error("RTSP_CAMERA_COPIES must be positive")
        enabled_rtsp_items = [
            (index, item)
            for index, item in enumerate(camera_config[:rtsp_limit])
            if isinstance(item, dict) and item.get("enabled", True)
        ]
        for copy_index in range(rtsp_copies):
          for index, item in enabled_rtsp_items:
            if not isinstance(item, dict):
                parser.error(f"camera entry {index} must be an object")
            source_uri = item.get("source_uri")
            if not isinstance(source_uri, str) or not source_uri:
                parser.error(f"camera entry {index} has no source_uri")
            original_id = str(
                item.get("camera_id", f"camera-{index + 1:02d}")
            )
            original_name = str(item.get("name", f"Camera {index + 1}"))
            if copy_index == 0 or not shared_rtsp_ingest:
                requested_sources.append(source_uri)
                source_index = len(requested_sources) - 1
            else:
                source_index = next(
                    int(camera["source_index"])
                    for camera in cameras
                    if camera.get("physical_camera_id") == original_id
                )
            cameras.append(
                {
                    "camera_id": (
                        f"{original_id}-"
                        f"{chr(ord('a') + copy_index)}"
                    ),
                    "name": (
                        f"{original_name} · "
                        f"{chr(ord('A') + copy_index)}"
                    ),
                    "tasks": item.get("tasks", []),
                    "frame_width": int(item.get("frame_width", 0)),
                    "frame_height": int(item.get("frame_height", 0)),
                    "source_index": source_index,
                    "physical_camera_id": original_id,
                    "independent_copy": copy_index + 1,
                    "stream_camera_id": original_id,
                }
            )
    if args.video_folder:
        folder = Path(args.video_folder)
        if not folder.exists() or not folder.is_dir():
            parser.error(f"--video-folder is not a directory: {folder}")
        supported = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
        video_paths = sorted(
            (
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in supported
            ),
            key=lambda item: str(item.relative_to(folder)).lower(),
        )
        static_emulation = os.getenv(
            "STATIC_CAMERA_EMULATION",
            "false",
        ).lower() in {"1", "true", "yes", "on"}
        static_fps = os.getenv(
            "STATIC_CAMERA_EMULATION_FPS",
            "source",
        ).lower()
        if static_emulation and static_fps != "source":
            parser.error(
                "STATIC_CAMERA_EMULATION_FPS currently supports only 'source'"
            )
        requested_static_count = int(
            os.getenv(
                "STATIC_CAMERA_LIMIT",
                os.getenv(
                    "STATIC_CAMERA_SOURCE_COUNT",
                    str(len(video_paths)),
                ),
            )
        )
        if requested_static_count < 1:
            parser.error("STATIC_CAMERA_SOURCE_COUNT must be positive")
        selected_paths = (
            [
                video_paths[index % len(video_paths)]
                for index in range(requested_static_count)
            ]
            if static_emulation and video_paths
            else video_paths
        )
        for file_index, path in enumerate(selected_paths, start=1):
            source_index = len(requested_sources)
            requested_sources.append(str(path))
            relative_name = str(path.relative_to(folder)).replace("\\", "/")
            generation = 1 + (file_index - 1) // max(1, len(video_paths))
            cameras.append(
                {
                    "camera_id": f"data-camera-{file_index:03d}",
                    "name": (
                        f"Camera Emulator {file_index:03d} · "
                        f"{relative_name}"
                    ),
                    "tasks": [],
                    "frame_width": 0,
                    "frame_height": 0,
                    "source_index": source_index,
                    "simulated": True,
                    "generation": generation,
                    # The supplied root-level 2.mp4 is MPEG-4 Part 2 rather
                    # than H.264, so WebRTC needs a single NVENC transcode.
                    "requires_gpu_transcode": relative_name == "2.mp4",
                }
            )

    if not requested_sources:
        parser.error("provide at least one URI or use --video-folder")

    normalized_sources: list[str] = []
    invalid_sources: list[str] = []

    for source in requested_sources:
        lowered = source.lower()

        if lowered.startswith(("rtsp://", "file://")):
            normalized_sources.append(source)
            continue

        local_path = Path(source)
        if local_path.exists() and local_path.is_file():
            normalized_sources.append(local_path.resolve().as_uri())
            continue

        invalid_sources.append(source)

    if invalid_sources:
        parser.error(
            "Each source must be an rtsp:// URL, file:// URL, or an "
            "existing local video path. Invalid values: "
            + ", ".join(invalid_sources)
            + ". An old output-folder argument such as 'frames1' is not "
              "supported by this publisher."
        )

    return Settings(
        uris=normalized_sources,
        cameras=cameras,
        publish_url=args.publish_url,
        output_width=args.output_width,
        output_height=args.output_height,
        mux_width=args.mux_width,
        mux_height=args.mux_height,
        bitrate=args.bitrate,
        gop=args.gop,
        rtsp_latency_ms=args.rtsp_latency,
        rtsp_transport=args.transport,
        batch_timeout_us=args.batch_timeout_us,
        gpu_id=args.gpu_id,
        decoder_low_latency=args.decoder_low_latency,
        loop_files=args.loop_files,
        web_port=args.web_port,
    )


def main() -> int:
    settings = parse_args()
    publisher = LowLatencyDeepStreamPublisher(settings)

    def stop_handler(_signum: int, _frame: Any) -> None:
        if publisher.loop is not None:
            publisher.loop.quit()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    try:
        publisher.run()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        publisher.close()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
