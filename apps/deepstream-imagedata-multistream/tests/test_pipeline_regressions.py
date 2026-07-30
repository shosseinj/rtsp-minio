from pathlib import Path


PIPELINE = Path(__file__).parents[1] / "deepstream_low_latency_rtsp_and_file.py"
COMPOSE = Path(__file__).parents[3] / "compose.yaml"


def test_live_publishers_are_clock_synchronised():
    source = PIPELINE.read_text(encoding="utf-8")
    assert "self._configure_rtsp_publisher(publisher, fullscreen_url, True)" in source
    assert "self._configure_rtsp_publisher(\n            publisher,\n            self.settings.publish_url,\n            True," in source


def test_gpu_snapshot_capture_is_enabled_for_minio_events():
    source = COMPOSE.read_text(encoding="utf-8")
    assert 'GPU_SNAPSHOT_ENABLED: "true"' in source


def test_fullscreen_switch_reuses_encoder_branch_instead_of_tearing_it_down():
    source = PIPELINE.read_text(encoding="utf-8")
    start = source.index("    def _start_fullscreen_branch(")
    end = source.index("    def _start_frontend(", start)
    body = source[start:end]
    assert "self._stop_fullscreen_branch()" not in body
    assert "fullscreen_encoder_reuse_count" in body


def test_fullscreen_close_does_not_destroy_live_nvidia_branch():
    source = PIPELINE.read_text(encoding="utf-8")
    start = source.index("    def _stop_fullscreen_branch(")
    end = source.index("    def _start_fullscreen_branch(", start)
    body = source[start:end]
    assert "self.pipeline.remove(element)" not in body
    assert "set_state(Gst.State.NULL)" not in body
