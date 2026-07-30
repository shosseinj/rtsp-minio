from app.dashboard.layout import wall_layout


def test_layout_expands_beyond_24_cameras():
    assert wall_layout(26, preferred_columns=6) == (5, 6)


def test_layout_keeps_24_as_six_by_four():
    assert wall_layout(24, preferred_columns=6) == (4, 6)


def test_layout_handles_empty_camera_list():
    assert wall_layout(0, preferred_columns=6) == (1, 1)
